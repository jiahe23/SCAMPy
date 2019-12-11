#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
from thermodynamic_functions cimport  *
from microphysics_functions cimport  *
import cython
cimport Grid
cimport ReferenceState
cimport EDMF_Rain
from Variables cimport GridMeanVariables
from NetCDFIO cimport NetCDFIO_Stats
from EDMF_Environment cimport EnvironmentVariables
from libc.math cimport fmax, fmin

cdef class UpdraftVariable:
    def __init__(self, nu, nz, loc, kind, name, units):
        self.values = np.zeros((nu,nz),dtype=np.double, order='c')
        self.old = np.zeros((nu,nz),dtype=np.double, order='c')  # needed for prognostic updrafts
        self.new = np.zeros((nu,nz),dtype=np.double, order='c') # needed for prognostic updrafts
        self.tendencies = np.zeros((nu,nz),dtype=np.double, order='c')
        self.flux = np.zeros((nu,nz),dtype=np.double, order='c')
        self.bulkvalues = np.zeros((nz,), dtype=np.double, order = 'c')
        if loc != 'half' and loc != 'full':
            print('Invalid location setting for variable! Must be half or full')
        self.loc = loc
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.kind = kind
        self.name = name
        self.units = units

    cpdef set_bcs(self,Grid.Grid Gr):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t start_low = Gr.gw - 1
            Py_ssize_t start_high = Gr.nzg - Gr.gw - 1

        n_updrafts = np.shape(self.values)[0]

        if self.name == 'w':
            for i in xrange(n_updrafts):
                self.values[i,start_high] = 0.0
                self.values[i,start_low] = 0.0
                for k in xrange(1,Gr.gw):
                    self.values[i,start_high+ k] = -self.values[i,start_high - k ]
                    self.values[i,start_low- k] = -self.values[i,start_low + k  ]
        else:
            for k in xrange(Gr.gw):
                for i in xrange(n_updrafts):
                    self.values[i,start_high + k +1] = self.values[i,start_high  - k]
                    self.values[i,start_low - k] = self.values[i,start_low + 1 + k]

        return

cdef class UpdraftVariables:
    def __init__(self, nu, namelist, paramlist, Grid.Grid Gr):
        self.Gr = Gr
        self.n_updrafts = nu
        cdef:
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t i, k

        self.W    = UpdraftVariable(nu, nzg, 'full', 'velocity', 'w','m/s' )

        self.Area = UpdraftVariable(nu, nzg, 'half', 'scalar', 'area_fraction','[-]' )
        self.QT = UpdraftVariable(nu, nzg, 'half', 'scalar', 'qt','kg/kg' )
        self.QL = UpdraftVariable(nu, nzg, 'half', 'scalar', 'ql','kg/kg' )
        self.RH = UpdraftVariable(nu, nzg, 'half', 'scalar', 'RH','%' )

        if namelist['thermodynamics']['thermal_variable'] == 'entropy':
            self.H = UpdraftVariable(nu, nzg, 'half', 'scalar', 's','J/kg/K' )
        elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
            self.H = UpdraftVariable(nu, nzg, 'half', 'scalar', 'thetal','K' )

        self.THL = UpdraftVariable(nu, nzg, 'half', 'scalar', 'thetal', 'K')
        self.T   = UpdraftVariable(nu, nzg, 'half', 'scalar', 'temperature','K' )
        self.B   = UpdraftVariable(nu, nzg, 'half', 'scalar', 'buoyancy','m^2/s^3' )

        if namelist['turbulence']['scheme'] == 'EDMF_PrognosticTKE':
            try:
                use_steady_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['use_steady_updrafts']
            except:
                use_steady_updrafts = False
            if use_steady_updrafts:
                self.prognostic = False
            else:
                self.prognostic = True
            self.updraft_fraction = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']
        else:
            self.prognostic = False
            self.updraft_fraction = paramlist['turbulence']['EDMF_BulkSteady']['surface_area']

        # cloud and rain diagnostics for output
        self.cloud_fraction = np.zeros((nzg,), dtype=np.double, order='c')

        self.cloud_base     = np.zeros((nu,),  dtype=np.double, order='c')
        self.cloud_top      = np.zeros((nu,),  dtype=np.double, order='c')
        self.cloud_cover    = np.zeros((nu,),  dtype=np.double, order='c')
        self.updraft_top    = np.zeros((nu,),  dtype=np.double, order='c')

        self.lwp = 0.
        return

    cpdef initialize(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t gw = self.Gr.gw
            double dz = self.Gr.dz

        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):

                    self.W.values[i,k] = 0.0
                    # Simple treatment for now, revise when multiple updraft closures
                    # become more well defined
                    if self.prognostic:
                        self.Area.values[i,k] = 0.0 #self.updraft_fraction/self.n_updrafts
                    else:
                        self.Area.values[i,k] = self.updraft_fraction/self.n_updrafts
                    self.QT.values[i,k] = GMV.QT.values[k]
                    self.QL.values[i,k] = GMV.QL.values[k]
                    self.H.values[i,k]  = GMV.H.values[k]
                    self.T.values[i,k]  = GMV.T.values[k]
                    self.B.values[i,k]  = 0.0

                self.Area.values[i,gw] = self.updraft_fraction/self.n_updrafts

        self.QT.set_bcs(self.Gr)
        self.H.set_bcs(self.Gr)

        return

    cpdef initialize_bubble(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t gw = self.Gr.gw
            double dz = self.Gr.dz

        z_in = np.array([
                          75.,  125.,  175.,  225.,  275.,  325.,  375.,  425.,  475.,
                         525.,  575.,  625.,  675.,  725.,  775.,  825.,  875.,  925.,
                         975., 1025., 1075., 1125., 1175., 1225., 1275., 1325., 1375.,
                        1425., 1475., 1525., 1575., 1625., 1675., 1725., 1775., 1825.,
                        1875., 1925., 1975., 2025., 2075., 2125., 2175., 2225., 2275.,
                        2325., 2375., 2425., 2475., 2525., 2575., 2625., 2675., 2725.,
                        2775., 2825., 2875., 2925., 2975., 3025., 3075., 3125., 3175.,
                        3225., 3275., 3325., 3375., 3425., 3475., 3525., 3575., 3625.,
                        3675., 3725., 3775., 3825., 3875., 3925.
        ])
        thetal_in = np.array([
                        295.65, 295.65, 295.65, 295.65, 295.66, 295.67, 295.69,
                        295.71, 295.73, 295.76, 295.79, 295.83, 295.87, 295.91,
                        295.95, 296.00, 296.04, 296.09, 296.15, 296.20, 296.25,
                        296.30, 296.35, 296.40, 296.45, 296.50, 296.54, 296.55,
                        296.59, 296.62, 296.65, 296.68, 296.70, 296.72, 296.73,
                        296.74, 296.74, 296.74, 296.73, 296.72, 296.70, 296.68,
                        296.65, 296.61, 296.58, 296.53, 296.49, 296.44, 296.38,
                        296.32, 296.26, 296.19, 296.12, 296.05, 295.98, 295.90,
                        295.83, 295.75, 295.67, 295.59, 295.52, 295.44, 295.35,
                        295.28, 295.21, 295.14, 295.08, 295.03, 294.97, 294.92,
                        294.87, 294.82, 294.78, 294.74, 294.70, 294.67, 294.64,
                        294.61
        ])
        Area_in = np.array([
                        0.025, 0.05 , 0.065, 0.075, 0.085, 0.09 , 0.095, 0.105,
                        0.11 , 0.11 , 0.11 , 0.11 , 0.11 , 0.11 , 0.115, 0.115, 0.115,
                        0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
                        0.115, 0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 ,
                        0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 ,
                        0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 ,
                        0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 , 0.12 ,
                        0.125, 0.125, 0.125, 0.125, 0.12 , 0.115, 0.115, 0.11 , 0.105,
                        0.1  , 0.09 , 0.085, 0.075, 0.065, 0.05 , 0.03
        ])

        W_in = np.array([
                        0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03,
                        0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.05,
                        0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.05, 0.06, 0.06,
                        0.06, 0.06, 0.06, 0.06, 0.06, 0.07, 0.07, 0.06, 0.07,
                        0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07,
                        0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07,
                        0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.06,
                        0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
                        0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06
        ])

        B_in = np.array([
                        0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.0009,
                        0.0012, 0.0015, 0.0019, 0.0023, 0.0027, 0.0031, 0.0035,
                        0.0041, 0.0046, 0.0052, 0.0056, 0.0063, 0.0066, 0.0073,
                        0.0077, 0.0083, 0.009, 0.0093, 0.0099, 0.0105, 0.011,
                        0.0112, 0.0117, 0.0121, 0.0126, 0.013, 0.0133, 0.0132,
                        0.0135, 0.0137, 0.0139, 0.014, 0.0141, 0.0141, 0.0141,
                        0.014, 0.0139, 0.0137, 0.0135, 0.0137, 0.0134, 0.013,
                        0.0126, 0.0122, 0.0117, 0.0116, 0.0111, 0.0105, 0.0099,
                        0.0096, 0.009, 0.0083, 0.0079, 0.0073, 0.0068, 0.0062,
                        0.0055, 0.005, 0.0046, 0.004, 0.0035, 0.0031, 0.0025,
                        0.0021, 0.0017, 0.0014, 0.0011, 0.0008, 0.0005, 0.0003,
                        0.0002, 0.0001
        ])

        T_in = np.array([
                    297.03, 296.84, 296.65, 296.46, 296.28, 296.09,
                    295.9, 295.72, 295.53, 295.35, 295.16, 294.98, 294.79,
                    294.61, 294.43, 294.24, 294.06, 293.88, 293.69, 293.51,
                    293.32, 293.13, 292.95, 292.76, 292.57, 292.38, 292.2,
                    292.0, 291.81, 291.62, 291.43, 291.23, 291.04, 290.83,
                    290.64, 290.44, 290.24, 290.03, 289.83, 289.62, 289.42,
                    289.21, 289.0, 288.79, 288.57, 288.37, 288.15, 287.93,
                    287.71, 287.49, 287.27, 287.05, 286.83, 286.6, 286.37,
                    286.15, 285.92, 285.69, 285.46, 285.23, 285.0, 284.77,
                    284.54, 284.31, 284.08, 283.84, 283.61, 283.38, 283.15,
                    282.92, 282.69, 282.46, 282.23, 282.0, 281.77, 281.54,
                    281.31, 281.08
        ])

        QL_in = np.array([
                        0.0008, 0.001, 0.0011, 0.0012, 0.0013, 0.0014,
                        0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.002, 0.0021,
                        0.0022, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028,
                        0.0029, 0.003, 0.0031, 0.0032, 0.0033, 0.0034, 0.0035,
                        0.0036, 0.0037, 0.0038, 0.0039, 0.004, 0.0041, 0.0042,
                        0.0043, 0.0045, 0.0046, 0.0047, 0.0048, 0.0049, 0.005,
                        0.0051, 0.0052, 0.0053, 0.0055, 0.0056, 0.0057, 0.0058,
                        0.0059, 0.006, 0.0061, 0.0063, 0.0064, 0.0065, 0.0066,
                        0.0067, 0.0068, 0.007, 0.0071, 0.0072, 0.0073, 0.0074,
                        0.0076, 0.0077, 0.0078, 0.0079, 0.008, 0.0081, 0.0082,
                        0.0083, 0.0085, 0.0086, 0.0087, 0.0088, 0.0089, 0.009,
                        0.0091, 0.0092
        ])

        # with nogil:
        for i in xrange(self.n_updrafts):
            for k in xrange(self.Gr.nzg):

                self.W.values[i,k] = np.interp(self.Gr.z_half[k+self.Gr.gw],z_in,W_in)
                # Simple treatment for now, revise when multiple updraft closures
                # become more well defined
                self.Area.values[i,k] = np.interp(self.Gr.z_half[k+gw],z_in,Area_in) #self.updraft_fraction/self.n_updrafts
                self.QT.values[i,k] = 0.0196
                self.QL.values[i,k] = np.interp(self.Gr.z_half[k+gw],z_in,QL_in)
                self.H.values[i,k]  = np.interp(self.Gr.z_half[k+gw],z_in,thetal_in)
                self.T.values[i,k]  = np.interp(self.Gr.z_half[k+gw],z_in,T_in)
                self.B.values[i,k]  = np.interp(self.Gr.z_half[k+gw],z_in,B_in)

        self.QT.set_bcs(self.Gr)
        self.H.set_bcs(self.Gr)

        return


    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('updraft_area')
        Stats.add_profile('updraft_w')
        Stats.add_profile('updraft_qt')
        Stats.add_profile('updraft_ql')
        Stats.add_profile('updraft_RH')

        if self.H.name == 'thetal':
            Stats.add_profile('updraft_thetal')
        else:
            # Stats.add_profile('updraft_thetal')
            Stats.add_profile('updraft_s')

        Stats.add_profile('updraft_temperature')
        Stats.add_profile('updraft_buoyancy')

        Stats.add_profile('updraft_cloud_fraction')

        Stats.add_ts('updraft_cloud_cover')
        Stats.add_ts('updraft_cloud_base')
        Stats.add_ts('updraft_cloud_top')
        Stats.add_ts('updraft_lwp')
        return

    cpdef set_means(self, GridMeanVariables GMV):

        cdef:
            Py_ssize_t i, k

        self.Area.bulkvalues = np.sum(self.Area.values,axis=0)
        self.W.bulkvalues[:] = 0.0
        self.QT.bulkvalues[:] = 0.0
        self.QL.bulkvalues[:] = 0.0
        self.H.bulkvalues[:] = 0.0
        self.T.bulkvalues[:] = 0.0
        self.B.bulkvalues[:] = 0.0
        self.RH.bulkvalues[:] = 0.0

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if self.Area.bulkvalues[k] > 1.0e-20:
                    for i in xrange(self.n_updrafts):
                        self.QT.bulkvalues[k] += self.Area.values[i,k] * self.QT.values[i,k]/self.Area.bulkvalues[k]
                        self.QL.bulkvalues[k] += self.Area.values[i,k] * self.QL.values[i,k]/self.Area.bulkvalues[k]
                        self.H.bulkvalues[k] += self.Area.values[i,k] * self.H.values[i,k]/self.Area.bulkvalues[k]
                        self.T.bulkvalues[k] += self.Area.values[i,k] * self.T.values[i,k]/self.Area.bulkvalues[k]
                        self.RH.bulkvalues[k] += self.Area.values[i,k] * self.RH.values[i,k]/self.Area.bulkvalues[k]
                        self.B.bulkvalues[k] += self.Area.values[i,k] * self.B.values[i,k]/self.Area.bulkvalues[k]
                        self.W.bulkvalues[k] += ((self.Area.values[i,k] + self.Area.values[i,k+1]) * self.W.values[i,k]
                                            /(self.Area.bulkvalues[k] + self.Area.bulkvalues[k+1]))

                else:
                    self.QT.bulkvalues[k] = GMV.QT.values[k]
                    self.QL.bulkvalues[k] = 0.0
                    self.H.bulkvalues[k] = GMV.H.values[k]
                    self.RH.bulkvalues[k] = GMV.RH.values[k]
                    self.T.bulkvalues[k] = GMV.T.values[k]
                    self.B.bulkvalues[k] = 0.0
                    self.W.bulkvalues[k] = 0.0

                if self.QL.bulkvalues[k] > 1e-8 and self.Area.bulkvalues[k] > 1e-3:
                    self.cloud_fraction[k] = 1.0
                else:
                    self.cloud_fraction[k] = 0.
        return

    # quick utility to set "new" arrays with values in the "values" arrays
    cpdef set_new_with_values(self):
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.new[i,k] = self.W.values[i,k]
                    self.Area.new[i,k] = self.Area.values[i,k]
                    self.QT.new[i,k] = self.QT.values[i,k]
                    self.QL.new[i,k] = self.QL.values[i,k]
                    self.H.new[i,k] = self.H.values[i,k]
                    self.THL.new[i,k] = self.THL.values[i,k]
                    self.T.new[i,k] = self.T.values[i,k]
                    self.B.new[i,k] = self.B.values[i,k]
        return

    # quick utility to set "new" arrays with values in the "values" arrays
    cpdef set_old_with_values(self):
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.old[i,k] = self.W.values[i,k]
                    self.Area.old[i,k] = self.Area.values[i,k]
                    self.QT.old[i,k] = self.QT.values[i,k]
                    self.QL.old[i,k] = self.QL.values[i,k]
                    self.H.old[i,k] = self.H.values[i,k]
                    self.THL.old[i,k] = self.THL.values[i,k]
                    self.T.old[i,k] = self.T.values[i,k]
                    self.B.old[i,k] = self.B.values[i,k]
        return

    # quick utility to set "tmp" arrays with values in the "new" arrays
    cpdef set_values_with_new(self):
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.values[i,k] = self.W.new[i,k]
                    self.Area.values[i,k] = self.Area.new[i,k]
                    self.QT.values[i,k] = self.QT.new[i,k]
                    self.QL.values[i,k] = self.QL.new[i,k]
                    self.H.values[i,k] = self.H.new[i,k]
                    self.THL.values[i,k] = self.THL.new[i,k]
                    self.T.values[i,k] = self.T.new[i,k]
                    self.B.values[i,k] = self.B.new[i,k]
        return

    cpdef io(self, NetCDFIO_Stats Stats, ReferenceState.ReferenceState Ref):

        Stats.write_profile('updraft_area', self.Area.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_w', self.W.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_qt', self.QT.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_ql', self.QL.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_RH', self.RH.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        if self.H.name == 'thetal':
            Stats.write_profile('updraft_thetal', self.H.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        else:
            Stats.write_profile('updraft_s', self.H.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            #Stats.write_profile('updraft_thetal', self.THL.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        Stats.write_profile('updraft_temperature', self.T.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_buoyancy', self.B.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        self.upd_cloud_diagnostics(Ref)
        Stats.write_profile('updraft_cloud_fraction', self.cloud_fraction[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        # Note definition of cloud cover : each updraft is associated with a cloud cover equal to the maximum
        # area fraction of the updraft where ql > 0. Each updraft is assumed to have maximum overlap with respect to
        # itself (i.e. no consideration of tilting due to shear) while the updraft classes are assumed to have no overlap
        # at all. Thus total updraft cover is the sum of each updraft's cover
        Stats.write_ts('updraft_cloud_cover', np.sum(self.cloud_cover))
        Stats.write_ts('updraft_cloud_base',  np.amin(self.cloud_base))
        Stats.write_ts('updraft_cloud_top',   np.amax(self.cloud_top))
        Stats.write_ts('updraft_lwp',         self.lwp)
        return

    cpdef upd_cloud_diagnostics(self, ReferenceState.ReferenceState Ref):
        cdef Py_ssize_t i, k
        self.lwp = 0.

        for i in xrange(self.n_updrafts):
            #TODO check the setting of ghost point z_half

            self.cloud_base[i] = self.Gr.z_half[self.Gr.nzg-self.Gr.gw-1]
            self.cloud_top[i] = 0.0
            self.updraft_top[i] = 0.0
            self.cloud_cover[i] = 0.0

            for k in xrange(self.Gr.gw,self.Gr.nzg-self.Gr.gw):

                if self.Area.values[i,k] > 1e-3:
                    self.updraft_top[i] = fmax(self.updraft_top[i], self.Gr.z_half[k])
                    self.lwp += Ref.rho0_half[k] * self.QL.values[i,k] * self.Area.values[i,k] * self.Gr.dz

                    if self.QL.values[i,k] > 1e-8:
                        self.cloud_base[i]  = fmin(self.cloud_base[i],  self.Gr.z_half[k])
                        self.cloud_top[i]   = fmax(self.cloud_top[i],   self.Gr.z_half[k])
                        self.cloud_cover[i] = fmax(self.cloud_cover[i], self.Area.values[i,k])

        return


cdef class UpdraftThermodynamics:
    def __init__(self, n_updraft, Grid.Grid Gr,
                 ReferenceState.ReferenceState Ref, UpdraftVariables UpdVar,
                 RainVariables Rain):
        self.Gr = Gr
        self.Ref = Ref
        self.n_updraft = n_updraft

        if UpdVar.H.name == 's':
            self.t_to_prog_fp = t_to_entropy_c
            self.prog_to_t_fp = eos_first_guess_entropy
        elif UpdVar.H.name == 'thetal':
            self.t_to_prog_fp = t_to_thetali_c
            self.prog_to_t_fp = eos_first_guess_thetal

        # rain source from each updraft from all sub-timesteps
        self.prec_source_h  = np.zeros((n_updraft, Gr.nzg), dtype=np.double, order='c')
        self.prec_source_qt = np.zeros((n_updraft, Gr.nzg), dtype=np.double, order='c')

        # rain source from all updrafts from all sub-timesteps
        self.prec_source_h_tot  = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.prec_source_qt_tot = np.zeros((Gr.nzg,), dtype=np.double, order='c')

        return

    cpdef clear_precip_sources(self):
        """
        clear precipitation source terms for QT and H from each updraft
        """
        self.prec_source_qt[:,:] = 0.
        self.prec_source_h[:,:]  = 0.
        return

    cpdef update_total_precip_sources(self):
        """
        sum precipitation source terms for QT and H from all sub-timesteps
        """
        self.prec_source_h_tot  = np.sum(self.prec_source_h,  axis=0)
        self.prec_source_qt_tot = np.sum(self.prec_source_qt, axis=0)
        return

    cpdef buoyancy(self, UpdraftVariables UpdVar, EnvironmentVariables EnvVar,
                   GridMeanVariables GMV, bint extrap):
        cdef:
            Py_ssize_t k, i
            double alpha, qv, qt, t, h
            Py_ssize_t gw = self.Gr.gw

        UpdVar.Area.bulkvalues = np.sum(UpdVar.Area.values,axis=0)

        if not extrap:
            with nogil:
                for i in xrange(self.n_updraft):
                    for k in xrange(self.Gr.nzg):
                        if UpdVar.Area.values[i,k] > 0.0:
                            qv = UpdVar.QT.values[i,k] - UpdVar.QL.values[i,k]
                            alpha = alpha_c(self.Ref.p0_half[k], UpdVar.T.values[i,k], UpdVar.QT.values[i,k], qv)
                            UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha) #- GMV.B.values[k]
                        else:
                            UpdVar.B.values[i,k] = EnvVar.B.values[k]
                        UpdVar.RH.values[i,k] = relative_humidity_c(self.Ref.p0_half[k], UpdVar.QT.values[i,k],
                                                    UpdVar.QL.values[i,k], 0.0, UpdVar.T.values[i,k])
        else:
            with nogil:
                for i in xrange(self.n_updraft):
                    for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                        if UpdVar.Area.values[i,k] > 0.0:
                            qt = UpdVar.QT.values[i,k]
                            qv = UpdVar.QT.values[i,k] - UpdVar.QL.values[i,k]
                            h = UpdVar.H.values[i,k]
                            t = UpdVar.T.values[i,k]
                            alpha = alpha_c(self.Ref.p0_half[k], t, qt, qv)
                            UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)
                            UpdVar.RH.values[i,k] = relative_humidity_c(self.Ref.p0_half[k], qt, qt-qv, 0.0, t)

                        elif UpdVar.Area.values[i,k-1] > 0.0 and k>self.Gr.gw:
                            sa = eos(self.t_to_prog_fp, self.prog_to_t_fp, self.Ref.p0_half[k],
                                     qt, h)
                            qt -= sa.ql
                            qv = qt
                            t = sa.T
                            alpha = alpha_c(self.Ref.p0_half[k], t, qt, qv)
                            UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)
                            UpdVar.RH.values[i,k] = relative_humidity_c(self.Ref.p0_half[k], qt, qt-qv, 0.0, t)
                        else:
                            UpdVar.B.values[i,k] = EnvVar.B.values[k]
                            UpdVar.RH.values[i,k] = EnvVar.RH.values[k]


        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                GMV.B.values[k] = (1.0 - UpdVar.Area.bulkvalues[k]) * EnvVar.B.values[k]
                for i in xrange(self.n_updraft):
                    GMV.B.values[k] += UpdVar.Area.values[i,k] * UpdVar.B.values[i,k]
                for i in xrange(self.n_updraft):
                    UpdVar.B.values[i,k] -= GMV.B.values[k]
                EnvVar.B.values[k] -= GMV.B.values[k]

        return

    cpdef microphysics(self, UpdraftVariables UpdVar, RainVariables Rain, double dt):
        """
        compute precipitation source terms
        """
        cdef:
            Py_ssize_t k, i

            rain_struct rst
            mph_struct  mph
            eos_struct  sa

        with nogil:
            for i in xrange(self.n_updraft):
                for k in xrange(self.Gr.nzg):

                    # autoconversion and accretion
                    mph = microphysics_rain_src(
                        Rain.rain_model,
                        UpdVar.QT.new[i,k],
                        UpdVar.QL.new[i,k],
                        Rain.Upd_QR.values[k],
                        UpdVar.Area.new[i,k],
                        UpdVar.T.new[i,k],
                        self.Ref.p0_half[k],
                        self.Ref.rho0_half[k],
                        dt
                    )

                    # update Updraft.new
                    UpdVar.QT.new[i,k] = mph.qt
                    UpdVar.QL.new[i,k] = mph.ql
                    UpdVar.H.new[i,k]  = mph.thl

                    # update rain sources of state variables
                    self.prec_source_qt[i,k] -= mph.qr_src * UpdVar.Area.new[i,k]
                    self.prec_source_h[i,k]  += mph.thl_rain_src * UpdVar.Area.new[i,k]
        return
