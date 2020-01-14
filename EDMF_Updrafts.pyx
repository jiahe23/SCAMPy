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

        # using LES diagnostics at simul time 10sec as input; updraft identification varies in critera
        # w>0.01 or b>1e-4 or (w>0.01 & b>1e-4)

        # # criterion 1: w>0.01
        # z_in = np.array([
        #                  125.,  175.,  225.,  275.,  325.,  375.,  425.,  475.,  525.,
        #                  575.,  625.,  675.,  725.,  775.,  825.,  875.,  925.,  975.,
        #                 1025., 1075., 1125., 1175., 1225., 1275., 1325., 1375., 1425.,
        #                 1475., 1525., 1575., 1625., 1675., 1725., 1775., 1825., 1875.,
        #                 1925., 1975., 2025., 2075., 2125., 2175., 2225., 2275., 2325.,
        #                 2375., 2425., 2475., 2525., 2575., 2625., 2675., 2725., 2775.,
        #                 2825., 2875., 2925., 2975., 3025., 3075., 3125., 3175., 3225.,
        #                 3275., 3325., 3375., 3425., 3475., 3525., 3575., 3625., 3675.,
        #                 3725., 3775., 3825., 3875., 3925., 3975., 4025., 4075., 4125.,
        #                 4175., 4225., 4275., 4325., 4375., 4425., 4475., 4525., 4575.,
        #                 4625., 4675., 4725., 4775., 4825., 4875., 4925., 4975., 5025.,
        #                 5075., 5125., 5175., 5225., 5275., 5325., 5375., 5425., 5475.,
        #                 5525., 5575., 5625., 5675., 5725., 5775., 5825., 5875., 5925.,
        #                 5975., 6025., 6075., 6125., 6175.
        # ])
        # thetal_in = np.array([
        #                 295.6471, 295.6477, 295.6551, 295.6632, 295.6783, 295.7006,
        #                 295.7226, 295.754 , 295.7804, 295.8182, 295.8498, 295.8935,
        #                 295.9399, 295.9891, 296.0253, 296.0767, 296.13  , 296.1836,
        #                 296.2376, 296.2922, 296.3237, 296.3753, 296.4263, 296.4748,
        #                 296.5214, 296.5663, 296.6074, 296.6454, 296.681 , 296.7118,
        #                 296.7078, 296.7301, 296.7481, 296.7613, 296.7688, 296.7727,
        #                 296.7706, 296.7632, 296.7505, 296.7326, 296.7095, 296.6811,
        #                 296.6478, 296.6094, 296.5662, 296.5185, 296.4663, 296.4101,
        #                 296.35  , 296.2862, 296.2195, 296.1495, 296.0773, 296.0028,
        #                 295.9265, 295.8491, 295.7704, 295.6917, 295.6125, 295.534 ,
        #                 295.456 , 295.3792, 295.3044, 295.2309, 295.1608, 295.0926,
        #                 295.0281, 294.9671, 294.9094, 294.8516, 294.8034, 294.7607,
        #                 294.7201, 294.6862, 294.6565, 294.6294, 294.6066, 294.585 ,
        #                 294.5658, 294.5461, 294.5257, 294.5059, 294.4849, 294.4647,
        #                 294.444 , 294.4227, 294.4017, 294.3798, 294.3586, 294.336 ,
        #                 294.3144, 294.2922, 294.2694, 294.2469, 294.2235, 294.2007,
        #                 294.1766, 294.1534, 294.1294, 294.1052, 294.0809, 294.056 ,
        #                 294.0314, 294.0058, 293.9808, 293.9546, 293.9291, 293.9028,
        #                 293.8765, 293.8497, 293.8227, 293.7955, 293.7678, 293.7402,
        #                 293.7118, 293.6837, 293.6547, 293.6261, 293.5965, 293.5673,
        #                 293.5373, 293.5073
        # ])
        # Area_in = np.array([
        #                 0.04 , 0.06 , 0.07 , 0.08 , 0.085, 0.085, 0.09 , 0.09 , 0.095,
        #                 0.095, 0.1  , 0.1  , 0.1  , 0.1  , 0.105, 0.105, 0.105, 0.105,
        #                 0.105, 0.105, 0.11 , 0.11 , 0.11 , 0.11 , 0.11 , 0.11 , 0.11 ,
        #                 0.11 , 0.11 , 0.11 , 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
        #                 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
        #                 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
        #                 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
        #                 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.12 , 0.12 , 0.12 ,
        #                 0.125, 0.125, 0.125, 0.13 , 0.13 , 0.13 , 0.13 , 0.135, 0.135,
        #                 0.135, 0.135, 0.135, 0.14 , 0.14 , 0.14 , 0.14 , 0.14 , 0.14 ,
        #                 0.14 , 0.14 , 0.14 , 0.14 , 0.14 , 0.135, 0.135, 0.135, 0.135,
        #                 0.135, 0.13 , 0.13 , 0.13 , 0.125, 0.125, 0.125, 0.12 , 0.12 ,
        #                 0.115, 0.11 , 0.11 , 0.105, 0.1  , 0.095, 0.09 , 0.085, 0.08 ,
        #                 0.075, 0.065, 0.06 , 0.05 , 0.035
        # ])
        #
        # W_in = np.array([
        #                 0.0116, 0.0148, 0.0181, 0.0209, 0.024 , 0.028 , 0.0308, 0.0347,
        #                 0.0372, 0.0411, 0.0433, 0.0471, 0.0509, 0.0547, 0.0562, 0.0598,
        #                 0.0634, 0.067 , 0.0706, 0.074 , 0.0744, 0.0776, 0.0807, 0.0837,
        #                 0.0867, 0.0895, 0.0922, 0.0948, 0.0972, 0.0994, 0.0975, 0.0994,
        #                 0.101 , 0.1025, 0.1038, 0.1049, 0.1059, 0.1066, 0.1071, 0.1075,
        #                 0.1076, 0.1075, 0.1073, 0.1068, 0.1062, 0.1054, 0.1044, 0.1032,
        #                 0.1019, 0.1004, 0.0987, 0.0969, 0.095 , 0.0929, 0.0908, 0.0885,
        #                 0.0862, 0.0838, 0.0813, 0.0787, 0.0762, 0.0735, 0.0709, 0.0683,
        #                 0.0657, 0.0632, 0.0606, 0.0582, 0.0558, 0.0517, 0.0496, 0.0476,
        #                 0.0443, 0.0426, 0.0409, 0.0383, 0.0369, 0.0355, 0.0342, 0.0321,
        #                 0.031 , 0.0299, 0.0289, 0.0279, 0.0264, 0.0255, 0.0247, 0.0239,
        #                 0.0231, 0.0224, 0.0217, 0.021 , 0.0203, 0.0197, 0.0191, 0.0188,
        #                 0.0182, 0.0177, 0.0171, 0.0166, 0.0164, 0.0159, 0.0154, 0.0152,
        #                 0.0147, 0.0143, 0.014 , 0.0136, 0.0134, 0.0131, 0.0128, 0.0125,
        #                 0.0123, 0.012 , 0.0118, 0.0115, 0.0113, 0.011 , 0.0108, 0.0106,
        #                 0.0104, 0.0102
        # ])
        #
        # B_in = np.array([
        #                 0.0002, 0.0003, 0.0005, 0.0007, 0.001 , 0.0014, 0.0017, 0.0022,
        #                 0.0026, 0.0032, 0.0036, 0.0043, 0.0049, 0.0056, 0.0061, 0.0069,
        #                 0.0076, 0.0084, 0.0092, 0.0099, 0.0104, 0.0112, 0.0119, 0.0126,
        #                 0.0133, 0.014 , 0.0146, 0.0152, 0.0157, 0.0163, 0.0164, 0.0168,
        #                 0.0172, 0.0175, 0.0177, 0.018 , 0.0181, 0.0182, 0.0183, 0.0182,
        #                 0.0181, 0.018 , 0.0178, 0.0176, 0.0172, 0.0169, 0.0164, 0.016 ,
        #                 0.0155, 0.0149, 0.0143, 0.0137, 0.013 , 0.0123, 0.0116, 0.0109,
        #                 0.0101, 0.0093, 0.0086, 0.0078, 0.0071, 0.0064, 0.0056, 0.005 ,
        #                 0.0043, 0.0037, 0.0031, 0.0025, 0.002 , 0.0015, 0.0011, 0.0008,
        #                 0.0005, 0.0003, 0.0002, 0.0001, 0.    , 0.    , 0.    , 0.    ,
        #                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
        #                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
        #                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
        #                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
        #                 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
        #                 0.    , 0.
        # ])
        #
        # T_in = np.array([
        #                 296.8382, 296.65  , 296.4624, 296.2763, 296.0902, 295.9057,
        #                 295.7219, 295.5388, 295.3544, 295.1738, 294.9897, 294.8086,
        #                 294.6291, 294.4487, 294.2644, 294.0851, 293.9047, 293.724 ,
        #                 293.5442, 293.3632, 293.1754, 292.9937, 292.8106, 292.6263,
        #                 292.4422, 292.2566, 292.0696, 291.8824, 291.6934, 291.5028,
        #                 291.3031, 291.1096, 290.9147, 290.7186, 290.5201, 290.3205,
        #                 290.1192, 289.9154, 289.7105, 289.5035, 289.2943, 289.0839,
        #                 288.8709, 288.6567, 288.4407, 288.2223, 288.003 , 287.7812,
        #                 287.5585, 287.3341, 287.1078, 286.881 , 286.6518, 286.4225,
        #                 286.1914, 285.9596, 285.7274, 285.4934, 285.2602, 285.025 ,
        #                 284.7908, 284.5555, 284.3203, 284.0858, 283.8502, 283.6166,
        #                 283.3813, 283.1483, 282.9148, 282.6808, 282.4502, 282.2191,
        #                 281.9903, 281.7608, 281.5335, 281.3063, 281.0793, 280.854 ,
        #                 280.6264, 280.4   , 280.1718, 279.9435, 279.7151, 279.4848,
        #                 279.2557, 279.0239, 278.7928, 278.5606, 278.3275, 278.0949,
        #                 277.8598, 277.626 , 277.3896, 277.1538, 276.9169, 276.679 ,
        #                 276.4414, 276.2016, 275.9629, 275.7214, 275.4807, 275.2383,
        #                 274.9957, 274.7524, 274.5077, 274.2635, 274.0167, 273.7712,
        #                 273.5227, 273.2751, 273.0255, 272.7759, 272.525 , 272.2734,
        #                 272.0213, 271.7675, 271.5141, 271.2583, 271.0034, 270.7455,
        #                 270.4888, 270.2292
        # ])
        #
        # QL_in = np.array([
        #                 0.001 , 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017,
        #                 0.0018, 0.0019, 0.002 , 0.0021, 0.0022, 0.0023, 0.0024, 0.0025,
        #                 0.0026, 0.0027, 0.0028, 0.0029, 0.003 , 0.0031, 0.0032, 0.0033,
        #                 0.0034, 0.0035, 0.0035, 0.0036, 0.0037, 0.0038, 0.004 , 0.0041,
        #                 0.0042, 0.0043, 0.0044, 0.0045, 0.0046, 0.0047, 0.0048, 0.0049,
        #                 0.005 , 0.0051, 0.0053, 0.0054, 0.0055, 0.0056, 0.0057, 0.0058,
        #                 0.006 , 0.0061, 0.0062, 0.0063, 0.0064, 0.0066, 0.0067, 0.0068,
        #                 0.0069, 0.0071, 0.0072, 0.0073, 0.0074, 0.0075, 0.0077, 0.0078,
        #                 0.0079, 0.008 , 0.0081, 0.0082, 0.0083, 0.0085, 0.0086, 0.0087,
        #                 0.0088, 0.0089, 0.009 , 0.0091, 0.0092, 0.0093, 0.0094, 0.0095,
        #                 0.0096, 0.0097, 0.0098, 0.0099, 0.01  , 0.0101, 0.0102, 0.0103,
        #                 0.0104, 0.0105, 0.0105, 0.0106, 0.0107, 0.0108, 0.0109, 0.011 ,
        #                 0.0111, 0.0112, 0.0113, 0.0114, 0.0115, 0.0116, 0.0117, 0.0117,
        #                 0.0118, 0.0119, 0.012 , 0.0121, 0.0122, 0.0123, 0.0124, 0.0125,
        #                 0.0125, 0.0126, 0.0127, 0.0128, 0.0129, 0.013 , 0.0131, 0.0131,
        #                 0.0132, 0.0133
        # ])

        # # criterion 2: b>1e-4
        # z_in = np.array([
        #                   75.,  125.,  175.,  225.,  275.,  325.,  375.,  425.,  475.,
        #                  525.,  575.,  625.,  675.,  725.,  775.,  825.,  875.,  925.,
        #                  975., 1025., 1075., 1125., 1175., 1225., 1275., 1325., 1375.,
        #                 1425., 1475., 1525., 1575., 1625., 1675., 1725., 1775., 1825.,
        #                 1875., 1925., 1975., 2025., 2075., 2125., 2175., 2225., 2275.,
        #                 2325., 2375., 2425., 2475., 2525., 2575., 2625., 2675., 2725.,
        #                 2775., 2825., 2875., 2925., 2975., 3025., 3075., 3125., 3175.,
        #                 3225., 3275., 3325., 3375., 3425., 3475., 3525., 3575., 3625.,
        #                 3675., 3725., 3775., 3825., 3875., 3925.
        # ])
        # thetal_in = np.array([
        #                 295.6455, 295.6453, 295.6462, 295.6529, 295.6603, 295.6746,
        #                 295.6911, 295.7059, 295.727 , 295.7569, 295.7825, 295.8095,
        #                 295.8373, 295.8774, 295.907 , 295.9505, 295.9799, 296.0255,
        #                 296.0534, 296.0989, 296.1244, 296.1683, 296.212 , 296.2313,
        #                 296.271 , 296.3096, 296.3463, 296.3525, 296.3831, 296.4112,
        #                 296.4356, 296.4567, 296.4745, 296.4567, 296.4662, 296.4706,
        #                 296.4719, 296.4681, 296.4598, 296.447 , 296.4298, 296.4081,
        #                 296.382 , 296.3516, 296.3169, 296.3092, 296.266 , 296.2191,
        #                 296.1686, 296.1149, 296.058 , 296.0253, 295.9624, 295.8974,
        #                 295.8305, 295.785 , 295.7146, 295.6433, 295.5914, 295.5185,
        #                 295.4635, 295.3908, 295.3193, 295.2631, 295.2058, 295.1383,
        #                 295.0826, 295.0281, 294.9671, 294.9159, 294.8673, 294.82  ,
        #                 294.7803, 294.7383, 294.7019, 294.6682, 294.6376, 294.6108
        # ])
        # Area_in = np.array([
        #                 0.025, 0.05 , 0.065, 0.075, 0.085, 0.09 , 0.095, 0.105, 0.11 ,
        #                 0.11 , 0.115, 0.12 , 0.125, 0.125, 0.13 , 0.13 , 0.135, 0.135,
        #                 0.14 , 0.14 , 0.145, 0.145, 0.145, 0.15 , 0.15 , 0.15 , 0.15 ,
        #                 0.155, 0.155, 0.155, 0.155, 0.155, 0.155, 0.16 , 0.16 , 0.16 ,
        #                 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 ,
        #                 0.155, 0.155, 0.155, 0.155, 0.155, 0.155, 0.15 , 0.15 , 0.15 ,
        #                 0.15 , 0.145, 0.145, 0.145, 0.14 , 0.14 , 0.135, 0.135, 0.135,
        #                 0.13 , 0.125, 0.125, 0.12 , 0.115, 0.115, 0.11 , 0.105, 0.1  ,
        #                 0.09 , 0.085, 0.075, 0.065, 0.05 , 0.03
        # ])
        #
        # W_in = np.array([
        #                 0.0072, 0.0111, 0.0144, 0.0175, 0.0201, 0.0231, 0.0259, 0.0273,
        #                 0.0294, 0.0328, 0.0347, 0.0363, 0.0376, 0.0407, 0.0417, 0.0447,
        #                 0.0453, 0.0481, 0.0483, 0.051 , 0.0508, 0.0532, 0.0556, 0.0549,
        #                 0.057 , 0.0592, 0.0612, 0.0597, 0.0615, 0.0632, 0.0647, 0.0662,
        #                 0.0675, 0.065 , 0.066 , 0.0669, 0.0677, 0.0684, 0.0689, 0.0693,
        #                 0.0696, 0.0698, 0.0698, 0.0697, 0.0694, 0.0729, 0.0724, 0.0718,
        #                 0.071 , 0.0701, 0.0692, 0.0717, 0.0705, 0.0691, 0.0677, 0.0696,
        #                 0.0679, 0.0662, 0.0675, 0.0656, 0.0666, 0.0645, 0.0623, 0.0629,
        #                 0.0632, 0.0608, 0.0608, 0.0606, 0.0582, 0.0577, 0.0571, 0.0564,
        #                 0.057 , 0.0558, 0.0558, 0.0554, 0.0554, 0.0551
        # ])
        #
        # B_in = np.array([
        #                 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.0009, 0.0012, 0.0015,
        #                 0.0019, 0.0023, 0.0027, 0.0031, 0.0035, 0.0041, 0.0046, 0.0052,
        #                 0.0056, 0.0063, 0.0066, 0.0073, 0.0077, 0.0083, 0.009 , 0.0093,
        #                 0.0099, 0.0105, 0.011 , 0.0112, 0.0117, 0.0121, 0.0126, 0.013 ,
        #                 0.0133, 0.0132, 0.0135, 0.0137, 0.0139, 0.014 , 0.0141, 0.0141,
        #                 0.0141, 0.014 , 0.0139, 0.0137, 0.0135, 0.0137, 0.0134, 0.013 ,
        #                 0.0126, 0.0122, 0.0117, 0.0116, 0.0111, 0.0105, 0.0099, 0.0096,
        #                 0.009 , 0.0083, 0.0079, 0.0073, 0.0068, 0.0062, 0.0055, 0.005 ,
        #                 0.0046, 0.004 , 0.0035, 0.0031, 0.0025, 0.0021, 0.0017, 0.0014,
        #                 0.0011, 0.0008, 0.0005, 0.0003, 0.0002, 0.0001
        # ])
        #
        # T_in = np.array([
        #                 297.0254, 296.8377, 296.6496, 296.4618, 296.2756, 296.0892,
        #                 295.9032, 295.7174, 295.5317, 295.3482, 295.1644, 294.979 ,
        #                 294.7937, 294.6125, 294.4267, 294.2445, 294.0592, 293.8765,
        #                 293.6891, 293.5068, 293.3177, 293.1335, 292.9493, 292.7574,
        #                 292.571 , 292.3842, 292.1963, 291.9999, 291.81  , 291.6189,
        #                 291.4266, 291.2333, 291.0384, 290.8335, 290.6359, 290.4366,
        #                 290.2358, 290.0337, 289.8297, 289.6242, 289.4173, 289.2083,
        #                 288.9981, 288.786 , 288.5723, 288.3662, 288.1491, 287.9308,
        #                 287.7108, 287.4894, 287.2668, 287.0506, 286.8253, 286.5983,
        #                 286.3709, 286.1489, 285.9192, 285.6888, 285.4631, 285.2314,
        #                 285.0035, 284.7708, 284.537 , 284.3076, 284.0779, 283.8432,
        #                 283.6134, 283.3813, 283.1483, 282.9169, 282.6858, 282.4555,
        #                 282.2254, 281.9961, 281.7659, 281.5373, 281.309 , 281.0807
        # ])
        #
        # QL_in = np.array([
        #                 0.0008, 0.001 , 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016,
        #                 0.0017, 0.0018, 0.0019, 0.002 , 0.0021, 0.0022, 0.0023, 0.0024,
        #                 0.0025, 0.0026, 0.0027, 0.0028, 0.0029, 0.003 , 0.0031, 0.0032,
        #                 0.0033, 0.0034, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039, 0.004 ,
        #                 0.0041, 0.0042, 0.0043, 0.0045, 0.0046, 0.0047, 0.0048, 0.0049,
        #                 0.005 , 0.0051, 0.0052, 0.0053, 0.0055, 0.0056, 0.0057, 0.0058,
        #                 0.0059, 0.006 , 0.0061, 0.0063, 0.0064, 0.0065, 0.0066, 0.0067,
        #                 0.0068, 0.007 , 0.0071, 0.0072, 0.0073, 0.0074, 0.0076, 0.0077,
        #                 0.0078, 0.0079, 0.008 , 0.0081, 0.0082, 0.0083, 0.0085, 0.0086,
        #                 0.0087, 0.0088, 0.0089, 0.009 , 0.0091, 0.0092
        # ])

        # criterion 3: w>0.01 & b>1e-4
        z_in = np.array([
                         125.,  175.,  225.,  275.,  325.,  375.,  425.,  475.,  525.,
                         575.,  625.,  675.,  725.,  775.,  825.,  875.,  925.,  975.,
                        1025., 1075., 1125., 1175., 1225., 1275., 1325., 1375., 1425.,
                        1475., 1525., 1575., 1625., 1675., 1725., 1775., 1825., 1875.,
                        1925., 1975., 2025., 2075., 2125., 2175., 2225., 2275., 2325.,
                        2375., 2425., 2475., 2525., 2575., 2625., 2675., 2725., 2775.,
                        2825., 2875., 2925., 2975., 3025., 3075., 3125., 3175., 3225.,
                        3275., 3325., 3375., 3425., 3475., 3525., 3575., 3625., 3675.,
                        3725., 3775., 3825., 3875., 3925.
        ])
        thetal_in = np.array([
                        295.6471, 295.6477, 295.6551, 295.6632, 295.6783, 295.7006,
                        295.7226, 295.754 , 295.7804, 295.8182, 295.8498, 295.8935,
                        295.9399, 295.9891, 296.0253, 296.0767, 296.13  , 296.1836,
                        296.2376, 296.2922, 296.3237, 296.3753, 296.4263, 296.4748,
                        296.5214, 296.5663, 296.6074, 296.6454, 296.681 , 296.7118,
                        296.7078, 296.7301, 296.7481, 296.7613, 296.7688, 296.7727,
                        296.7706, 296.7632, 296.7505, 296.7326, 296.7095, 296.6811,
                        296.6478, 296.6094, 296.5662, 296.5185, 296.4663, 296.4101,
                        296.35  , 296.2862, 296.2195, 296.1495, 296.0773, 296.0028,
                        295.9265, 295.8491, 295.7704, 295.6917, 295.6125, 295.534 ,
                        295.456 , 295.3792, 295.3044, 295.2309, 295.1608, 295.0926,
                        295.0281, 294.9671, 294.9159, 294.8673, 294.82  , 294.7803,
                        294.7383, 294.7019, 294.6682, 294.6376, 294.6108
        ])
        Area_in = np.array([
                        0.04 , 0.06 , 0.07 , 0.08 , 0.085, 0.085, 0.09 , 0.09 , 0.095,
                        0.095, 0.1  , 0.1  , 0.1  , 0.1  , 0.105, 0.105, 0.105, 0.105,
                        0.105, 0.105, 0.11 , 0.11 , 0.11 , 0.11 , 0.11 , 0.11 , 0.11 ,
                        0.11 , 0.11 , 0.11 , 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
                        0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
                        0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
                        0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
                        0.115, 0.115, 0.115, 0.115, 0.115, 0.11 , 0.105, 0.1  , 0.09 ,
                        0.085, 0.075, 0.065, 0.05 , 0.03
        ])

        W_in = np.array([
                        0.0116, 0.0148, 0.0181, 0.0209, 0.024 , 0.028 , 0.0308, 0.0347,
                        0.0372, 0.0411, 0.0433, 0.0471, 0.0509, 0.0547, 0.0562, 0.0598,
                        0.0634, 0.067 , 0.0706, 0.074 , 0.0744, 0.0776, 0.0807, 0.0837,
                        0.0867, 0.0895, 0.0922, 0.0948, 0.0972, 0.0994, 0.0975, 0.0994,
                        0.101 , 0.1025, 0.1038, 0.1049, 0.1059, 0.1066, 0.1071, 0.1075,
                        0.1076, 0.1075, 0.1073, 0.1068, 0.1062, 0.1054, 0.1044, 0.1032,
                        0.1019, 0.1004, 0.0987, 0.0969, 0.095 , 0.0929, 0.0908, 0.0885,
                        0.0862, 0.0838, 0.0813, 0.0787, 0.0762, 0.0735, 0.0709, 0.0683,
                        0.0657, 0.0632, 0.0606, 0.0582, 0.0577, 0.0571, 0.0564, 0.057 ,
                        0.0558, 0.0558, 0.0554, 0.0554, 0.0551
        ])

        B_in = np.array([
                        0.0002, 0.0003, 0.0005, 0.0007, 0.001 , 0.0014, 0.0017, 0.0022,
                        0.0026, 0.0032, 0.0036, 0.0043, 0.0049, 0.0056, 0.0061, 0.0069,
                        0.0076, 0.0084, 0.0092, 0.0099, 0.0104, 0.0112, 0.0119, 0.0126,
                        0.0133, 0.014 , 0.0146, 0.0152, 0.0157, 0.0163, 0.0164, 0.0168,
                        0.0172, 0.0175, 0.0177, 0.018 , 0.0181, 0.0182, 0.0183, 0.0182,
                        0.0181, 0.018 , 0.0178, 0.0176, 0.0172, 0.0169, 0.0164, 0.016 ,
                        0.0155, 0.0149, 0.0143, 0.0137, 0.013 , 0.0123, 0.0116, 0.0109,
                        0.0101, 0.0093, 0.0086, 0.0078, 0.0071, 0.0064, 0.0056, 0.005 ,
                        0.0043, 0.0037, 0.0031, 0.0025, 0.0021, 0.0017, 0.0014, 0.0011,
                        0.0008, 0.0005, 0.0003, 0.0002, 0.0001
        ])

        T_in = np.array([
                        296.8382, 296.65  , 296.4624, 296.2763, 296.0902, 295.9057,
                        295.7219, 295.5388, 295.3544, 295.1738, 294.9897, 294.8086,
                        294.6291, 294.4487, 294.2644, 294.0851, 293.9047, 293.724 ,
                        293.5442, 293.3632, 293.1754, 292.9937, 292.8106, 292.6263,
                        292.4422, 292.2566, 292.0696, 291.8824, 291.6934, 291.5028,
                        291.3031, 291.1096, 290.9147, 290.7186, 290.5201, 290.3205,
                        290.1192, 289.9154, 289.7105, 289.5035, 289.2943, 289.0839,
                        288.8709, 288.6567, 288.4407, 288.2223, 288.003 , 287.7812,
                        287.5585, 287.3341, 287.1078, 286.881 , 286.6518, 286.4225,
                        286.1914, 285.9596, 285.7274, 285.4934, 285.2602, 285.025 ,
                        284.7908, 284.5555, 284.3203, 284.0858, 283.8502, 283.6166,
                        283.3813, 283.1483, 282.9169, 282.6858, 282.4555, 282.2254,
                        281.9961, 281.7659, 281.5373, 281.309 , 281.0807
        ])

        QL_in = np.array([
                        0.001 , 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017,
                        0.0018, 0.0019, 0.002 , 0.0021, 0.0022, 0.0023, 0.0024, 0.0025,
                        0.0026, 0.0027, 0.0028, 0.0029, 0.003 , 0.0031, 0.0032, 0.0033,
                        0.0034, 0.0035, 0.0035, 0.0036, 0.0037, 0.0038, 0.004 , 0.0041,
                        0.0042, 0.0043, 0.0044, 0.0045, 0.0046, 0.0047, 0.0048, 0.0049,
                        0.005 , 0.0051, 0.0053, 0.0054, 0.0055, 0.0056, 0.0057, 0.0058,
                        0.006 , 0.0061, 0.0062, 0.0063, 0.0064, 0.0066, 0.0067, 0.0068,
                        0.0069, 0.0071, 0.0072, 0.0073, 0.0074, 0.0075, 0.0077, 0.0078,
                        0.0079, 0.008 , 0.0081, 0.0082, 0.0083, 0.0085, 0.0086, 0.0087,
                        0.0088, 0.0089, 0.009 , 0.0091, 0.0092
        ])

        # with nogil:
        for i in xrange(self.n_updrafts):
            for k in xrange(self.Gr.nzg):
                if self.Gr.z_half[k+self.Gr.gw]<=z_in.max():
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

        self.set_means(GMV)

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
