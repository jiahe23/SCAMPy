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

        # criterion 1: w>0.01
        z_in = np.array([
                         125.,  175.,  225.,  275.,  325.,  375.,  425.,  475.,  525.,
                         575.,  625.,  675.,  725.,  775.,  825.,  875.,  925.,  975.,
                        1025., 1075., 1125., 1175., 1225., 1275., 1325., 1375., 1425.,
                        1475., 1525., 1575., 1625., 1675., 1725., 1775., 1825., 1875.,
                        1925., 1975., 2025., 2075., 2125., 2175., 2225., 2275., 2325.,
                        2375., 2425., 2475., 2525., 2575., 2625., 2675., 2725., 2775.,
                        2825., 2875., 2925., 2975., 3025., 3075., 3125., 3175., 3225.,
                        3275., 3325., 3375., 3425., 3475., 3525., 3575., 3625., 3675.,
                        3725., 3775., 3825., 3875., 3925., 3975., 4025., 4075., 4125.,
                        4175., 4225., 4275., 4325., 4375., 4425., 4475., 4525., 4575.,
                        4625., 4675., 4725., 4775., 4825., 4875., 4925., 4975., 5025.,
                        5075., 5125., 5175., 5225., 5275., 5325., 5375., 5425., 5475.,
                        5525., 5575., 5625., 5675., 5725., 5775., 5825., 5875., 5925.,
                        5975., 6025., 6075., 6125., 6175.
        ])
        thetal_in = np.array([
                        295.6471, 295.6478, 295.6552, 295.6634, 295.6785, 295.7008,
                        295.7229, 295.7543, 295.7808, 295.8188, 295.8503, 295.8941,
                        295.9406, 295.9899, 296.0261, 296.0777, 296.131 , 296.1847,
                        296.2387, 296.2934, 296.3249, 296.3765, 296.4275, 296.4761,
                        296.5226, 296.5675, 296.6085, 296.6465, 296.682 , 296.7127,
                        296.7086, 296.7308, 296.7487, 296.7619, 296.7693, 296.7731,
                        296.7707, 296.7633, 296.7504, 296.7324, 296.7091, 296.6807,
                        296.6472, 296.6087, 296.5654, 296.5176, 296.4653, 296.409 ,
                        296.3488, 296.2849, 296.2182, 296.1482, 296.076 , 296.0014,
                        295.9252, 295.8478, 295.7691, 295.6904, 295.6112, 295.5328,
                        295.4549, 295.3781, 295.3034, 295.23  , 295.1599, 295.0919,
                        295.0275, 294.9665, 294.9089, 294.8512, 294.803 , 294.7604,
                        294.7199, 294.6861, 294.6564, 294.6293, 294.6066, 294.585 ,
                        294.5658, 294.5461, 294.5257, 294.5059, 294.4849, 294.4647,
                        294.444 , 294.4227, 294.4017, 294.3798, 294.3586, 294.336 ,
                        294.3144, 294.2922, 294.2694, 294.2469, 294.2235, 294.2007,
                        294.1766, 294.1534, 294.1294, 294.1052, 294.0809, 294.056 ,
                        294.0314, 294.0058, 293.9808, 293.9546, 293.9291, 293.9028,
                        293.8765, 293.8497, 293.8227, 293.7955, 293.7678, 293.7402,
                        293.7118, 293.6837, 293.6547, 293.6261, 293.5965, 293.5673,
                        293.5373, 293.5073
        ])
        Area_in = np.array([
                        0.04 , 0.06 , 0.07 , 0.08 , 0.085, 0.085, 0.09 , 0.09 , 0.095,
                        0.095, 0.1  , 0.1  , 0.1  , 0.1  , 0.105, 0.105, 0.105, 0.105,
                        0.105, 0.105, 0.11 , 0.11 , 0.11 , 0.11 , 0.11 , 0.11 , 0.11 ,
                        0.11 , 0.11 , 0.11 , 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
                        0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
                        0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
                        0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
                        0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.12 , 0.12 , 0.12 ,
                        0.125, 0.125, 0.125, 0.13 , 0.13 , 0.13 , 0.13 , 0.135, 0.135,
                        0.135, 0.135, 0.135, 0.14 , 0.14 , 0.14 , 0.14 , 0.14 , 0.14 ,
                        0.14 , 0.14 , 0.14 , 0.14 , 0.14 , 0.135, 0.135, 0.135, 0.135,
                        0.135, 0.13 , 0.13 , 0.13 , 0.125, 0.125, 0.125, 0.12 , 0.12 ,
                        0.115, 0.11 , 0.11 , 0.105, 0.1  , 0.095, 0.09 , 0.085, 0.08 ,
                        0.075, 0.065, 0.06 , 0.045, 0.035
        ])

        W_in = np.array([
                        0.0012, 0.0015, 0.0018, 0.0021, 0.0024, 0.0028, 0.0031, 0.0035,
                        0.0037, 0.0041, 0.0043, 0.0047, 0.0051, 0.0055, 0.0056, 0.006 ,
                        0.0063, 0.0067, 0.0071, 0.0074, 0.0074, 0.0078, 0.0081, 0.0084,
                        0.0087, 0.009 , 0.0092, 0.0095, 0.0097, 0.0099, 0.0098, 0.0099,
                        0.0101, 0.0103, 0.0104, 0.0105, 0.0106, 0.0107, 0.0107, 0.0107,
                        0.0108, 0.0108, 0.0107, 0.0107, 0.0106, 0.0105, 0.0104, 0.0103,
                        0.0102, 0.01  , 0.0099, 0.0097, 0.0095, 0.0093, 0.0091, 0.0088,
                        0.0086, 0.0084, 0.0081, 0.0079, 0.0076, 0.0073, 0.0071, 0.0068,
                        0.0066, 0.0063, 0.0061, 0.0058, 0.0056, 0.0052, 0.005 , 0.0048,
                        0.0044, 0.0043, 0.0041, 0.0038, 0.0037, 0.0035, 0.0034, 0.0032,
                        0.0031, 0.003 , 0.0029, 0.0028, 0.0026, 0.0025, 0.0025, 0.0024,
                        0.0023, 0.0022, 0.0022, 0.0021, 0.002 , 0.002 , 0.0019, 0.0019,
                        0.0018, 0.0018, 0.0017, 0.0017, 0.0016, 0.0016, 0.0015, 0.0015,
                        0.0015, 0.0014, 0.0014, 0.0014, 0.0013, 0.0013, 0.0013, 0.0013,
                        0.0012, 0.0012, 0.0012, 0.0012, 0.0011, 0.0011, 0.0011, 0.0011,
                        0.001 , 0.001
        ])

        B_in = np.array([
                        0.0002, 0.0003, 0.0005, 0.0007, 0.001 , 0.0014, 0.0017, 0.0022,
                        0.0026, 0.0032, 0.0036, 0.0043, 0.0049, 0.0056, 0.0061, 0.0069,
                        0.0076, 0.0084, 0.0092, 0.0099, 0.0104, 0.0112, 0.0119, 0.0126,
                        0.0133, 0.014 , 0.0146, 0.0152, 0.0158, 0.0163, 0.0164, 0.0168,
                        0.0172, 0.0175, 0.0177, 0.018 , 0.0181, 0.0182, 0.0183, 0.0182,
                        0.0181, 0.018 , 0.0178, 0.0175, 0.0172, 0.0169, 0.0164, 0.0159,
                        0.0155, 0.0149, 0.0143, 0.0137, 0.013 , 0.0123, 0.0116, 0.0109,
                        0.0101, 0.0093, 0.0086, 0.0078, 0.0071, 0.0064, 0.0056, 0.005 ,
                        0.0043, 0.0037, 0.0031, 0.0025, 0.002 , 0.0015, 0.0011, 0.0008,
                        0.0005, 0.0003, 0.0002, 0.0001, 0.    , 0.    , 0.    , 0.    ,
                        0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                        0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                        0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                        0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                        0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
                        0.    , 0.
        ])

        T_in = np.array([
                        296.8382, 296.6501, 296.4624, 296.2764, 296.0903, 295.9058,
                        295.7219, 295.5389, 295.3545, 295.1739, 294.9899, 294.8087,
                        294.6293, 294.4489, 294.2646, 294.0853, 293.905 , 293.7243,
                        293.5445, 293.3635, 293.1757, 292.994 , 292.8109, 292.6266,
                        292.4426, 292.2569, 292.0699, 291.8827, 291.6936, 291.5031,
                        291.3034, 291.1098, 290.9149, 290.7188, 290.5203, 290.3206,
                        290.1193, 289.9154, 289.7105, 289.5034, 289.2942, 289.0838,
                        288.8708, 288.6565, 288.4405, 288.222 , 288.0027, 287.7809,
                        287.5581, 287.3338, 287.1074, 286.8806, 286.6514, 286.4221,
                        286.191 , 285.9592, 285.727 , 285.493 , 285.2598, 285.0246,
                        284.7905, 284.5551, 284.32  , 284.0855, 283.8499, 283.6163,
                        283.3811, 283.1481, 282.9146, 282.6807, 282.4501, 282.219 ,
                        281.9903, 281.7608, 281.5335, 281.3063, 281.0793, 280.854 ,
                        280.6264, 280.4   , 280.1718, 279.9435, 279.7151, 279.4848,
                        279.2557, 279.0239, 278.7928, 278.5606, 278.3275, 278.0949,
                        277.8598, 277.626 , 277.3896, 277.1538, 276.9169, 276.679 ,
                        276.4414, 276.2016, 275.9629, 275.7214, 275.4807, 275.2383,
                        274.9957, 274.7524, 274.5077, 274.2635, 274.0167, 273.7712,
                        273.5227, 273.2751, 273.0255, 272.7759, 272.525 , 272.2734,
                        272.0213, 271.7675, 271.5141, 271.2583, 271.0034, 270.7455,
                        270.4888, 270.2292
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
                        0.0088, 0.0089, 0.009 , 0.0091, 0.0092, 0.0093, 0.0094, 0.0095,
                        0.0096, 0.0097, 0.0098, 0.0099, 0.01  , 0.0101, 0.0102, 0.0103,
                        0.0104, 0.0105, 0.0105, 0.0106, 0.0107, 0.0108, 0.0109, 0.011 ,
                        0.0111, 0.0112, 0.0113, 0.0114, 0.0115, 0.0116, 0.0117, 0.0117,
                        0.0118, 0.0119, 0.012 , 0.0121, 0.0122, 0.0123, 0.0124, 0.0125,
                        0.0125, 0.0126, 0.0127, 0.0128, 0.0129, 0.013 , 0.0131, 0.0131,
                        0.0132, 0.0133
        ])

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
        #                 295.6455, 295.6454, 295.6463, 295.653 , 295.6604, 295.6747,
        #                 295.6913, 295.7062, 295.7273, 295.7573, 295.7829, 295.81  ,
        #                 295.8379, 295.8781, 295.9077, 295.9368, 295.9807, 296.0263,
        #                 296.0543, 296.0998, 296.1254, 296.1693, 296.213 , 296.2323,
        #                 296.272 , 296.3106, 296.3473, 296.3534, 296.3839, 296.4121,
        #                 296.4364, 296.4574, 296.4751, 296.4573, 296.4666, 296.471 ,
        #                 296.4721, 296.4682, 296.4598, 296.447 , 296.4296, 296.4078,
        #                 296.3816, 296.3511, 296.3164, 296.3085, 296.2653, 296.2183,
        #                 296.1677, 296.1139, 296.057 , 296.0242, 295.9613, 295.8963,
        #                 295.8294, 295.7839, 295.7134, 295.6422, 295.5903, 295.5174,
        #                 295.4625, 295.3898, 295.3184, 295.2622, 295.2049, 295.1375,
        #                 295.0818, 295.0275, 294.9742, 294.9154, 294.8669, 294.8243,
        #                 294.7799, 294.7381, 294.7017, 294.668 , 294.6375, 294.6107
        # ])
        # Area_in = np.array([
        #                 0.025, 0.05 , 0.065, 0.075, 0.085, 0.09 , 0.095, 0.105, 0.11 ,
        #                 0.11 , 0.115, 0.12 , 0.125, 0.125, 0.13 , 0.135, 0.135, 0.135,
        #                 0.14 , 0.14 , 0.145, 0.145, 0.145, 0.15 , 0.15 , 0.15 , 0.15 ,
        #                 0.155, 0.155, 0.155, 0.155, 0.155, 0.155, 0.16 , 0.16 , 0.16 ,
        #                 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 , 0.16 ,
        #                 0.155, 0.155, 0.155, 0.155, 0.155, 0.155, 0.15 , 0.15 , 0.15 ,
        #                 0.15 , 0.145, 0.145, 0.145, 0.14 , 0.14 , 0.135, 0.135, 0.135,
        #                 0.13 , 0.125, 0.125, 0.12 , 0.115, 0.11 , 0.11 , 0.105, 0.095,
        #                 0.09 , 0.085, 0.075, 0.065, 0.05 , 0.03
        # ])
        #
        # W_in = np.array([
        #                 0.0007, 0.0011, 0.0014, 0.0018, 0.002 , 0.0023, 0.0026, 0.0027,
        #                 0.0029, 0.0033, 0.0035, 0.0036, 0.0038, 0.0041, 0.0042, 0.0042,
        #                 0.0045, 0.0048, 0.0048, 0.0051, 0.0051, 0.0053, 0.0056, 0.0055,
        #                 0.0057, 0.0059, 0.0061, 0.006 , 0.0062, 0.0063, 0.0065, 0.0066,
        #                 0.0068, 0.0065, 0.0066, 0.0067, 0.0068, 0.0068, 0.0069, 0.0069,
        #                 0.007 , 0.007 , 0.007 , 0.007 , 0.0069, 0.0073, 0.0072, 0.0072,
        #                 0.0071, 0.007 , 0.0069, 0.0072, 0.007 , 0.0069, 0.0068, 0.007 ,
        #                 0.0068, 0.0066, 0.0067, 0.0066, 0.0067, 0.0064, 0.0062, 0.0063,
        #                 0.0063, 0.0061, 0.0061, 0.0061, 0.006 , 0.0058, 0.0057, 0.0058,
        #                 0.0057, 0.0056, 0.0056, 0.0055, 0.0055, 0.0055
        # ])
        #
        # B_in = np.array([
        #                 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.0009, 0.0012, 0.0015,
        #                 0.0019, 0.0023, 0.0027, 0.0031, 0.0035, 0.0041, 0.0046, 0.005 ,
        #                 0.0056, 0.0063, 0.0067, 0.0073, 0.0077, 0.0083, 0.009 , 0.0093,
        #                 0.0099, 0.0105, 0.011 , 0.0112, 0.0117, 0.0121, 0.0126, 0.013 ,
        #                 0.0133, 0.0132, 0.0135, 0.0137, 0.0139, 0.014 , 0.0141, 0.0141,
        #                 0.0141, 0.014 , 0.0139, 0.0137, 0.0135, 0.0137, 0.0134, 0.013 ,
        #                 0.0126, 0.0122, 0.0117, 0.0116, 0.011 , 0.0105, 0.0099, 0.0096,
        #                 0.009 , 0.0083, 0.0079, 0.0073, 0.0068, 0.0062, 0.0055, 0.005 ,
        #                 0.0046, 0.0039, 0.0035, 0.0031, 0.0026, 0.0021, 0.0017, 0.0014,
        #                 0.0011, 0.0008, 0.0005, 0.0003, 0.0002, 0.0001
        # ])
        #
        # T_in = np.array([
        #                 297.0254, 296.8377, 296.6497, 296.4618, 296.2756, 296.0893,
        #                 295.9033, 295.7175, 295.5318, 295.3483, 295.1645, 294.9791,
        #                 294.7938, 294.6127, 294.4269, 294.2408, 294.0594, 293.8768,
        #                 293.6893, 293.507 , 293.318 , 293.1338, 292.9496, 292.7577,
        #                 292.5713, 292.3845, 292.1965, 292.0002, 291.8103, 291.6191,
        #                 291.4269, 291.2335, 291.0386, 290.8336, 290.636 , 290.4367,
        #                 290.2359, 290.0338, 289.8297, 289.6242, 289.4172, 289.2083,
        #                 288.9979, 288.7859, 288.5721, 288.366 , 288.1489, 287.9305,
        #                 287.7105, 287.4892, 287.2665, 287.0503, 286.8249, 286.598 ,
        #                 286.3706, 286.1485, 285.9189, 285.6884, 285.4627, 285.2311,
        #                 285.0031, 284.7705, 284.5367, 284.3074, 284.0777, 283.8429,
        #                 283.6132, 283.3811, 283.1505, 282.9167, 282.6856, 282.4569,
        #                 282.2253, 281.996 , 281.7658, 281.5372, 281.309 , 281.0807
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
        #                 0.0069, 0.007 , 0.0071, 0.0072, 0.0073, 0.0074, 0.0076, 0.0077,
        #                 0.0078, 0.0079, 0.008 , 0.0081, 0.0082, 0.0083, 0.0085, 0.0086,
        #                 0.0087, 0.0088, 0.0089, 0.009 , 0.0091, 0.0092
        # ])
        #
        # # criterion 3: w>0.01 & b>1e-4
        # z_in = np.array([
        #                  125.,  175.,  225.,  275.,  325.,  375.,  425.,  475.,  525.,
        #                  575.,  625.,  675.,  725.,  775.,  825.,  875.,  925.,  975.,
        #                 1025., 1075., 1125., 1175., 1225., 1275., 1325., 1375., 1425.,
        #                 1475., 1525., 1575., 1625., 1675., 1725., 1775., 1825., 1875.,
        #                 1925., 1975., 2025., 2075., 2125., 2175., 2225., 2275., 2325.,
        #                 2375., 2425., 2475., 2525., 2575., 2625., 2675., 2725., 2775.,
        #                 2825., 2875., 2925., 2975., 3025., 3075., 3125., 3175., 3225.,
        #                 3275., 3325., 3375., 3425., 3475., 3525., 3575., 3625., 3675.,
        #                 3725., 3775., 3825., 3875., 3925.
        # ])
        # thetal_in = np.array([
        #                 295.6471, 295.6478, 295.6552, 295.6634, 295.6785, 295.7008,
        #                 295.7229, 295.7543, 295.7808, 295.8188, 295.8503, 295.8941,
        #                 295.9406, 295.9899, 296.0261, 296.0777, 296.131 , 296.1847,
        #                 296.2387, 296.2934, 296.3249, 296.3765, 296.4275, 296.4761,
        #                 296.5226, 296.5675, 296.6085, 296.6465, 296.682 , 296.7127,
        #                 296.7086, 296.7308, 296.7487, 296.7619, 296.7693, 296.7731,
        #                 296.7707, 296.7633, 296.7504, 296.7324, 296.7091, 296.6807,
        #                 296.6472, 296.6087, 296.5654, 296.5176, 296.4653, 296.409 ,
        #                 296.3488, 296.2849, 296.2182, 296.1482, 296.076 , 296.0014,
        #                 295.9252, 295.8478, 295.7691, 295.6904, 295.6112, 295.5328,
        #                 295.4549, 295.3781, 295.3034, 295.23  , 295.1599, 295.0919,
        #                 295.0275, 294.9742, 294.9154, 294.8669, 294.8243, 294.7799,
        #                 294.7381, 294.7017, 294.668 , 294.6375, 294.6107
        # ])
        # Area_in = np.array([
        #                 0.04 , 0.06 , 0.07 , 0.08 , 0.085, 0.085, 0.09 , 0.09 , 0.095,
        #                 0.095, 0.1  , 0.1  , 0.1  , 0.1  , 0.105, 0.105, 0.105, 0.105,
        #                 0.105, 0.105, 0.11 , 0.11 , 0.11 , 0.11 , 0.11 , 0.11 , 0.11 ,
        #                 0.11 , 0.11 , 0.11 , 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
        #                 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
        #                 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
        #                 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115,
        #                 0.115, 0.115, 0.115, 0.115, 0.11 , 0.11 , 0.105, 0.095, 0.09 ,
        #                 0.085, 0.075, 0.065, 0.05 , 0.03
        # ])
        #
        # W_in = np.array([
        #                 0.0012, 0.0015, 0.0018, 0.0021, 0.0024, 0.0028, 0.0031, 0.0035,
        #                 0.0037, 0.0041, 0.0043, 0.0047, 0.0051, 0.0055, 0.0056, 0.006 ,
        #                 0.0063, 0.0067, 0.0071, 0.0074, 0.0074, 0.0078, 0.0081, 0.0084,
        #                 0.0087, 0.009 , 0.0092, 0.0095, 0.0097, 0.0099, 0.0098, 0.0099,
        #                 0.0101, 0.0103, 0.0104, 0.0105, 0.0106, 0.0107, 0.0107, 0.0107,
        #                 0.0108, 0.0108, 0.0107, 0.0107, 0.0106, 0.0105, 0.0104, 0.0103,
        #                 0.0102, 0.01  , 0.0099, 0.0097, 0.0095, 0.0093, 0.0091, 0.0088,
        #                 0.0086, 0.0084, 0.0081, 0.0079, 0.0076, 0.0073, 0.0071, 0.0068,
        #                 0.0066, 0.0063, 0.0061, 0.006 , 0.0058, 0.0057, 0.0058, 0.0057,
        #                 0.0056, 0.0056, 0.0055, 0.0055, 0.0055
        # ])
        #
        # B_in = np.array([
        #                 0.0002, 0.0003, 0.0005, 0.0007, 0.001 , 0.0014, 0.0017, 0.0022,
        #                 0.0026, 0.0032, 0.0036, 0.0043, 0.0049, 0.0056, 0.0061, 0.0069,
        #                 0.0076, 0.0084, 0.0092, 0.0099, 0.0104, 0.0112, 0.0119, 0.0126,
        #                 0.0133, 0.014 , 0.0146, 0.0152, 0.0158, 0.0163, 0.0164, 0.0168,
        #                 0.0172, 0.0175, 0.0177, 0.018 , 0.0181, 0.0182, 0.0183, 0.0182,
        #                 0.0181, 0.018 , 0.0178, 0.0175, 0.0172, 0.0169, 0.0164, 0.0159,
        #                 0.0155, 0.0149, 0.0143, 0.0137, 0.013 , 0.0123, 0.0116, 0.0109,
        #                 0.0101, 0.0093, 0.0086, 0.0078, 0.0071, 0.0064, 0.0056, 0.005 ,
        #                 0.0043, 0.0037, 0.0031, 0.0026, 0.0021, 0.0017, 0.0014, 0.0011,
        #                 0.0008, 0.0005, 0.0003, 0.0002, 0.0001
        # ])
        #
        # T_in = np.array([
        #                 296.8382, 296.6501, 296.4624, 296.2764, 296.0903, 295.9058,
        #                 295.7219, 295.5389, 295.3545, 295.1739, 294.9899, 294.8087,
        #                 294.6293, 294.4489, 294.2646, 294.0853, 293.905 , 293.7243,
        #                 293.5445, 293.3635, 293.1757, 292.994 , 292.8109, 292.6266,
        #                 292.4426, 292.2569, 292.0699, 291.8827, 291.6936, 291.5031,
        #                 291.3034, 291.1098, 290.9149, 290.7188, 290.5203, 290.3206,
        #                 290.1193, 289.9154, 289.7105, 289.5034, 289.2942, 289.0838,
        #                 288.8708, 288.6565, 288.4405, 288.222 , 288.0027, 287.7809,
        #                 287.5581, 287.3338, 287.1074, 286.8806, 286.6514, 286.4221,
        #                 286.191 , 285.9592, 285.727 , 285.493 , 285.2598, 285.0246,
        #                 284.7905, 284.5551, 284.32  , 284.0855, 283.8499, 283.6163,
        #                 283.3811, 283.1505, 282.9167, 282.6856, 282.4569, 282.2253,
        #                 281.996 , 281.7658, 281.5372, 281.309 , 281.0807
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
        #                 0.0088, 0.0089, 0.009 , 0.0091, 0.0092
        # ])

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
