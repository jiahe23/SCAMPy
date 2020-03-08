import time
import numpy as np
cimport numpy as np
from Variables cimport GridMeanVariables
from EDMF_Updrafts cimport UpdraftVariables
from Turbulence import ParameterizationFactory
from Cases import CasesFactory
cimport Grid
cimport ReferenceState
cimport Cases
from Surface cimport  SurfaceBase
from Cases cimport  CasesBase
from NetCDFIO cimport NetCDFIO_Stats
cimport TimeStepping

import matplotlib.pyplot as plt

class Simulation1d:

    def __init__(self, namelist, paramlist):
        # try:
        #     self.n_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']
        # except:
        #     self.n_updrafts = 1
        #     print('Turbulence--EDMF_PrognosticTKE: defaulting to single updraft')
        self.Gr = Grid.Grid(namelist)
        self.Ref = ReferenceState.ReferenceState(self.Gr)
        self.GMV = GridMeanVariables(namelist, self.Gr, self.Ref)
        # self.UpdVar = UpdraftVariables(self.n_updrafts, namelist, paramlist, self.Gr)
        self.Case = CasesFactory(namelist, paramlist)
        self.Turb = ParameterizationFactory(namelist,paramlist, self.Gr, self.Ref)
        self.TS = TimeStepping.TimeStepping(namelist)
        self.Stats = NetCDFIO_Stats(namelist, paramlist, self.Gr)
        return

    def initialize(self, namelist):
        self.Case.initialize_reference(self.Gr, self.Ref, self.Stats)
        self.Case.initialize_profiles(self.Gr, self.GMV, self.Ref)
        self.Case.initialize_surface(self.Gr, self.Ref )
        self.Case.initialize_forcing(self.Gr, self.Ref, self.GMV)
        self.Turb.initialize(self.Case, self.GMV, self.Ref)
        self.initialize_io()
        self.io()

        return

    def run(self):
        while self.TS.t <= self.TS.t_max:
            print self.TS.t
            print "\n\n================== now updating =================="
            print "sim time: "+str(self.TS.t)
            self.GMV.zero_tendencies()
            self.Case.update_surface(self.GMV, self.TS)
            self.Case.update_forcing(self.GMV, self.TS)
            self.Turb.update(self.GMV, self.Case, self.TS)
            self.TS.update()
            # Apply the tendencies, also update the BCs and diagnostic thermodynamics
            self.GMV.update(self.TS)
            self.Turb.update_GMV_diagnostics(self.GMV)
            if np.mod(self.TS.t, self.Stats.frequency) == 0:
                self.io()
        return

    def initialize_io(self):

        self.GMV.initialize_io(self.Stats)
        self.Case.initialize_io(self.Stats)
        self.Turb.initialize_io(self.Stats)
        return

    def io(self):
        self.Stats.open_files()
        self.Stats.write_simulation_time(self.TS.t)
        self.GMV.io(self.Stats)
        self.Case.io(self.Stats)
        self.Turb.io(self.Stats, self.TS)
        self.Stats.close_files()
        return

    def force_io(self):
        return
