import sys
sys.path.insert(0, "./")
sys.path.insert(0, "../")

import os
import subprocess
import json
import warnings

from netCDF4 import Dataset

import pytest
import numpy as np

import main as scampy
import common as cmn
import plot_scripts as pls
import pprint as pp

@pytest.fixture(scope="module")
def sim_data(request):

    # remove netcdf file from previous failed test
    request.addfinalizer(cmn.removing_files)
    # generate namelists and paramlists
    setup = cmn.simulation_setup('Rico')

    # run scampy
    subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data

def test_plot_Rico(sim_data):
    """
    plot Rico timeseries
    """
    # make directory
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/Rico/")
    except:
        print('Rico folder exists')
    try:
        os.mkdir(localpath + "/plots/output/Rico/all_variables/")
    except:
        print('Rico/all_variables folder exists')

    if (os.path.exists(localpath + "/les_data/Rico.nc")):
        les_data = Dataset(localpath + "/les_data/Rico.nc", 'r')
    else:
        url_ = "https://www.dropbox.com/s/c2bvey47y8xryuc/Rico.nc?dl=0"
        os.system("wget -O "+localpath+"/les_data/Rico.nc "+url_)
        les_data = Dataset(localpath + "/les_data/Rico.nc", 'r')

    f1 = "plots/output/Rico/"
    f2 = f1 + "all_variables/"
    cn = "Rico_"
    t0 = 22
    t1 = 24
    zmin = 0.0
    zmax = 4.0
    cb_min = [0., 0.]
    cb_max = [0.05, 5]
    fixed_cbar = True
    cb_min_t = [296, 296, 297, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 9,\
                -0.2, 0, -10, -7,\
                 0, -0.05, 0,\
                -0.2, -0.05, -0.2,\
                 0., -0.3, -1e-4]
    cb_max_t = [330, 330, 308, 0.04, 0.02, 1.5, 0.05, 0.03, 100, 3, 100, 18, 18, 18,\
                0, 4, 3, -3,\
                0.24, 0.03, 1.5,\
                0.04, 0.05, 0.04,\
                0.3, 0.05, 4e-4]

    scm_dict = cmn.read_scm_data(sim_data)
    les_dict = cmn.read_les_data(les_data)

    scm_dict_t = cmn.read_scm_data_timeseries(sim_data)
    les_dict_t = cmn.read_les_data_timeseries(les_data)

    pls.plot_closures(scm_dict, les_dict, t0, t1, zmin, zmax, cn+"closures.pdf", folder=f1)
    pls.plot_spec_hum(scm_dict, les_dict, t0, t1, zmin, zmax,  cn+"humidities.pdf", folder=f1)
    pls.plot_upd_prop(scm_dict, les_dict, t0, t1, zmin, zmax,  cn+"updraft_properties.pdf", folder=f1)
    pls.plot_fluxes(scm_dict, les_dict, t0, t1, zmin, zmax,  cn+"mean_fluxes.pdf", folder=f1)
    pls.plot_tke_comp(scm_dict, les_dict, t0, t1, zmin, zmax,  cn+"tke_components.pdf", folder=f1)

    pls.plot_cvar_mean(scm_dict, les_dict, t0, t1, zmin, zmax,  cn+"var_covar_mean.pdf", folder=f2)
    pls.plot_cvar_comp(scm_dict, t0, t1, zmin, zmax,  cn+"var_covar_components.pdf", folder=f2)
    pls.plot_tke_break(scm_dict, les_dict, t0, t1, zmin, zmax, cn+"tke_breakdown.pdf",folder=f2)

    pls.plot_contour_t(scm_dict, les_dict, fixed_cbar, cb_min_t, cb_max_t, zmin, zmax, folder=f2)
    pls.plot_mean_prof(scm_dict, les_dict, t0, t1,  zmin, zmax, folder=f2)

    pls.plot_main(scm_dict_t, les_dict_t, scm_dict, les_dict,
                  cn+"main_timeseries.pdf", cb_min, cb_max, zmin, zmax, folder=f1)

    pls.plot_1D(scm_dict_t, les_dict_t, cn, folder=f2)

