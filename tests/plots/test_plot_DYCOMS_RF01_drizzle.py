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

@pytest.fixture(scope="module")
def sim_data(request):

    # generate namelists and paramlists
    cmn.removing_files
    setup = cmn.simulation_setup('DYCOMS_RF01')

    #setup['namelist']['thermodynamics']['sgs'] = 'mean'
    setup['namelist']['thermodynamics']['sgs'] = 'quadrature'
    setup['namelist']['microphysics']['rain_model']          = True
    setup['namelist']['microphysics']['max_supersaturation'] = 0.05

    # run scampy
    subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf files after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data

def test_plot_timeseries_DYCOMS_RF01(sim_data):
    """
    plot DYCOMS_RF01 timeseries
    """
    # make directory
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/DYCOMS_RF01_drizzle/")
    except:
        print('DYCOMS_RF01_drizzle folder exists')
    try:
        os.mkdir(localpath + "/plots/output/DYCOMS_RF01_drizzle/all_variables/")
    except:
        print('DYCOMS_RF01_drizzle/all_variables folder exists')

    if (os.path.exists(localpath + "/les_data/DYCOMS_RF01.nc")):
        les_data = Dataset(localpath + "/les_data/DYCOMS_RF01.nc", 'r')
    else:
        url_ = "https://www.dropbox.com/s/dh636h4owlt6a79/DYCOMS_RF01.nc?dl=0"
        os.system("wget -O "+localpath+"/les_data/DYCOMS_RF01.nc "+url_)
        les_data = Dataset(localpath + "/les_data/DYCOMS_RF01.nc", 'r')

    data_to_plot = cmn.read_data_srs(sim_data)
    les_data_to_plot = cmn.read_les_data_srs(les_data)

    pls.plot_closures(data_to_plot, les_data_to_plot,3,4,           "drizzle_DYCOMS_RF01_closures.pdf",           folder="plots/output/DYCOMS_RF01_drizzle/")
    pls.plot_humidities(data_to_plot, les_data_to_plot,3,4,         "drizzle_DYCOMS_RF01_humidities.pdf",         folder="plots/output/DYCOMS_RF01_drizzle/")
    pls.plot_updraft_properties(data_to_plot, les_data_to_plot,3,4, "drizzle_DYCOMS_RF01_updraft_properties.pdf", folder="plots/output/DYCOMS_RF01_drizzle/")
    pls.plot_tke_components(data_to_plot, les_data_to_plot, 3,4,    "drizzle_DYCOMS_RF01_tke_components.pdf",     folder="plots/output/DYCOMS_RF01_drizzle/")

    pls.plot_timeseries(data_to_plot, les_data_to_plot,          folder="plots/output/DYCOMS_RF01_drizzle/all_variables/")
    pls.plot_mean(data_to_plot, les_data_to_plot,3,4,            folder="plots/output/DYCOMS_RF01_drizzle/all_variables/")
    pls.plot_var_covar_mean(data_to_plot, les_data_to_plot, 3,4, "drizzle_DYCOMS_RF01_var_covar_mean.pdf", folder="plots/output/DYCOMS_RF01_drizzle/all_variables/")
    pls.plot_var_covar_components(data_to_plot,3,4,              "drizzle_DYCOMS_RF01_var_covar_components.pdf", folder="plots/output/DYCOMS_RF01_drizzle/all_variables/")
    pls.plot_tke_breakdown(data_to_plot, les_data_to_plot, 3,4,  "drizzle_DYCOMS_RF01_tke_breakdown.pdf", folder="plots/output/DYCOMS_RF01_drizzle/all_variables/")

def test_plot_timeseries_1D_DYCOMS_RF01(sim_data):
    """
    plot DYCOMS_RF01 1D timeseries
    """
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/DYCOMS_RF01_drizzle/")
        print()
    except:
        print('DYCOMS_RF01_drizzle folder exists')
    try:
        os.mkdir(localpath + "/plots/output/DYCOMS_RF01_drizzle/all_variables/")
    except:
        print('DYCOMS_RF01_drizzle/all_variables folder exists')

    if (os.path.exists(localpath + "/les_data/DYCOMS_RF01.nc")):
        les_data = Dataset(localpath + "/les_data/DYCOMS_RF01.nc", 'r')
    else:
        url_ = "https://www.dropbox.com/s/dh636h4owlt6a79/DYCOMS_RF01.nc?dl=0"
        os.system("wget -O "+localpath+"/les_data/DYCOMS_RF01.nc "+url_)
        les_data = Dataset(localpath + "/les_data/DYCOMS_RF01.nc", 'r')

    data_to_plot = cmn.read_data_timeseries(sim_data)
    les_data_to_plot = cmn.read_les_data_timeseries(les_data)
    data_to_plot_ = cmn.read_data_srs(sim_data)
    les_data_to_plot_ = cmn.read_les_data_srs(les_data)

    pls.plot_main_timeseries(data_to_plot, les_data_to_plot, data_to_plot_, les_data_to_plot_, "DYCOMS_RF01_drizzle_main_timeseries.pdf",folder="plots/output/DYCOMS_RF01_drizzle/")
    pls.plot_timeseries_1D(data_to_plot,  les_data_to_plot,  folder="plots/output/DYCOMS_RF01_drizzle/all_variables/")
