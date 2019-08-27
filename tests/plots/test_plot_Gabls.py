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
    setup = cmn.simulation_setup('GABLS')
    # change the defaults
    setup['namelist']['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var'] = True

    # run scampy
    subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data

def test_plot_timeseries_Gabls(sim_data):
    """
    plot Gabls timeseries
    """
    # make directory
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/Gabls/")
    except:
        print('Gabls folder exists')
    les_data = Dataset(localpath +'/les_data/Gabls.nc', 'r')
    data_to_plot = cmn.read_data_srs(sim_data)
    les_data_to_plot = cmn.read_les_data_srs(les_data)

    pls.plot_timeseries(data_to_plot, les_data_to_plot,          folder="plots/output/Gabls/")
    pls.plot_mean(data_to_plot, les_data_to_plot,10,12,             folder="plots/output/Gabls/")
    pls.plot_closures(data_to_plot, les_data_to_plot,10,12,         "Gabls_closures.pdf", folder="plots/output/Gabls/")
    pls.plot_var_covar_mean(data_to_plot, les_data_to_plot, 10,12,  "Gabls_var_covar_mean.pdf", folder="plots/output/Gabls/")
    pls.plot_var_covar_components(data_to_plot,10,12,               "Gabls_var_covar_components.pdf", folder="plots/output/Gabls/")
    pls.plot_tke_components(data_to_plot, les_data_to_plot, 10,12,  "Gabls_tke_components.pdf", folder="plots/output/Gabls/")
    pls.plot_tke_breakdown(data_to_plot, les_data_to_plot, 10,12,   "Gabls_tke_breakdown.pdf", folder="plots/output/Gabls/")

def test_plot_timeseries_1D_Gabls(sim_data):
    """
    plot Gabls 1D timeseries
    """
    localpath = os.getcwd()
    # try:
    #     os.mkdir(localpath + "/plots/output/Gabls/")
    # except:
    #     print('Gabls folder exists')
    les_data = Dataset(localpath + '/les_data/Gabls.nc', 'r')
    data_to_plot = cmn.read_data_timeseries(sim_data)
    les_data_to_plot = cmn.read_les_data_timeseries(les_data)

    pls.plot_timeseries_1D(data_to_plot,  les_data_to_plot, folder="plots/output/Gabls/")
