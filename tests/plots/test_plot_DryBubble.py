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

    # remove netcdf file from previous failed test
    request.addfinalizer(cmn.removing_files)
    # generate namelists and paramlists
    setup = cmn.simulation_setup('DryBubble')
    # change the defaults
    #setup['namelist']['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var'] = True

    # run scampy
    subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data


def test_plot_DryBubble(sim_data):
    """
    plot DryBubble timeseries
    """
    # make directory
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/DryBubble/")
    except:
        print('DryBubble folder exists')

    if (os.path.exists(localpath + "/les_data/DryBubble.nc")):
        les_data = Dataset(localpath + "/les_data/DryBubble.nc", 'r')
    else:
        url_ = "https://www.dropbox.com/s/zrhxou8i80bfdk2/DryBubble.nc?dl=0"
        os.system("wget -O "+localpath+"/les_data/DryBubble.nc "+url_)
        les_data = Dataset(localpath + "/les_data/DryBubble.nc", 'r')

    f1 = "plots/output/DryBubble/"

    scm_dict = cmn.read_scm_bubble(sim_data)
    les_dict = cmn.read_les_bubble(les_data)

    scm_dict["t"] = scm_dict["t"]+0

    pls.plot_bubble(scm_dict, les_dict, folder=f1)
