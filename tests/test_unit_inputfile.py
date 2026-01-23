import sys

import numpy as np
import pandas as pd

from simulated_galaxy_props import model
import matplotlib.pyplot as plt
from importlib.resources import files

data = pd.read_csv(files('simulated_galaxy_props').joinpath('data/mock_observed_galaxy_data.csv'))


def test_inputfile():
    print("Now we are testing the input file")
    # Load the data
    mag_data = data[['mag_u', 'mag_g','mag_r','mag_z']]
    mag_data.columns = ['u','g','r','z']
    
    bands = ['u', 'b', 'v', 'k', 'g', 'r', 'i', 'z']

    assert np.isin(mag_data.columns,bands).all()
    assert np.isnan(mag_data.values).all()==False


    
