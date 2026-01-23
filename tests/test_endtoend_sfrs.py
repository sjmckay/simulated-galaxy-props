import numpy as np
import pandas as pd
from simulated_galaxy_props import model as model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from importlib.resources import files
data = pd.read_csv(files('simulated_galaxy_props').joinpath('data/mock_observed_galaxy_data.csv'))

#data = pd.read_csv()
# print(data.head(4))
mag_data = data[['mag_u', 'mag_g','mag_r','mag_z']]
mag_data.columns = ['u','g','r','z']
mag_data -= 37.68

model = model.InferenceModel()
model.train(bands_to_use=['u','g','r','z'], properties=['sfr'])

test_sfrs = model.predict(user_data=mag_data)

print(mean_absolute_error(np.log10(test_sfrs), data['log_SFR']))

assert np.all(np.isfinite(test_sfrs))  # makes sure that the sfrs are kinda reasonable