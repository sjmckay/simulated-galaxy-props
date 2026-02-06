from tabnanny import verbose
from xml.parsers.expat import model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .dataset import load_tng_data, split_dataset
from sklearn.ensemble import RandomForestRegressor
import warnings
from importlib.resources import files

class InferenceModel(RandomForestRegressor):
    def __init__(self,random_state=42, n_estimators=200):
        super().__init__(random_state=random_state, n_estimators=n_estimators)
        self.trained = False

    def fit(self, X, Y):
        super().fit(X, Y)
        self.trained = True
    
    def train(self, bands_to_use=['u', 'b', 'v', 'k', 'g', 'r', 'i', 'z'], 
              properties=['sfr'], verbose=False):
        '''model training'''
        mag_path = files('simulated_galaxy_props').joinpath('data/SubhaloMag.npy')
        sfr_path = files('simulated_galaxy_props').joinpath('data/subhaloSFR.npy')
        X, Y = load_tng_data(mag_path, sfr_path, bands_to_use=bands_to_use,
                                   properties=properties) #TODO: update dataset loader to accept properties
        x_train, x_test, y_train, y_test = split_dataset(X, Y)
        self.fit(x_train, np.ravel(y_train))
        train_score = super().score(x_train, y_train)
        test_score = super().score(x_test, y_test)
        if verbose:
            print(train_score)
            print(test_score)
        return train_score, test_score
    
    def predict(self, user_data):
        '''property computation

        Calculate desired properties (currently SFRs) of the user-defined galaxies using the model trained on TNG data.

        Args:
            bands (list): list of strings where each string refers to a different photometric band. These are the 
            photometric bands input by the user. However, at the moment they need to correspond to one of ['u', 'b', 'v', 'k', 'g', 'r', 'i', 'z']
            to match the existing TNG data.
            user_data (array): absolute magnitudes in each band specified in "bands."

        Returns:
            array: SFRs of the galaxies
        '''
        if not self.trained:
            warnings.warn("Model must be trained before prediction.")
        pred = super().predict(user_data)
        return pred