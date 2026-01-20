import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


## only load the bands that correspond to the users bands
def load_tng_data(mag_file, props_file, bands_to_use=['u', 'b', 'v', 'k', 'g', 'r', 'i', 'z'],properties=['sfr']): 
    """Load TNG data
    
    ##TODO: update to allow loading different properties

    Load the TNG data and return the magnitudes and SFRs.
    
    Args:
        mag_file(string): the file containing the magnitudes
        props_file(string): the file containing the properties
        bands_to_use(list): the bands to use in the calculation, default will use all bands
        
    Returns:
        mags_to_use(pandas.DataFrame): the magnitudes to use
        logProps(pandas.DataFrame): the log(properties) to use, clipped to -5,100
    """
    try:
        mags = pd.DataFrame(np.load(mag_file))
    except:
        print(f'Error when loading {mag_file}.')
    try:
        props = pd.DataFrame(np.load(props_file))
    except:
        print(f'Error when loading {props_file}.')
    mags.columns = ['u', 'b', 'v', 'k', 'g', 'r', 'i', 'z']
    mags_to_use = mags[bands_to_use]
    return mags_to_use, props

def split_dataset(X, Y):
    """Split data set
    
    Split the TNG data and return the magnitudes and SFRs of the tests and training sets.
    
    Args:
        X: the file containing the TNG magnitudes we will use
        Y: the file containing the TNG properties we will use

    Returns:
        X_train:TNG magnitudes training set
        X_test:TNG magnitudes test set
        Y_train: TNG properties training set
        Y_test: TNG properties test set
        
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15,
                                                                random_state=12)
    return X_train, X_test, Y_train, Y_test