from simulated_galaxy_props import model
import numpy as np
import pandas as pd

def test_compute():
    """
    Function to test the compute() method of sfr_calculator.
    """
    test_phot = np.random.rand(10,3) * 5 - 21 # random floats from -21 to -16 (abs mags)
    tphot = pd.DataFrame(test_phot,columns=['u','b','z'])
    model = model.InferenceModel()
    model.train(bands_to_use=['u','b','z'], properties=['sfr'])

    test_sfrs = model.predict(user_data=tphot)
    assert len(test_sfrs) == len(test_phot)
    assert np.all(np.isfinite(test_sfrs))