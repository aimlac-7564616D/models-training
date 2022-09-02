import numpy as np
from utils.utils import *
import pandas as pd


def guess_future(X:pd.DataFrame, model_name:str, architecture:str) -> np.array:
    """Function to predict/forcast
    
    Input the driving data () load the model by the model name and return the ouptput of the network
    
    Input:
        X (np.array): The data - format should be the same as one of the outputs output of utils.preprocess_data()
            depending on the data the model was trained on/ is trying to forcast
        model_name (str): The name of the model
        architecture (str): one of the models -> ('gru', 'lstm')
        
    Output: 
        y (np.array): reverse normalized output of the network
    """
    model, input_shape = load_model(model_name, architecture)

    # The data should be the length required for a full 24 hr prediction
    X_trim = snip_timeframe(X, look_back=input_shape[0])
    
    X_norm = model.scaler.transform(X)
    Xs, ys = create_dataset(X_norm, look_back=input_shape[0])
    return model.forcast(Xs)
