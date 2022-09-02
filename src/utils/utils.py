import sys
sys.path.append('../')

from architectures.models import my_GRU, my_LSTM
from data_processing.data_processing import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Union
import pickle
import numpy as np
import os

save_folder = '../trained_models/'

def read_data(filename: str) -> pd.DataFrame:
    """Reads data from a csv file
    
    Using a file path this function will read the data and produce a pandas dataframe
    
    Input: 
        filename (str): The file path of the csv file you want to import
        
    Ouput:
        pd.DataFrame : returns the data in a pandas dataframe 
    """
    return pd.read_csv(filename)


def preprocess_data(met: pd.DataFrame, open:pd.DataFrame, energy:pd.DataFrame) -> pd.DataFrame:
    """Creates the data for the wind power and solar power generation. 

    Using the met office data, open weather data, and energy onsite data the data is processed. Inserting synthetic data where required, synchronising the dataframes
    and then seperating the driving series and target series for each of the respective targets (wind, solar)
    
    Input:
        met (pd.DataFrame): The met office data in a pandas dataframe
        open (pd.DataFrame): The open weather data in a pandas dataframe
        energy (pd.DataFrame): The energy onsite data in a pandas dataframe
        
    Ouput:
        wind_data (pd.DataFrame): The collected, processed wind data
        solar_data (pd.DataFrame): The collected, processed solar data
    """
    met_prep = prepare_data(met)
    open_prep = prepare_data(open)
    energy_new = prepare_data(energy)

    met_new = insert_synthetic_data(met_prep)
    open_new = insert_synthetic_data(open_prep)
    #energy_new = insert_synthetic_data(energy_prep)

    met_new, open_new, energy_new = synchronise_data(met_new, open_new, energy_new)

    wind_data = generate_wind_data(met_new, open_new, energy_new)
    solar_data = generate_solar_data(met_new, open_new, energy_new)

    return wind_data, solar_data

def shift_target(frame: pd.DataFrame, days:int = 1) -> np.array:
    """Shift the target variable by a number of days

    Takes the target variable and shifts it by a number of days (48 datapoints - data must be in 30 min increments)

    Input:
        frame (pd.DataFrame): The sorted data you want to shift
        days (int): The number of days you want to shift your target variable away from the input time
    Ouptut:
        np.array : The shifted data in a numpy array
    """
    target = frame.iloc[:, -1].to_numpy()
    data = frame.iloc[:, :-1].to_numpy()

    data = data[:-48*days, :]
    target = target[48*days:].reshape(-1, 1)

    return np.append(data, target, axis=1)


def test_train_split(arr:np.array, split:float) -> np.array:
    """Splits the data
    
    splits the data by percentag, where you specify the percent of data you want for training
    
    Input:
        arr (np.array): Input data to be split
        split (float): The percent of the data you want for training

    Output: 
        tr (np.array): The training data
        te (np.array): The testing data    
    """
    [N, _] = arr.shape
    tr_size = int(N * split)
    tr = arr[:tr_size, :]
    te = arr[tr_size:, :]
    return tr, te

def reverse_normlisation(y:np.array, scaler:MinMaxScaler) -> np.array:
    """Reverses the normilization done to the data

    Provided with normalized data, and the scalar used to normalize the data. This function will return the data in their true values.
    Used when the ground truth values are normalized and required for comparison to the output of the network

    Input:
        y (np.array) : normalized target variable 
        scalar (MinMaxScalar) : The scalar used to initially normalize the data

    Ouptut:
        np.array : Reverse normalized output of the network
    """
    n_features = scaler.n_features_in_
    toReturn = np.zeros(shape=(len(y), n_features))
    toReturn[:,-1] = y.reshape(1,-1)
    toReturn = scaler.inverse_transform(toReturn)
    return toReturn[:,-1]

def create_dataset(X: np.array, look_back: int = 1) -> np.array:
    """Transform 2D data into 3D Tensor

    Transforms the data from  2D into a 3D tensor which will be the shape = (N-look_back, look_back, features) where look_back = number of timesteps
    used for each training example

    Input:
        X (np.array): The training data
        look_back (int): How far to look back in time (1 = 30mins & 48 = 24hrs)

    Output:
        Xs (np.array): input data for the network
        ys (np.array): target variable of the network
    """
    Xs, ys = [], []
    for i in range(X.shape[0] - look_back):
        Xs.append(X[i:i+look_back, :-1])
        ys.append(X[i+look_back, -1])
    return np.array(Xs), np.array(ys)

def plot_loss(history, model_name:str) -> None:
    """plot the loss 
    
    Plot the loss curves from training, shows the loss and validation loss curves and saves them in their respective model folder
    in the output folder
    
    Input:
        history (keras.model.fit().history?) : The training history of the network
        model_name (str) : The name of the model
        
    Output:
        None
    """
    s_path = save_folder + model_name + '/output'
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss for ' + model_name)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    if os.path.exists(s_path):
        print('output file there already')
        plt.savefig(s_path + '/loss.png', bbox_inches = 'tight')
    else: 
        os.mkdir(s_path)
        plt.savefig(s_path + '/loss.png', bbox_inches = 'tight')

# Plot test data vs prediction
"""Plot the predicted values against the ground truth

Take the output of the network and the ground truth and plot them against one another to see the performance of the network
Saves the plot to the output folder within the ..trained_models/{model_name}/ directory 

Input:
    prediction (np.array): The output of the network, y_hat
    model_name (str): The name of the model (should be the same as the folder name)
    y_test (np.array)

Ouput: 
    None
"""
def plot_future(prediction:np.array, model_name:str, y_test:np.array) -> None:
    plt.figure(figsize=(10, 6))
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y_test), 
             label='Test data')
    plt.plot(np.arange(range_future), 
             np.array(prediction),label='Prediction')
    plt.title('Test data vs prediction for - Model:' + model_name)
    plt.legend(loc='upper left')
    plt.xlabel('Time (30mins)')
    plt.ylabel('Energy Generation')
    plt.savefig(save_folder + model_name + '/output/forcast.png', bbox_inches = 'tight')

def load_model(model_name:str, architecture:str) -> Union[my_LSTM, my_GRU]:
    """Load the model by name
    
    Takes the name of the model, and the type of model (gru, lstm) and assembles the architecture of the network
    loads the weights of the network, the scalar used for normalization, and the network properties
    
    Input: 
        model_name (str): The name of the model you want to load
        architecture (str): should be one of the following types of RNN ('gru', 'lstm')
        
    Output:
        architectures.models.[my_GRU, my_LSTM]: Retruns the network 
        input_shape (np.array): Input shape of the network"""
    model_save_location = "../trained_models/"
    if os.path.exists(model_save_location + model_name + "/"):
        with open(model_save_location + model_name + "/network_props.pkl", 'rb') as f:
            units, input_shape = pickle.load(f)
        if architecture == 'gru':
            model = my_GRU(units=units, input_shape=input_shape, model_name=model_name, scaler=MinMaxScaler())
            
        else:
            model = my_LSTM(units=units, input_shape=input_shape, model_name=model_name, scaler=MinMaxScaler())

        model.load(model_save_location)

    else:
        print('This model does not exist')
        model = None
        input_shape = 0
    return model, input_shape

def snip_timeframe(X:pd.DataFrame, look_back:int) -> pd.DataFrame:
    """Takes a slice of the data
    
    Takes the last N data points from the data and returns the sliced dataframe
    
    Input: 
        X (pd.DataFrame) : The dataframe that you want to extract the latest data required to predict the next 24 hrs
        look_back (int) : Should be the look_back of the model you're snipping the data for, can be found as model.input_shape[0]
        
    Output 
        trim (pd.DataFrame) : The trimmed dataframe only including the latest N points
    """
    T = len(X.iloc[:,1])
    Ts = T - (look_back + 48) 
    trim = X.iloc[Ts:, :]
    return trim

    
    
