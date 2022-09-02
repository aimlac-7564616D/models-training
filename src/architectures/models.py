from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM
from tensorflow.keras.metrics import *
import joblib
import pickle
import os
import numpy as np


class my_GRU:
    """Class that creates a fixed GRU network
    
        The class creates a sequential keras model with 1 GRU layer with units={units}, followed by 1 dense layers into 1
        ouptput node. Using the 'adam' opimizer and mean-squared-error as the loss function

        Input:
            units (int): The number of GRU units
            input_shape (np.array): The shape of the input into the network [look_back, features]
            model_name (str): Name of the model
            scaler (MinMaxScalar): Used to normalize the data
        
        Returns:
            keras.model.Sequential() : Returns an untrained, compiled model
    """

    def __init__(self, units, input_shape, model_name, scaler):
        self.model = Sequential()
        self.model.add(GRU(units=units, input_shape = input_shape))
        self.model.add(Dense(int(units/2)))
        self.model.add(Dense(1))
        #compile the model
        self.model.compile(optimizer='adam', loss='mse', metrics=[
            RootMeanSquaredError(),
            MeanAbsoluteError(),
        ])
        self.scaler = scaler
        self.model_name = model_name
        self.units = units
        self.input_shape = input_shape


    def train(self, X, y, epochs, val_split, batch_size):
        """Train the network

        Takes data and uses it to train the network, using the provided training paramaters

        Inputs:
            X (np.array): Input data for the network, should be a 3D tensor of size = (N, look_back, features)
            y (np.array): Target variable of the network
            epochs (int): Max number of epochs for trainging
            val_split (float): Percent split of the validation set
            batch_size(int): Batch size during training

        returns:
            History : Loss and validation loss through training, recorded after every epoch of training
        """
        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 10)
        history = self.model.fit(X, y, epochs = epochs,  
                        validation_split = val_split,
                        batch_size = batch_size, shuffle = False, 
                        callbacks = [early_stop])
        return history

    def forcast(self, X):
        """Use the network to forcast/predict future values

        Method takes in data and predicts the y_hat values 

        Input:
            X (np.array): Input data to the network, should be of shape = (look_back, features)

        Returns:
            y_hat (np.array): Un-normalized output of the network, the predicted values.  Shape = (1, N)
        """
        y_hat = self.model.predict(X)
        n_features = self.scaler.n_features_in_
        toReturn = np.zeros(shape=(len(y_hat), n_features))
        toReturn[:,-1] = y_hat.reshape(1,-1)
        toReturn = self.scaler.inverse_transform(toReturn)
        return toReturn[:,-1]

    
    def save(self, save_folder:str) -> None:
        """Saves the model 

        Saves the model weights, scalar used to normalize the data, and the netowrk properties (units, input_shape)

        Input:
            save_folder (str): The folder path you would like to save to 
        """
        save_location = save_folder + '/' + self.model_name
        if not os.path.exists(save_location):
            os.mkdir(save_location)

        with open(save_location + '/network_props.pkl', 'wb') as f:
            pickle.dump([self.units, self.input_shape], f)
        self.model.save_weights(save_location+ '/weights')
        joblib.dump(self.scaler, save_location + '/scaler')

    
    def load(self, load_folder:str) -> None:
        """loads a model

        Loads the models weights and scalar for a file location - should be kept static

        Input:
            load_folder (str) : The folder where you would find a folder with the model name, inside of which will be weights file and scalar file

        Returns:
            (None)
        """
        load_location = load_folder + '/' + self.model_name
        if os.path.exists(load_location):
            self.model.load_weights(load_location + '/weights')
            self.scaler = joblib.load(load_location + '/scaler') 
        else:
            print("Model: {self.model_name} does not exist in the provided directory")


class my_LSTM:
    """Class that creates a fixed LSTM network
    
        The class creates a sequential keras model with 1 LSMT layer with units={units}, followed by 1 dense layers into 1
        ouptput node. Using the 'adam' opimizer and mean-squared-error as the loss function

        Input:
            units (int): The number of LSTM units
            input_shape (np.array): The shape of the input into the network [look_back, features]
            model_name (str): Name of the model
            scaler (MinMaxScalar): Used to normalize the data
        
        Returns:
            keras.model.Sequential() : Returns an untrained, compiled model
    """

    def __init__(self, units, input_shape, model_name, scaler):
        self.model = Sequential()
        self.model.add(LSTM(units=units,  input_shape=input_shape))
        self.model.add(Dense(int(units/2)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse', metrics=[
            RootMeanSquaredError(),
            MeanAbsoluteError(),
        ])
        self.scaler = scaler
        self.model_name = model_name
        self.units = units
        self.input_shape = input_shape

    def train(self, X, y, epochs, val_split, batch_size):
        """Train the network

        Takes data and uses it to train the network, using the provided training paramaters

        Inputs:
            X (np.array): Input data for the network, should be a 3D tensor of size = (N, look_back, features)
            y (np.array): Target variable of the network
            epochs (int): Max number of epochs for trainging
            val_split (float): Percent split of the validation set
            batch_size(int): Batch size during training

        returns:
            History : Loss and validation loss through training, recorded after every epoch of training
        """
        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 20)
        history = self.model.fit(X, y, epochs = epochs,  
                        validation_split = val_split,
                        batch_size = batch_size, shuffle = False, 
                        callbacks = [early_stop])
        return history

    def forcast(self, X):
        """Use the network to forcast/predict future values

        Method takes in data and predicts the y_hat values 

        Input:
            X (np.array): Input data to the network, should be of shape = (look_back, features)

        Returns:
            y_hat (np.array): Un-normalized output of the network, the predicted values.  Shape = (1, N)
        """
        y_hat = self.model.predict(X)
        n_features = self.scaler.n_features_in_
        toReturn = np.zeros(shape=(len(y_hat), n_features))
        toReturn[:,-1] = y_hat.reshape(1,-1)
        toReturn = self.scaler.inverse_transform(toReturn)
        return toReturn[:,-1]
    
    def save(self, save_folder:str) -> None:
        """Use the network to forcast/predict future values

        Method takes in data and predicts the y_hat values 

        Input:
            X (np.array): Input data to the network, should be of shape = (look_back, features)

        Returns:
            y_hat (np.array): Un-normalized output of the network, the predicted values.  Shape = (1, N)
        """
        save_location = save_folder + '/' + self.model_name
        if not os.path.exists(save_location):
            os.mkdir(save_location)

        with open(save_location + '/network_props.pkl', 'wb') as f:
            pickle.dump([self.units, self.input_shape], f)
        self.model.save_weights(save_location+ '/weights')
        joblib.dump(self.scaler, save_location + '/scaler')
    
    def load(self, load_folder:str) -> None:
        """loads a model

        Loads the models weights and scalar for a file location - should be kept static

        Input:
            load_folder (str) : The folder where you would find a folder with the model name, inside of which will be weights file and scalar file

        Returns:
            (None)
        """
        load_location = load_folder + '/' + self.model_name
        if os.path.exists(load_location):
            self.model.load_weights(load_location + '/weights')
            self.scaler = joblib.load(load_location + '/scaler') 
            
        else:
            print("Model: {self.model_name} does not exist in the provided directory")

    
