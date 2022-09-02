import sys
sys.path.append('../')

from utils.utils import *
from architectures.models import *
from sklearn.metrics import mean_squared_error, mean_absolute_error

save_folder = "../trained_models"

# Read the data
def train_model(model_type:str, target_var:str, met_data_location:str, open_data_location:str, 
        energy_data_location:str, look_back:int, model_name:str, units:int, epochs:int, 
        val_split:float, batch_size:int) -> Union[my_GRU,my_LSTM]:
    
    """Train a model

    Transforms the data available, and processes so that it's usable. Shifts the target variable by 1 day, splits the data
    into training (80%) and testing (20%). Normalizes the data from the network, generated the 3D tensor, decides on the network type
    and trains the model and plots the training curve and the forcast of the testing data overlayed with the ground truth.

    Input:
        model_type (str): The type of model ('gru', 'lstm') 
        target_var (str): The data you want to train on ('w', 's') -> Wind, Solar
        met_data_location (str): The file path to the met office data 
        open_data_location (str): The file path to the open weather data 
        energy_data_location (str): The file path to the energy onsite data
        look_back (int): The length of each training sample (T-look_back -> T)
        model_name (str): The name of the model
        units (int): The number of rnn units within the network 
        epochs (int): Max number of epochs allowed for training
        val_split (float): portion of the data used for validation  
        batch_size (int): How many examples are used per iteration (after how many examples should the model update)

    Output:
        architectures.models.[my_GRU, my_LSTM] : The trained model
        eval (np.array): returns - [mse (mean square error), mae (mean absolute error, msle (mean squared log error)]
            on the training section of the data

    """

    # read the data
    m, o, e = read_data(met_data_location), read_data(open_data_location), read_data(energy_data_location)

    # pre-process the data from training
    if target_var == 'w':
        data, _ = preprocess_data(m, o, e) 
    else:
        _, data = preprocess_data(m, o, e)

    data = shift_target(data) # shift target by 24hrs

    # split the data to training and testing
    tr_data, te_data = test_train_split(data, 0.8)

    # Normalise the data
    scalar = MinMaxScaler().fit(tr_data)
    tr_norm = scalar.transform(tr_data)
    te_norm = scalar.transform(te_data)

    # Restructure the dataset into a 3D tensor for the network
    X, y = create_dataset(tr_norm, look_back=look_back)
    input_shape = [X.shape[1], X.shape[2]]

    # train the network
    if model_type == 'gru':
        model = my_GRU(units=units, input_shape=input_shape, model_name=model_name,  scaler=scalar)
    else:
        model = my_LSTM(units=units, input_shape=input_shape, model_name=model_name, scaler=scalar)
    
    history = model.train(X, y, epochs=epochs, val_split=val_split, batch_size=batch_size)
    model.save(save_folder)

    # plot the loss of the training process and store it in ../trained_model/{model_name}/output/loss.png
    plot_loss(history=history, model_name=model_name)

    # test on the unseen data
    Xte, yte = create_dataset(te_norm, look_back=look_back)
    y_hat = model.forcast(Xte)

    # show plot of the predicion against the ground truth
    y_gt = reverse_normlisation(yte, scalar)
    plot_future(y_hat, model_name, y_gt)

    # calculate and store the evaluation metrics somewhere
    #eval_metrics = model.model.evaluate()
    mse, mae = mean_squared_error(y_gt, y_hat), mean_absolute_error(y_gt, y_hat)

    eval = np.array([mse, mae, msle])

    

    return model, eval












