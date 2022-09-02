# models-training
Training code for the models.

There are 2 model architectures we have investigated, 'GRU' and 'LSTM'. Both models have a simple design consisting of 1 RNN layer, one hidden layer (Dense) and an output layer. We firstly 
pre-processed the data to remove discontinuity, insert synthetic data, and synchronise the 3 input databases. Once the datasets were synchronised, they were consolidated them into their respective targets, 
i.e. wind or solar. 

In order to choose the best performing model, an experiment was conducted. A 'GRU' and 'LSTM' network were created for each of the target variables and by varying the timestep, batch size, 
and the number of units within the RNN layer of the network. Around 600 models were trained and tested, and each compared on the MSE (Mean Squared Error) evaluation  metric. 

The results of the experiment can be found in the model_experiment folder, unfortunately they were not up to scratch. From this we have decided to continue using the slimjab model.

The performance of each model on unseen data can be found in the trained_models folder, where each model contains an output folder displaying the model prediction vs ground truth. 
