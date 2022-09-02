import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_results(names:str, evals:np.array, title:str, save_name:str) -> None:
    plt.figure(figsize=(110, 10))
    x = np.arange(len(names))
    plt.xticks(x, names, rotation=70)
    plt.plot(x, np.ones(len(x))*np.min(evals[:, 0]), 'r')
    plt.plot(x, evals[:,0], label='mse')
    plt.plot(x, evals[:,1], label='mae')
    plt.grid(True)
    plt.title(title)
    plt.legend(loc='upper left')
    plt.xlabel('Model Name')
    plt.ylabel('MSE / MAE')
    plt.savefig(save_name, bbox_inches = 'tight')

with open('results.pkl', 'rb') as f:
    results = pickle.load(f)

n = len(results)
half = int(n/2)
quarter = int(n/4)

gru_w, gru_s, lstm_w, lstm_s = [], [], [], []
gru_w_n, gru_s_n, lstm_w_n, lstm_s_n = [], [], [], []

for i, key in enumerate(results.keys()):
    eval = results[key]
    if i < quarter:
        gru_w.append([eval['mse'],eval['mae']])
        gru_w_n.append(key)
    elif i < half:
        gru_s.append([eval['mse'],eval['mae']])
        gru_s_n.append(key)
    elif i < half + quarter:
        lstm_w.append([eval['mse'],eval['mae']])
        lstm_w_n.append(key)
    else:
        lstm_s.append([eval['mse'],eval['mae']])
        lstm_s_n.append(key)

gru_w = np.array(gru_w)
gru_s = np.array(gru_s)
lstm_w = np.array(lstm_w)
lstm_s = np.array(lstm_s)

plot_results(gru_w_n, gru_w, 'GRU Wind (MSE)', 'gru_w.png')
plot_results(gru_s_n, gru_s, 'GRU Solar (MSE)', 'gru_s.png')
plot_results(lstm_w_n, lstm_w, 'LSTM Wind (MSE)', 'lstm_w.png')
plot_results(lstm_s_n, lstm_s, 'LSTM Solar (MSE)', 'lstm_s.png')
