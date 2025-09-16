import os
import matplotlib.pyplot as plt
from options import path_output

# save the target and prediction time series of the first 9 (or less) samples from the validation set
def save_sample_figure(data, name):
    n_samples = len(data)
    fig = plt.figure()
    for i_sample in range(n_samples):
        ts = data[i_sample]['time_series']
        target = ts[0]
        prediction = ts[1]
        loss = data[i_sample]['loss']
        fig.add_subplot(3, 3, i_sample+1)
        plt.plot(target, 'b', label='target')
        plt.plot(prediction, 'r', label='prediction')
        plt.title(f'Loss: {loss:.4f}')
    plt.legend()
    full_path = os.path.join(path_output, 'Figures', f'samples_{name}.png')
    plt.savefig(full_path)

# save training and validation losses as subplots in one figure
def save_loss_figure(losses, name):
    loss_training, loss_validation = losses
    
    fig = plt.figure()
    fig.add_subplot(121)
    plt.plot(loss_training)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training loss')
    fig.add_subplot(122)
    plt.plot(loss_validation)
    plt.xlabel('epoch')
    plt.title('Validation loss')
    
    full_path = os.path.join(path_output, 'Figures', f'losses_{name}.png')
    plt.savefig(full_path)
    

class Plotter():
    
    def __init__(self, sequence_length=None):
        plt.ion()
        self.fig = plt.figure()
        # if sequence length is not specified, we assume there are 101 data points in a single sequence
        if sequence_length is None:
            self.xlim = (1, 101)
        else:
            self.xlim = (1, sequence_length)
        
    def plot_losses(self, losses, titles):
        n = len(losses)
        for i in range(n):
            self.fig.add_subplot(1, n, i+1)
            plt.plot(losses[i])
            plt.title(titles[i])
            plt.axis([1, len(losses[i]), 0, max(losses[i])])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
    def plot_samples(self, data, titles):
        n = len(data)
        for i in range(n):
            self.fig.add_subplot(1, n, i+1)
            plt.plot(data[i])
            plt.title(titles[i])
            plt.axis([self.xlim[0], self.xlim[1], -1.0, 1.0])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        



