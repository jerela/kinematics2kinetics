import matplotlib.pyplot as plt


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
        



