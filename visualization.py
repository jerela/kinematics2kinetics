import matplotlib.pyplot as plt


class Plotter():
    
    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        
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
            plt.axis([1, 101, -1.5, 1.5])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        



