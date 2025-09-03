import matplotlib.pyplot as plt


class Plotter():
    
    def __init__(self):
        plt.ioff()
        self.fig = plt.figure()
        
    def plot(self, data, titles):
        with plt.ion():
            n = len(data)
            for i in range(n):
                self.fig.add_subplot(1, n, i+1)
                plt.plot(data[i])
                plt.title(titles[i])
                plt.axis([1, 101, -1.5, 1.5]) 
            plt.show()
            plt.pause(1e-9)
        



