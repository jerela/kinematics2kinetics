import matplotlib.pyplot as plt

def plotter(data,titles):
    n = len(data)
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.plot(data[i])
        plt.title(titles[i])
    plt.show()
    



