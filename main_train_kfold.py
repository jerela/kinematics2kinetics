

import torch

from helpers_train_test import run_kfold_cnn
from options import rng_seed, batch_size, plot_losses, plot_sample, workers, path_output



torch.manual_seed(rng_seed)





def main():
    run_kfold_cnn()
    
if __name__ == "__main__":
    main()




