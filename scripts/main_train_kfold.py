import time
import torch
from helpers_train_test import run_kfold_gru, run_kfold_cnn, run_kfold_cnn2d, run_kfold_lstm, run_kfold_cnnlstm, run_kfold_mlstmfcn, run_kfold_xformer
from options import rng_seed, batch_size, plot_losses, plot_sample, workers, path_output

torch.manual_seed(rng_seed)



def main():
    start = time.time()
    #run_kfold_gru()
    run_kfold_cnn()
    #run_kfold_cnn2d()
    #run_kfold_lstm()
    #run_kfold_xformer()
    #run_kfold_cnnlstm()
    #run_kfold_mlstmfcn()
    print(f'The program finished in {time.time()-start} seconds.')
    
if __name__ == "__main__":
    main()




