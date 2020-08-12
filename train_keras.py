import random
import os
import scipy.sparse as sparse
import tensorflow as tf 
from tensorflow.keras.optimizers import dam
from WaveNetKerasFunctional import create_model

random.seed(1337)

if __name__ == '__main__':
    # define hyperparameters
    seq_len = 132300  # 3 second long input
    in_ch = 256  # 256 input channels
    causal_ch = 512  # first causal conv will produce 512 channels
    stack_ch = 128  # each causal stack will produce 128
    scaled_ch = 128
    filter_dim = 2  # filter dimension
    max_dilation = 10  # maximum dilation
    n_stacks = 4  # number of causal conv stacks
    time_shift = 1  # timesteps to shift 

    # generate all file list and randomly permute
    data_dir = "./sparse_matrices"
    all_files = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, x) for x in all_files]
    random.shuffle(all_files)
    
    # training / test split
    val_pct = 0.1
    train_pct = 1. - val_pct     
    train_qty = int(len(os.listdir(data_dir)) * train_pct)
    training_set = all_files[:train_qty]
    validation_set = all_files[train_qty:]
    
    # create the model
    model, pred_size = create_model(seq_len=seq_len, 
            in_ch=in_ch,
            causal_ch=causal_ch,
            stack_ch=stack_ch,
            scaled_ch=scaled_ch,
            filter_dim=filter_dim,
            max_dilation=max_dilation,
            n_stacks=n_stacks)

