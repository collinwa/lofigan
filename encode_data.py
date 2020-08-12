from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os 
import sys
import string
import pydub
import datetime
import scipy.sparse
from train_keras import *  # for hyperparameters
from multiprocessing import Pool
from functools import partial
from string import ascii_lowercase, ascii_uppercase

pred_size = 128204
CATEGORIES = np.arange(0, 256)

def process_audio(f, 
        out_dir='./sparse_matrices', 
        pred_size=128204, 
        seq_len=132300,
        categories=np.arange(0, 256)):

    def preprocess_audio(f, encoding='mp4'): 
        segment = pydub.AudioSegment.from_file(f, encoding)
        data = np.array(segment.get_array_of_samples(), dtype=np.float32) / 2 ** 16

        # stereo data, convert to mono by averaging the channels
        if (segment.channels == 2):
            data = data.reshape((-1, 2))
            data = (data[:,0] + data[:,1]) / 2

        mu_transform = np.multiply(np.sign(data), np.log(1 + 255 * np.abs(data)) / np.log(256))
        # scale values from -128 to 127, inclusive
        mu_transform[mu_transform >= 0] = mu_transform[mu_transform >=0 ] * 127
        mu_transform[mu_transform < 0] = mu_transform[mu_transform < 0] * 128
        # shift values from 0 - 255, inclusive
        return mu_transform.astype('int32') + 128

    def one_hot(encoder, data):
        data = np.squeeze(data)[:,None]
        return encoder.fit_transform(data)

    def trim_uneven(data, pred_region, seq_len=132300):
        # trim off first 5 seconds to eliminate introductions
        data = data[220500:,:]

        # calculate additional time to trim
        to_trim = data.shape[0] % pred_region
        return data[to_trim:,:], int(data.shape[0]-seq_len-2)  # can we get away with a -1 here?
    
    print("Processing %s" % f)
    # use encoder, process audio
    encoder = OneHotEncoder(categories=[categories], handle_unknown='ignore')
    data = preprocess_audio(f)

    # second half is unusable
    data = one_hot(encoder, data)[:int(data.shape[0]/2),:].toarray()
    data, max_idx = trim_uneven(data, pred_size)
    
    # create ndarrays 
    all_x = []
    all_y = []
    for i in range(0, max_idx):
        # seq_len timesteps for x
        all_x.append(data[i:i+seq_len,:])
        # pred_size timesteps for y, shifted over by time_shift and lost_timesteps
        all_y.append(data[i+lost_timesteps+time_shift:i+lost_timesteps+time_shift+pred_size,:])
        if (i > 3):
            break
    # run all_x and all_y
    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    
    # random character identifier
    id_len = 5
    alphabet = ascii_lowercase + ascii_uppercase
    song_id = str(datetime.datetime.now().timestamp()) + ''.join(random.sample(alphabet, k=id_len))
    scipy.sparse.save_npz(os.path.join(out_dir, '%s_X.sparse.npz' % song_id), sparse.csr_matrix(all_x))
    scipy.sparse.save_npz(os.path.join(out_dir, '%s_Y.sparse.npz' % song_id), sparse.csr_matrix(all_y))

    total_processed += 1
    print('%s just processed' % (f))


if __name__ == '__main__':
    dirs_to_process = ["./lofibeats", "./lofibeats_2", "./lofibeats_3"] 
    all_files = []
    out_dir = './sparse_matrices'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    total_processed = 0
    lost_timesteps = seq_len - pred_size 
    
    for d in dirs_to_process:
        all_files += [os.path.join(d, f) for f in os.listdir(d)]
    
    print('Running pool of workers')
    workers = Pool(processes=1)
    f = partial(process_audio, 
            out_dir=out_dir,
            pred_size=pred_size,
            categories=CATEGORIES,
            seq_len=seq_len)
    workers.map(f, all_files)

