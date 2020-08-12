from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os 
import sys
import string
import pydub
import datetime
import scipy.sparse
from train_keras import *  # for hyperparameters
pred_size = 128204
CATEGORIES = np.arange(0, 256)

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

def trim_uneven(data, pred_region):
    # trim off first 5 seconds to eliminate introductions
    data = data[220500:,:]

    # calculate additional time to trim
    to_trim = data.shape[0] % pred_region
    return data[to_trim:,:], int(np.floor(data.shape[0] / pred_region))

if __name__ == '__main__':
    encoder = OneHotEncoder(categories=[CATEGORIES], handle_unknown='ignore')
    dirs_to_process = ["./lofibeats"]
    out_dir = './sparse_matrices'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    total_processed = 0
    all_tracks = sum([len(os.listdir(x)) for x in dirs_to_process])
    lost_timesteps = seq_len - pred_size

    for songdir in dirs_to_process:
        for f in os.listdir(songdir):
            data = preprocess_audio(os.path.join(songdir, f))
            # second half is unusable
            data = one_hot(encoder, data)[:int(data.shape[0]/2),:].toarray()
            data, max_idx = trim_uneven(data, pred_size)

            all_x = []
            all_y = []
            for i in range(0, max_idx):
                # seq_len timesteps for x
                all_x.append(data[i:i+seq_len,:])
                # pred_size timesteps for y, shifted over by time_shift and lost_timesteps
                all_y.append(data[i+lost_timesteps+time_shift:i+lost_timesteps+time_shift+pred_size,:])
                if (i > 3):
                    break
            all_x = np.concatenate(all_x, axis=0)
            all_y = np.concatenate(all_y, axis=0)

            song_id = datetime.datetime.now().timestamp()
            scipy.sparse.save_npz(os.path.join(out_dir, '%i_X.sparse.npz' % song_id), sparse.csr_matrix(all_x))
            scipy.sparse.save_npz(os.path.join(out_dir, '%i_Y.sparse.npz' % song_id), sparse.csr_matrix(all_y))

            total_processed += 1
            sys.exit(0)
            print('Processed %i/%i total tracks (%s just processed)' % (total_processed, all_tracks, os.path.join(songdir, f)))

