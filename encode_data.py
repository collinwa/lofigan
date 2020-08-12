from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os 
import string
import pydub
import datetime
import scipy.sparse

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

if __name__ == '__main__':
    encoder = OneHotEncoder(categories=[CATEGORIES], handle_unknown='ignore')
    dirs_to_process = ['./lofibeats_3']
    out_dir = './sparse_matrices'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    total_processed = 0
    all_tracks = sum([len(os.listdir(x)) for x in dirs_to_process])
    for songdir in dirs_to_process:
        for f in os.listdir(songdir):
            data = preprocess_audio(os.path.join(songdir, f))
            data = one_hot(encoder, data)
            song_id = datetime.datetime.now().timestamp()
            scipy.sparse.save_npz(os.path.join(out_dir, 'song_%i.sparse.npz' % song_id), data)
            total_processed += 1
            print('Processed %i/%i total tracks (%s just processed)' % (total_processed, all_tracks, os.path.join(songdir, f)))

