import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os 
import pydub
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
    mu_transform[mu_transform >= 0] = mu_transform[mu_transform >=0 ] * 127
    mu_transform[mu_transform < 0] = mu_transform[mu_transform < 0] * 128
    return mu_transform.astype('int32') + 127

def one_hot(encoder, data):
    data = np.squeeze(data)[:,None]
    return encoder.fit_transform(data)

if __name__ == '__main__':
    encoder = OneHotEncoder(categories=[CATEGORIES], handle_unknown='ignore')
    dirs_to_process = ['./lofibeats', './lofibeats_2']
    out_dir = './sparse_matrices'
    total_processed = 0
    all_tracks = sum([len(os.listdir(x)) for x in dirs_to_process])
    for songdir in dirs_to_process:
        for f in os.listdir(songdir):
            data = preprocess_audio(os.path.join(songdir, f))
            data = one_hot(encoder, data)
            scipy.sparse.save_npz(os.path.join(out_dir, 'song_%i.sparse.npz' % total_processed), data)
            total_processed += 1
            print('Processed %i/%i total tracks (%s just processed)' % (total_processed, all_tracks, os.path.join(songdir, f)))

