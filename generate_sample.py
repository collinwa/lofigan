import scipy.sparse
import random
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, Layer
from tensorflow.keras.activations import tanh, sigmoid, relu, softmax
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as sio

def random_cut(audio, duration):
    return random.randint(0, audio.shape[0] - duration)

def inverse_mu(arr, mu_vaue=256):
    arr = arr.astype(np.float64)
    arr = (arr - 127)
    arr[arr >= 0] = arr[arr >= 0] / 127
    arr[arr < 0] = arr[arr < 0] / 128

    sign = np.sign(arr)
    base = np.zeros_like(arr)
    base.fill(255)
    recip = np.reciprocal(base)
    power = np.power(base+1, np.abs(arr)) - 1
    out = np.multiply(sign, recip)
    out = np.multiply(out, power)
    return out

if __name__ == '__main__':
    base_dir = sys.argv[1]
    save_dir = sys.argv[2]

    training_set = sorted(os.listdir(base_dir))[0:140]
    validation_set = sorted(os.listdir(base_dir))[140:]

    model = tf.saved_model.load(os.path.join(save_dir, 'model_epoch_0'))
    f = random.choice(validation_set)
    start_audio = scipy.sparse.load_npz(os.path.join(base_dir, f))

    num_steps = 10
    duration = 44100

    start = random_cut(start_audio, duration)
    start_audio = start_audio[start:start+duration, :].toarray()[None,:,:] # batch dimension here
    start_audio = tf.convert_to_tensor(start_audio)

    prev_pred = None
    depth = 256
    for i in range(0, num_steps):
        prev_timesteps = tf.cast(start_audio[:,-duration:,:], tf.float32)
        prediction = model(prev_timesteps)
        prediction = prediction[0,:,:]
        prediction = tf.random.categorical(prediction, 1)
        if prev_pred is None:
            prev_pred = prediction
        else:
            prev_pred = tf.concat([prev_pred, tf.cast(prediction, tf.int64)], axis=0)
        prediction = prediction[:,0]
        prediction = tf.one_hot(prediction, depth)[None,:,:]
        start_audio = tf.concat([start_audio, tf.cast(prediction, tf.float64)], axis=1)


    new_audio = prev_pred.numpy()
    plt.hist(new_audio[:,0], bins=200)
    plt.show()

    transformed_audio = inverse_mu(new_audio)
    plt.hist(transformed_audio, bins=200)
    plt.show()

    transformed_audio = transformed_audio * 2 ** 16
    transformed_audio = transformed_audio.astype(np.int16)

    sio.write(os.path.join(home_dir, 'test_gen_2.wav'), 44100, transformed_audio)
