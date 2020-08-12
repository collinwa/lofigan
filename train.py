import random
import tensorflow as tf
import scipy.sparse as sparse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax
from parameters import *

def random_cut(audio, duration):
        return random.randint(0, audio.shape[0] - duration)

def get_max_t(audio, duration):
    return audio.shape[0] - (audio.shape[0] % duration) - 1

if __name__ == '__main__':
    loss = tf.nn.softmax_cross_entropy_with_logits

    training_set = sorted(os.listdir(base_dir))[0:140]
    validation_set = sorted(os.listdir(base_dir))[140:]

    optimizer = Adam(learning_rate=lr)
    model = WaveNet(batch=batch_size, duration=duration, max_dilation=max_dilation)
    cur_losses = []
    total_counter=0
    time_steps = 1

    with tf.device("/GPU:0"):
        for epoch in range(epochs):
            for i, f in enumerate(training_set):
                print('Loading %s in TimeStep %i' % (f, i))

                # load the whole sparse audio matrix
                audio = sparse.load_npz(os.path.join(base_dir, f))

                # maximum timestep that we can predict with
                max_t = get_max_t(audio, duration)

                # get the previous timestep to predict the next
                # prev_audio = tf.convert_to_tensor(audio[0:duration,:].toarray())[None,:,:]
                jump_size=random.randint(10, 22050)
                this_its = 0
                for j in range(0, (max_t-1), jump_size):
                    # align the original y
                    prev_audio = audio[j:j+duration,:]
                    prev_audio = tf.convert_to_tensor(prev_audio.toarray())[None,:,:]

                    # shift the step size to be aligned + 100 timesteps
                    next_y = audio[j+pow(2, max_dilation)+time_steps:j+pow(2, max_dilation)+pred_size+time_steps,:]
                    next_y = tf.convert_to_tensor(next_y.toarray())[None,:,:]

                    # run backwards
                    with tf.GradientTape() as tape:
                            logits = model(prev_audio)
                            cur_loss = loss(next_y, logits)

                    # compute gradients and backprop
                    grads = tape.gradient(cur_loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))

                    # track the mean loss 
                    cur_loss = tf.stop_gradient(cur_loss)
                    cur_losses.append(tf.math.reduce_mean(cur_loss))
                    total_counter += 1
                    this_its += 1
                    if total_counter % 100 == 0:
                        print('Epoch {} Timestep {} Current Loss {}'.format(epoch, total_counter, tf.math.reduce_mean(cur_loss)))

                model.save(os.path.join(save_dir, 'wavenet_tmpsave_withjumps'), save_format='tf')

        model.save(os.path.join(save_dir, 'model_epoch_%i' % epoch), save_format='tf')

