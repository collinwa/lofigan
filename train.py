import tensorflow as tf
import scipy.sparse as sparse
import random
import os
from tensorflow.keras.optimizers import Adam

def random_cut(audio, duration):
    return random.randint(0, audio.shape[0] - duration)

if __name__ == '__main__':
    duration = 16384
    epochs = 50
    lr=1e-3
    batch_size=1
    loss = tf.nn.softmax_cross_entropy_with_logits

    training_set = sorted(os.listdir(base_dir))[0:140]
    validation_set = sorted(os.listdir(base_dir))[140:]
    optimizer = Adam(learning_rate=lr)
    model = WaveNet(batch=batch_size, duration=duration)
    end_cut_size = model.final_dim
    cur_losses = []
    with tf.device("/GPU:0"):
      for epoch in range(epochs):
          for i, f in enumerate(training_set):
              print('Loading %s in TimeStep %i' % (f, i))
              audio = sparse.load_npz(os.path.join(base_dir, f))
              print('loaded')

              start = random_cut(audio, duration)
              audio = audio[start:start+duration,:].toarray()
              audio = audio[None,:,:]

              audio = tf.convert_to_tensor(audio)
              
              with tf.GradientTape() as tape:
                  logits = model(audio)
                  cur_loss = loss(audio[:,-end_cut_size:,:], logits)

              grads = tape.gradient(cur_loss, model.trainable_weights)
              optimizer.apply_gradients(zip(grads, model.trainable_weights))

              cur_loss = tf.stop_gradient(cur_loss)
              cur_losses.append(tf.math.reduce_mean(cur_loss))

              print('Epoch {} Timestep {} Current Loss {}'.format(epoch, i, tf.math.reduce_mean(cur_loss)))

          model.save(os.path.join(save_dir, 'model_epoch_%i' % epoch), save_format='tf')
