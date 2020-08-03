import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.activations import tanh, sigmoid, relu, softmax

class WaveNet(Model):
    def __init__(self,
            batch=1,
            duration=16384,
            first_ch=256,
            block_ch=24,
            scaled_ch=128,
            input_ch=256,
            channels='channels_last', 
            dim=2,
            max_dilation=10):
        super(WaveNet, self).__init__()
        self.residual_array = []
        self.residual_scale = []
        self.causal_conv = Conv1D(first_ch, dim)
        for i in range(max_dilation):
            self.residual_array.append(ConvGatedDilation(out_ch=block_ch, dim=dim, dilation_rate=pow(2, i)))
            self.residual_scale.append(Conv1D(scaled_ch, 1))

        self.final_dim = duration - pow(2, max_dilation) + 1
        self.batch_dim = batch
        self.input_ch = input_ch
        self.final_conv1 = Conv1D(input_ch, 1)
        self.final_conv2 = Conv1D(input_ch, 1)


    def call(self, x):
        final_accumulator = tf.zeros((self.batch_dim, self.final_dim, self.input_ch))
        prev_in = None 
        out = self.causal_conv(x)
        for res, conv in zip(self.residual_scale, self.residual_array):
            prev_in = out
            out = conv(out)
            to_trim = prev_in.shape[1] - out.shape[1]
            print(to_trim)
            if to_trim % 2 == 1:
                left = to_trim // 2 + 1
                right = -to_trim // 2
            else:
                left = to_trim // 2
                right = -to_trim // 2
            
            if (prev_in.shape[-1] == out.shape[-1]):
                final_accumulator =  res(prev_in[:,left:right,:] + out)
 
        final_accumulator = relu(final_accumulator)
        final_accumulator = relu(self.final_conv1(final_accumulator))
        final_accumulator = softmax(self.final_conv2(final_accumulator))
        return final_accumulator


class ConvGatedDilation(Model):
    def __init__(self, out_ch=24, dim=2, dilation_rate=1):
        super(Model, self).__init__()
        self.gate_conv = Conv1D(out_ch, dim, dilation_rate=dilation_rate)
        self.filter_conv = Conv1D(out_ch, dim, dilation_rate=dilation_rate)
        self.scale_conv = Conv1D(out_ch, 1)

    def call(self, x):
        g = sigmoid(self.gate_conv(x))
        f = tanh(self.filter_conv(x))
        x = tf.math.multiply(f, g)
        x = self.scale_conv(x)
        return x

