# code for defining functional model 
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Concatenate, Conv1D, Layer, Dense, Lambda, Multiply, Add, ReLU
import tensorflow.keras.backend as kb

def nll_loss(y_true, y_pred):
    # softmax for y_pred
    y_pred = kb.softmax(y_pred, axis=-1)
    # elementwise product
    loss = y_true * y_pred
    # elementwise logarithm
    loss = kb.log(loss)
    # ll is sum of log probabilities
    # sum across the channels and t dimension
    loss = kb.sum(loss, axis=-1)
    loss = kb.sum(loss, axis=-1)
    # convert ll to nll
    return -1 * loss


def create_model(seq_len=44100, in_ch=256, causal_ch=512, stack_ch=64,
        scaled_ch=128, filter_dim=2, max_dilation=10, n_stacks=1):

    # no slicing in keras, this will return slice_size timesteps from the end to the end
    def slice_end(slice_size):
        trim_func = lambda x, slice_size=slice_size: x[:,-slice_size:,:]
        return Lambda(trim_func)

    # this will return slice_size timesteps from the front to the end
    def slice_front(slice_size):
        trim_func = lambda x, slice_size=slice_size: x[:,slice_size:,:]
        return Lambda(trim_func)

    # we lose a total of 2^n values due to padding: 1 from the first convolution, then 1+2+...+2^n-1 = 2^n - 1 from the dilated conv; 2^max_dilation per stack of causal convs
    pred_size = seq_len - 1 - (pow(2, max_dilation)-1) * n_stacks

    # create the input tensor; ignore batch dimension
    inputs = keras.Input(shape=(seq_len, in_ch))

    # first causal convolution; (, T, causal_ch)
    x = Conv1D(causal_ch, filter_dim, activation='relu')(inputs)

    output = None

    for n in range(n_stacks):
        for i in range(max_dilation):
            # forget get from the input, produces a (,T, stack_ch) tensor
            forget_gate = Conv1D(stack_ch, 
                    filter_dim, 
                    activation='sigmoid',
                    dilation_rate=pow(2, i))(x)
            value_gate = Conv1D(stack_ch, 
                    filter_dim, 
                    activation='tanh',
                    dilation_rate=pow(2, i))(x) 

            # hadamard product between value and forget gate
            new_x = Multiply()([value_gate, forget_gate])

            # use the Conv1D to obtain a re-scaling
            out_x = Conv1D(stack_ch, 1)(new_x)

            # if first output, we set it to the sliced output tensor
            if output is None:
                output = slice_end(pred_size)(out_x)
            # if second or larger output, we add to the existing output tensor
            else:
                output = Add()([output, slice_end(pred_size)(out_x)])
            
            # if first prediction, no concat; just set x=out(x) for next iteration
            if (n == 0 and i == 0):
                x = out_x

            # if second prediction, use res_net skip connection
            # slice dilation off of front and add to output
            else:
                x = Add()([out_x, slice_front(pow(2, i))(x)])  # skip connections
    
    output = ReLU()(output)
    output = Conv1D(scaled_ch, filter_dim, activation='relu')(output)
    output = Conv1D(in_ch, filter_dim, activation='relu')(output)
    
    model = keras.Model(inputs=inputs, outputs=output, name='WaveNet')

    return model, pred_size

