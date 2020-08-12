duration = 44100
epochs = 50
lr = 1e-3
max_dilation = 10
pred_size = duration - pow(2, max_dilation)
batch_size = 1
base_dir = './sparse_matrices'
