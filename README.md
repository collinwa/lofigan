# lofi wavennet

A tensorflow implementation of DeepMind's WaveNet architecture for generating lo-fi music, or any music, really. `webscraper.py` will scrape YouTube and download .mp4 audio files from a given Playlist link. 

You'll need to encode these files as sparse matrices before training. Use the `encode_data.py` script to convert the files into T X N size matrices, where T is the number of timesteps at 44khz sampling rate and N=256 is the number of channels after running a mu companding transform, quantizing, and one-hot encoding. 

`train.py` will construct the net. Hyperparameters here include duration, the maximum dilation of the network, the number of one-hot channels (though this should probably stay at 256), etc.

`WaveNet.py` contains the actual model. 

