import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt

#import keras
#from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv1D, Dense
from keras.layers import LSTM, TimeDistributed
from keras.layers import Flatten, Reshape, Concatenate
#from keras.layers import RepeatVector
#from keras.callbacks import ModelCheckpoint





class FakeData:

    def __init__(self,seq_len):       
        self.len = seq_len
        self.data = []
        self.meta = []

    def add_data(self, start_range, step_size, noise, num):
        self.meta.append([start_range, step_size, noise, num])       
        for _ in range(num):
	        seq_data = [np.random.uniform(*start_range)]            
	        for _ in range(self.len-1):
	            seq_data.append(seq_data[-1]+step_size+np.random.normal(0,noise))           
	        self.data.append(seq_data)

    def output_data(self):
        return np.array(self.data)

    def check_meta(self):
        for mt in self.meta:
            print("start_range:{}, step_size:{}, noise:{}, number:{}".format(*mt))

    def plot(self, colors):
        assert len(colors) == len(self.meta)
        x = [i for i in range(self.len)]
        done = 0
        for batch, size in enumerate([m[-1] for m in self.meta]):
        	col = colors[batch]
        	for i in range(size):
        		plt.plot(x, self.data[done], c=col)
        		done += 1
        plt.show()

    def sample(self,num):
        picked = []
        num_each_batch = [m[-1] for m in self.meta]
        c = 0
        for n in num_each_batch:
            picked += sorted(list(np.random.choice(range(c,c+n),num,replace=False)))
            c += n
        return picked



def get_model(seq_len,embedding_size):

    #############
    # ENCODER
    #############
    seq_input = Input(shape=(seq_len,1))
    
    encoder_layer = Conv1D(1, 3)(seq_input)
    encoder_layer = Flatten()(encoder_layer)

    encoder_layer_1 = Dense(8, activation='relu')(encoder_layer)
    encoder_layer_1 = Dense(4, activation='relu')(encoder_layer_1)
    encoder_layer_1 = Dense(1, activation='linear')(encoder_layer_1)

    encoder_layer_2 = Dense(4, activation='relu')(encoder_layer)
    encoder_layer_2 = Dense(2, activation='relu')(encoder_layer_2)
    encoder_layer_2 = Dense(1, activation='linear')(encoder_layer_2)

    encoder_layer = Concatenate()([encoder_layer_1,encoder_layer_2])

    encoder = Model(seq_input, encoder_layer, name='encoder')

    #############
    # DECODER
    #############

    decoder_input = Input(shape=(embedding_size,))

    decoder_layer = Dense(seq_len*2)(decoder_input)
    decoder_layer = Reshape((seq_len,2))(decoder_layer)
    decoder_layer = LSTM(2, activation='relu', return_sequences=True)(decoder_layer)
    decoder_layer = TimeDistributed(Dense(1))(decoder_layer)

    decoder = Model(decoder_input, decoder_layer, name='decoder')

    #############
    # AutoEncoder
    #############

    final_output = decoder(encoder(seq_input))
    model        = Model(seq_input, final_output, name='seq_embedding')

    return model




