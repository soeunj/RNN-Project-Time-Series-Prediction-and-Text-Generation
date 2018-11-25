import numpy as np
import string

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

def build_part1_RNN(window_size):
    # build an RNN to perform regression on our time series input/output data
    model = Sequential()
    model.add(LSTM(5, input_shape = (window_size, 1)))
    model.add(Dense(1))
    
    return model

def cleaned_text(text):
    # return the text input with only ascii lowercase and the punctuation given below included.
    punctuation = ['!', ',', '.', ':', ';', '?']
    alp = list(string.ascii_lowercase)
    chars = set(punctuation + alp)
    
    text_set = set(text)
    chars_to_remove = text_set - chars
    print(chars_to_remove)
    for c in chars_to_remove:
        text = text.replace(c, ' ')
    return text
  
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i : i + window_size]) 
        outputs.append(text[i + window_size])
        
    return inputs,outputs

def build_part2_RNN(window_size, num_chars):
    # build the required RNN model : a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
    model = Sequential()
    model.add(LSTM(200, input_shape = (window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    
    return model
