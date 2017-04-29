#!/usr/bin/env python3

from keras.models import Sequential, Model
from keras.layers import LSTM, Input, concatenate, Reshape, RepeatVector
# from utils import *

nsent = 25
ignorelen = 15
training_set_size = 1400
nep = 3
output_size = 50
shared_unit = LSTM(output_size)

inputs = []

for i in range(nsent):
    inputs.append(Input(shape=(None,output_size)))

encs = list(map(shared_unit, inputs))
reshaped_units = list(map(Reshape(target_shape=(1,output_size)), encs))

merged = concatenate(reshaped_units, axis=1)
final_layer = LSTM(output_size)(merged)
repeat_output = RepeatVector(10)(final_layer)

# reshape_encoder = Reshape(target_shape=(1,20))(final_layer)
decoder = LSTM(output_size,return_sequences=True)(repeat_output)
model = Model(inputs=inputs, outputs=decoder)

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[])
print("Model Compiled")
