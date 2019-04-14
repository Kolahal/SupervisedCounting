#!/usr/bin/python2.7
import os
import sys
import glob
import h5py
import numpy as np
import time
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import load_model
from keras import initializers
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
from keras import losses
from keras.callbacks import ModelCheckpoint
from keras import backend as K
#K.clear_session()
#from slurmpy import Slurm
print('Hello RAT')

#arr = np.array(list_of_args)
#print('---> '+str(arr.shape))
K.clear_session()
#gpu_count = len([dev for dev in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if len(dev.strip()) > 0])
#print('---> '+str(gpu_count))

model = Sequential()
model.add(Dense(1024, input_dim=701, kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1024, kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1, kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))

adam = Adam(lr=0.5, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

x = []

f = h5py.File('TestHDF5/PulseData_PMT_11_0.hdf5','r')
a = np.array(f['dataList'])

print('length of array '+str(a.shape[0]))
print('shape of array '+str(a.shape))

for ev in range(a.shape[0]):
	feature_vec_for_this_pmt = a[ev]
	fv = feature_vec_for_this_pmt.tolist()
	fv = fv[:-1]
	#print('length of modified fv '+str(len(fv)))
	feature_vec_for_this_pmt = np.array(fv)
	#print('shape of feature_vec_for_this_pmt '+str(feature_vec_for_this_pmt.shape))
	u = feature_vec_for_this_pmt.reshape((701,1))
	v = u.reshape(1,701)
	#print('shape of transposed vector '+str(v.shape))
	#print('load weight file')
	model.load_weights('keras_model_beta0.h5')
	score = model.predict(v)
	x.append(score)
	print(str(score))
#print(str(x))
'''
	feature_vec_for_this_pmt = arr[ipmt]
	u = feature_vec_for_this_pmt.reshape((701,1))
	v = u.reshape(1,701)
	model.load_weights('keras_model_beta0.h5')
	score = model.predict(v)
	x.append(score)
'''
'''
	print('now, into the loop..')
	for ipmt in range(a.shape[0]):
		feature_vec_for_this_pmt = arr[ipmt]
		#print('shape of feature_vec_for_this_pmt '+str(feature_vec_for_this_pmt.shape))
		u = feature_vec_for_this_pmt.reshape((701,1))
		#print('shape of feature_vec_for_this_pmt after reshaping '+str(u.shape))
		v = u.reshape(1,701)
		#print('shape of transposed vector '+str(v.shape))
		#K.clear_session()
		#print('load weight file: /people/bhat731/Package/RATDEV/rat/src/calib/keras_model'+str(ipmt)+'.h5:')
		model.load_weights('/people/bhat731/Package/RATDEV/rat/src/calib/keras_model_alpha'+str(ipmt)+'.h5')
		#adam = Adam(lr=0.5, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
		#model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
		score = model.predict(v)
		
		#print(str(score))
		x.append(score)
		#K.clear_session()
	
	#print('type of returned x '+str(type(x)))
'''
