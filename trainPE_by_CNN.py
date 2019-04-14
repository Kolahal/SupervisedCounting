import os
import sys
import glob
import h5py
import numpy as np
import time
import tensorflow as tf

import keras
from keras import initializers
from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
from keras import losses
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

if __name__ == '__main__':
        #PMT_id = sys.argv[1]
	
        fileCount = 0
        x = np.zeros(shape=(10000,92,701))
        X = []
	y = np.zeros(shape=(10000))
	Y = []
	#/pic/projects/miniclean/kb/rat2282/Ntuples/gAr/PEcounting/CNN/beta/PulseData_beta_0.hdf5
	#PulseData_beta_0_test.hdf5
        for iFile in glob.glob('/pic/projects/miniclean/kb/rat2282/Ntuples/gAr/PEcounting/CNN/beta/PulseData_beta_0.hdf5'):
                print(iFile)
                f=h5py.File(iFile,'r')
                print(str(f['featureList_CNN']))
		print(str(f['no_MCPEList_CNN']))
                print('before: x shape '+str(x.shape)+'     '+str(len(X)))
		print('before: y shape '+str(y.shape)+'     '+str(len(Y)))
		
                x = np.array(f['featureList_CNN'])
		y = np.array(f['no_MCPEList_CNN'])
		
                if fileCount == 0:
                        X = x.tolist()
			Y = y.tolist()
                else:
                        X.extend(x.tolist())
			Y.extend(y.tolist())
		
                print('after: x shape '+str(x.shape)+'     '+str(len(X)))
		print('after: y shape '+str(y.shape)+'     '+str(len(X)))
		
                fileCount = fileCount + 1
		if fileCount==1:
			break
	
	print('comes outside the loop')
	
        train_feature_data = np.asarray(X,dtype=np.float16)
        print(str(train_feature_data.shape))
        train_label_data   = np.asarray(Y,dtype=np.float16)
	print(str(train_label_data.shape))
        n_train=len(train_label_data)
        train_label_data=train_label_data.reshape((n_train,1))
        print('reshaped to: '+str(train_label_data.shape))
	
	train_feature_data = train_feature_data.reshape(train_feature_data.shape[0],1,92,701)
	
	
	testf  = h5py.File('/people/bhat731/HomeRAT/PhotonCounting/PulseData_beta_0_test.hdf5','r')
	test_x = np.array(testf['featureList_CNN'])
	test_y = np.array(testf['no_MCPEList_CNN'])
	test_X = test_x.tolist()
	test_Y = test_y.tolist()
	test_feature_data = np.asarray(test_X,dtype=np.float16)
	test_label_data	  = np.asarray(test_Y,dtype=np.float16)
	print(str(test_feature_data.shape))
	print(str(test_label_data.shape))
	n_test=len(test_label_data)
	test_label_data   = test_label_data.reshape((n_test,1))
	print('reshaped to: '+str(test_label_data.shape))
	
	test_feature_data = test_feature_data.reshape(test_feature_data.shape[0],1,92,701)
	
	numPE = 1
	print('before layer 1')
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same', input_shape=(1,92,701), data_format='channels_first'))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	print('before layer 2')
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	print('before layer 3')
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(numPE))
	model.add(Activation('relu'))
	print('before optimizer')
	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
	print('before compile')
	# Let's train the model using RMSprop
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
	print('before model fit')
	#train_feature_data1 = train_feature_data.reshape((9360,92,701,1))
	#print(str(train_feature_data1.shape))
	model.fit(train_feature_data, train_label_data, batch_size=2, epochs=10, shuffle=True)
	
	'''
	model = Sequential()
	model.add(Conv1D(32, (3), input_shape=(92, 701), activation='relu'))
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(numPE, activation='relu'))
	print('before compilation')
	model.compile(
			loss=keras.losses.mean_squared_error,
			optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
			metrics=['accuracy']
			)
	batch_size = 128
	epochs = 10
	print('before fitting')
	
	model.fit(
			train_feature_data,
			train_label_data,
			batch_size=batch_size,
			epochs=epochs,
			verbose=1,
			validation_data=(test_feature_data, test_label_data)
		)
	'''
