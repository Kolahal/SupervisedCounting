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
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
from keras import losses

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
        PMT_id = sys.argv[1]
	
        fileCount = 0
        x = np.zeros(shape=(5000000,702))
        X = []
	
	#for iFile in glob.glob('/pic/projects/miniclean/kb/rat2282/Ntuples/gAr/PEcounting/beta_training/PulseData_PMT_'+str(PMT_id)+'_*'):
        for iFile in sorted(glob.glob('/pic/projects/miniclean/kb/rat2282/Ntuples/gAr/PEcounting/gAr_beta/PulseData_gAr_beta_PMT_'+str(PMT_id)+'_*')):
	        print(iFile)
                #b=os.path.getsize(iFile)
                #print(str(b))
                f=h5py.File(iFile,'r')
                print(str(f['dataList']))
                print('before filling, shape '+str(x.shape)+'     '+str(len(X)))
                x = np.array(f['dataList'])
                #X = x.tolist()
		
                if fileCount == 0:
                        X = x.tolist()
                else:
                        X.extend(x.tolist())
		
                print('after filling, shape '+str(x.shape)+'     '+str(len(X)))
                fileCount = fileCount + 1
                #if fileCount == 1:
                #        break
	
        train_data = np.asarray(X,dtype=np.float16)
        print(str(train_data.shape))
        train_label=train_data[:,701]
        n_train=len(train_label)
        train_label=train_label.reshape((n_train,1))
        train_label=train_label#/100.0
        train_feature=train_data[:,:-1]
	
        print(str(train_data.shape))
        print('train label shape'+str(train_label.shape))
        print('train feature shape '+str(train_feature.shape))
        print('train feature length '+str(len(train_feature)))
        print('train data shape '+str(train_data.shape))
	
        #test_PMT = sys.argv[2]
        #testf = h5py.File('TestHDF5/PulseData_alpha3_PMT_'+str(PMT_id)+'_0.hdf5','r')
        #testf = h5py.File('TestHDF5/PulseData_PMT_1_0.hdf5','r')
        testf = h5py.File('TestHDF5/PulseData_gamma_PMT_'+str(PMT_id)+'_0.hdf5','r')
	test_data=np.array(testf['dataList'])
        test_label=test_data[:,701]
        n_test=len(test_label)
        test_label=test_label.reshape((n_test,1))
        test_label=test_label#/100.0
        test_feature=test_data[:,:-1]
	
        print('test label shape'+str(test_label.shape))
        print('test feature shape '+str(test_feature.shape))
        print('test data shape '+str(test_data.shape))
        print('............................................')
	
	model = Sequential()
	model.add(Dense(1024, input_dim=701, kernel_initializer='glorot_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	#model.add(Dropout(0.50))
	
	model.add(Dense(1024, kernel_initializer='glorot_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	#model.add(Dropout(0.75))
	'''
	model.add(Dense(64, kernel_initializer='glorot_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.75))
	
	model.add(Dense(8, kernel_initializer='glorot_uniform'))
	model.add(BatchNormalization())
        model.add(Activation('relu'))
	'''
	model.add(Dense(1, kernel_initializer='glorot_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	
	#y_pred = model
	
	adam = Adam(lr=0.5, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['accuracy'])#'mean_squared_error' 'poisson'
	history = model.fit(train_feature, train_label, epochs=100, batch_size=65536, validation_data=(test_feature, test_label), shuffle=True, verbose = 2)#65536
	score = model.predict(test_feature)
	#print(str(len(score)))
	
	with open('gAr_gamma_predictions_for_nPE_for_PMT_'+str(PMT_id)+'_bN.txt', 'w+') as outputfile:
		for m in range(len(score)):
			print(str(test_label[m])+'     '+str(score[m]))
			outputfile.write(str(test_label[m])+'     '+str(score[m])+'\n')
	#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
	#model.predict(test_feature, verbose=1)
	print('---x---')
	
	print(history.history.keys())
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.savefig('gAr_gamma_accuracy_vs_epoch_PMT'+str(PMT_id)+'.png')
	plt.clf()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.savefig('gAr_gamma_loss_vs_epoch_PMT'+str(PMT_id)+'.png')
	
	# serialize model to JSON
	model_json = model.to_json()
	with open('keras_model_gAr_gamma'+str(PMT_id)+'.json', "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights('keras_model_gAr_gamma'+str(PMT_id)+'.h5')
	print("Saved model to disk")
