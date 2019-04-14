#!/usr/bin/env python
import sys
import ROOT as root
import rat
import numpy as np
import tensorflow as tf
import time
import math
import h5py
from multiprocessing import Process, Queue

# function for training a network
def run_training(training_dataset):
    #print('Accessing the training function:')
    features = []
    labels   = []
    
    features = training_dataset[:,:-1]
    labels   = training_dataset[:,-1]
    print('---> '+str(len(training_dataset[:,-1])))
    labels   = labels.reshape((len(training_dataset[:,-1]),1))
    labels   =labels/1000
    
    print('feature shape '+str(features.shape))
    print('label shape '+str(labels.shape))
    
    print(features)
    print(labels)
    ############################
    
    # Parameters
    learning_rate   =0.0023
    training_epochs =   100
    batch_size      =    29
    display_step    =     1
    
    # Network Parameters
    n_hidden_1 = 1000 # 1st layer number of neurons
    n_hidden_2 =  100 # 2nd layer number of neurons
    n_hidden_3 =   10
    n_input    =  701 # feature vector size
    n_classes  =    1 # MC photon count
    
    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])
    
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
    }
    
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # Create model
    def multilayer_perceptron(x):
        # Hidden fully connected layer with 10000 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        
        # Hidden fully connected layer with 1000 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)
        
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.sigmoid(layer_3)
        
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
        out_layer = tf.nn.sigmoid(out_layer)
        
        return out_layer
    
    # Construct model
    logits = multilayer_perceptron(X)
    print('logits shape '+str(logits.shape))
    
    # Define loss and optimizer
    #loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    loss_op  = tf.losses.mean_squared_error(logits, Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(features)/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x = features[i*batch_size:(i+1)*batch_size]
                batch_y = labels[i*batch_size:(i+1)*batch_size]
                
                #batch_x, batch_y = mnist.train.next_batch(batch_size)
                #print('batch_x size '+str(batch_x.shape))
                #print('batch_y size '+str(batch_y.shape))
                
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
                #print('prediction '+str(sess.run([logits], feed_dict={X: batch_x, Y:batch_y})))
                pred = sess.run([logits],feed_dict={X: batch_x, Y: batch_y})
                #print(str(batch_x))
                #print(str(i)+'     '+str(pred)+'     '+str(batch_y)+'     '+str(c))
                #print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")
        
        # Test model
        pred = logits  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        #print("Prediction:", pred.eval({X: test_features, Y: test_labels}))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #print("Accuracy:", accuracy.eval({X: test_features, Y: test_labels}))
        
#function to read data
def read_data_into_list(infile):
    print('Accessing read function from main function')
    filelist = infile
    feature_list = []
    pmt_features = []
    event_list   = []
    n = 0
    with open(filelist) as f:
        lines = f.readlines()
        for l in lines:
            print(l)
            del feature_list[:]
            f=root.TFile.Open(l.strip())
            tree = f.Get("T")
            no_event_processed_in_this_file = 0
            match= 0
            #rds = rat.RAT.DS.Root()
            rds_reader = rat.RAT.DSReader(l.strip())
            rds = rds_reader.NextEvent()
            while(rds_reader.NextEvent()):
                n_ev = rds.GetEVCount()
                if n_ev==0 or n_ev>1:
                    continue
                if n_ev==1:
                    mc = rds.GetMC()
                    #n_MCPE = mc.GetNumPE()
                    
                    for i in range(n_ev):
                        ev = rds.GetEV(i)
                        ev_id=ev.GetEventID()
                        #print('event ID '+str(ev_id))
                        #fp = ev.GetFprompt()
                        #print(str(ev_id)+'      '+str(fp)+'     '+str(n_MCPE))
                        del pmt_features[:]
                        for j in range(ev.GetPMTCount()):
                            #print('j '+str(j))
                            this_pmt_is_bad=False
                            del feature_list[:]
                            j_pmt = ev.GetPMT(j)
                            mc_j_pmt = 0
                            feature_list.append(j_pmt.GetPulseCount())
                            
                            for k in range(100):
                                n = n+1
                                #print(str(k)+'     '+str(j_pmt.GetPulseCount()))
                                if k<j_pmt.GetPulseCount():
                                    #print('filling data ...')
                                    k_pulse = j_pmt.GetPulse(k)
                                    if abs(k_pulse.GetTime())>1.e5 or abs(k_pulse.GetCharge())>1.e5 or abs(k_pulse.GetPeak())>1.e10 or abs(k_pulse.GetPeakCharge())>1.e5 or abs(k_pulse.GetLeftEdge())>1.e5 or abs(k_pulse.GetRightEdge())>1.e5 or abs(k_pulse.GetWidth())>1.e5:
                                        #print(str(k_pulse.GetTime())+'     '+str(k_pulse.GetCharge())+'     '+str(k_pulse.GetPeak())+'     '+str(k_pulse.GetPeakCharge())+'     '+str(k_pulse.GetLeftEdge())+'     '+str(k_pulse.GetRightEdge())+'      '+str(k_pulse.GetWidth()))
                                        feature_list.append(0)
                                        feature_list.append(0)
                                        feature_list.append(0)
                                        feature_list.append(0)
                                        feature_list.append(0)
                                        feature_list.append(0)
                                        feature_list.append(0)
                                        this_pmt_is_bad=True
                                    else:
                                        feature_list.append(k_pulse.GetTime())
                                        feature_list.append(k_pulse.GetCharge())
                                        feature_list.append(k_pulse.GetPeak())
                                        feature_list.append(k_pulse.GetPeakCharge())
                                        feature_list.append(k_pulse.GetLeftEdge())
                                        feature_list.append(k_pulse.GetRightEdge())
                                        feature_list.append(k_pulse.GetWidth())
                                else:
                                    #print('filling zero ...')
                                    feature_list.append(0)
                                    feature_list.append(0)
                                    feature_list.append(0)
                                    feature_list.append(0)
                                    feature_list.append(0)
                                    feature_list.append(0)
                                    feature_list.append(0)
                            
                            rec_pmtID= j_pmt.GetID()
                            mc_pmtID = 0
                            mc_pmt_npe=0
                            for p in range(mc.GetPMTCount()):
                                if this_pmt_is_bad==True:
                                    mc_pmt_npe = 0
                                    break
                                else:
                                    mc_j_pmt = mc.GetPMT(p)
                                    mc_pmtID = mc_j_pmt.id
                                    if rec_pmtID == mc_pmtID:
                                        match = match + 1
                                        mc_pmt_npe=mc_j_pmt.GetMCPhotonCount()
                                        #print('----> '+str(rec_pmtID)+'     '+str(mc_pmtID)+'     '+str(mc_pmt_npe))
                            #print('# of MC PE in this PMT is '+str(mc_pmt_npe))
                            feature_list.append(mc_pmt_npe)
                            if this_pmt_is_bad==True:
                                #print('0-th element of feature-list is '+str(feature_list[0])+', but resetting it to zero...')
                                feature_list[0]=0
                            #print(str(math.isnan(feature_list[:])))
                            #print(feature_list)
                            #print('length of feature list '+str(len(feature_list)))
                            # loop on # of PMTs ends here
                            pmt_features.append(feature_list[:])
                            #print('length of pmt_features '+str(len(pmt_features)))
                            #print(pmt_features)
                            #print('............................................')
                        #npa = np.asarray(pmt_features, dtype=np.float32)
                        #print('type of np array '+str(type(npa))+'     shape '+str(npa.shape))
                        # loop on # of event ends here
                        ##############################
                    # scope of outer if else clause:
                    # within n_ev = 1
                # information of this event ends here...
                # Next event will be processed now... 
                
                event_list.append(pmt_features[:])
                np_ev_pmt_features_array = np.asarray(event_list,dtype=np.float32)
                #print('length of event list '+str(len(event_list))+', length of PMT list '+str(len(pmt_features))+', length of feature_list '+str(len(feature_list)))
                #print('shape of np_ev_pmt_features_array '+str(np_ev_pmt_features_array.shape))
		no_event_processed_in_this_file = no_event_processed_in_this_file + 1
                #print('---> '+str(no_event_processed_in_this_file))
                if no_event_processed_in_this_file==1:
                    break
            print('.................................')
    with h5py.File('test_'+str(index)+'.h5', 'w') as hf:
        hf.create_dataset('dataList',data=event_list)
    return event_list

if __name__ == '__main__':
    print('comes inside main function ')
    filelist = sys.argv[1]
    
    tic = time.time()
    dataset = read_data_into_list(filelist,index)
    #print(dataset)
    np_ev_pmt_features_array = np.asarray(dataset,dtype=np.float16)
    #print('shape of returned list '+str(np_ev_pmt_features_array.shape))
    #print('shape of a given 2D (event, feature) slice of 3D dataset '+str(np_ev_pmt_features_array[:,0,:].shape))
    #print(str(len(np_ev_pmt_features_array[0])))
    #_events=np.array([np.array(xi) for xi in dataset])
    #print('shape of numpy arrays '+str(_events.shape))
    toc = time.time()
    print(str(toc-tic))
    #for k in range(len(np_ev_pmt_features_array[0])):
    #    #print('shape of a given 2D (event, feature) slice of 3D dataset for '+str(k)+'-th PMT is '+str(np_ev_pmt_features_array[:,k,:].shape))
    #    #print(str(np_ev_pmt_features_array[:,k,:]))
    #    #run_training(np_ev_pmt_features_array[:,k,:])
    #    #if k==0:
    #        break
