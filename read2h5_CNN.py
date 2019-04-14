#!/usr/bin/env python
import sys
import ROOT as root
import rat
import numpy as np
#import tensorflow as tf
import time
import math
import h5py
import shutil

def read_data_into_h5(infile,index):
	print('Accessing read function from main function')
	
	feature_list = []
	pmt_features = []
	event_list   = []
	numPE_list   = []
	n = 0
	
	print(infile)
	del feature_list[:]
	f=root.TFile.Open(infile)
	tree = f.Get("T")
	no_event_processed_in_this_file = 0
	match= 0
	numPE= 0
	RDS = rat.RAT.DS.Root()
	#numEntries = tree.GetEntries()
	#print(type(numEntries))
	tree.SetBranchAddress("ds",RDS)
	
	for ix in range(tree.GetEntries()):
		tree.GetEntry(ix)
		No_ev = RDS.GetEVCount()
		#print(str(No_ev))
		
		if No_ev==0 or No_ev>1:
			continue
		if No_ev==1:
			mc = RDS.GetMC()
			numPE = mc.GetNumPE()
			numPE_list.append(numPE)
			for i in range(No_ev):
				ev = RDS.GetEV(i)
				#print(str(ev.GetQPE()))
				#ev = rds.GetEV(i)
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
							else:
								feature_list.append(k_pulse.GetTime())
								feature_list.append(k_pulse.GetCharge())
								feature_list.append(k_pulse.GetPeak())
								feature_list.append(k_pulse.GetPeakCharge())
								feature_list.append(k_pulse.GetLeftEdge())
								feature_list.append(k_pulse.GetRightEdge())
								feature_list.append(k_pulse.GetWidth())
						else:
							feature_list.append(0)
							feature_list.append(0)
							feature_list.append(0)
							feature_list.append(0)
							feature_list.append(0)
							feature_list.append(0)
							feature_list.append(0)
					'''
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
					feature_list.append(mc_pmt_npe)
					'''
					if this_pmt_is_bad==True:
						feature_list[0]=0
					#print('PMT ID '+str(rec_pmtID)+'\n')
					#print(feature_list)
					#print('........................................................................................')
					# loop on # of PMTs ends here
					pmt_features.append(feature_list[:])
				#print('--->'+str(ev.GetQPE()))
				# loop on # of event ends here
				##############################
			# scope of outer if else clause:
			# within n_ev = 1
		#print(str(match))
		# information of this event ends here...
		# Next event will be processed now... 
		event_list.append(pmt_features[:])
		#np_ev_pmt_features_array = np.asarray(event_list,dtype=np.float16)
		#np.swapaxes(np_ev_pmt_features_array,0,1)
		
		#print('length of event list '+str(len(event_list))+', length of PMT list '+str(len(pmt_features))+', length of feature_list '+str(len(feature_list)))
		#print('shape of np_ev_pmt_features_array '+str(np_ev_pmt_features_array.shape))
		no_event_processed_in_this_file = no_event_processed_in_this_file + 1
		print('---> '+str(no_event_processed_in_this_file))
		if no_event_processed_in_this_file==50:
			break
	print('.................................')
	
	np_ev_pmt_features_array = np.array(event_list,dtype=np.float32)
	print(str(type(np_ev_pmt_features_array)))
	print('shape of np_ev_pmt_features_array '+str(np_ev_pmt_features_array.shape))
	event_feature_list = np_ev_pmt_features_array.tolist()
	index = int(index)
	
	with h5py.File('PulseData_beta_'+str(index)+'_test.hdf5', 'w') as hf:
		hf.create_dataset('featureList_CNN',data=event_feature_list)
		hf.create_dataset('no_MCPEList_CNN',data=numPE_list)

if __name__ == '__main__':
	infile= sys.argv[1]
	index = sys.argv[2]
	
	tic = time.time()
	read_data_into_h5(infile,index)
	toc = time.time()
	print(str(toc-tic))
