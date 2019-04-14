#!/usr/bin/env python
import sys
import ROOT as root
from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F
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
	n = 0
	
	print(infile)
	
	#print(str(int(filter(str.isdigit, infile))))
	#print(str(infile[]))
	start = 'alpha_decay_type'
	end   = '/'
	s     = infile
	alpha_type = s[s.find(start)+len(start):s.rfind(end)]
	print('=> '+alpha_type)
	
	del feature_list[:]
	f=root.TFile.Open(infile)
	tree = f.Get("T")
	no_event_processed_in_this_file = 0
	match= 0
	
	RDS = rat.RAT.DS.Root()
	#numEntries = tree.GetEntries()
	#print(type(numEntries))
	tree.SetBranchAddress("ds",RDS)
	'''
	h_nPulse= TH1F('no. of pulse', '', 		100, 0, 100)
	h_time	= TH1F('time histogram', '', 		100, -1000.0,10000)
	h_charge= TH1F('charge histogram', '', 		100, -100.0, 300)
	h_peak	= TH1F('peak histogram', '', 		100, -100.0, 5000)
	h_peakC	= TH1F('peak charge  histogram', '', 	100, -20.0,  200)
	h_lE	= TH1F('left edge histogram', '', 	100, -20.0,  5000)
	h_rE	= TH1F('right edge histogram', '', 	100, -20.0,  5000)
	h_width = TH1F('width histogram', '', 		100, -20.0,  1000)
	'''
	for ix in range(tree.GetEntries()):
		tree.GetEntry(ix)
		No_ev = RDS.GetEVCount()
		#print(str(No_ev))
		
		if No_ev==0 or No_ev>1:
			continue
		if No_ev==1:
			mc = RDS.GetMC()
			
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
					#h_nPulse.Fill(j_pmt.GetPulseCount())
					for k in range(100):
						n = n+1
						#print(str(k)+'     '+str(j_pmt.GetPulseCount()))
						if k<j_pmt.GetPulseCount():
							k_pulse = j_pmt.GetPulse(k)
							if abs(k_pulse.GetTime())>1.e5 or abs(k_pulse.GetCharge())>1.e5 or abs(k_pulse.GetPeak())>1.e10 or abs(k_pulse.GetPeakCharge())>1.e5 or abs(k_pulse.GetLeftEdge())>1.e5 or abs(k_pulse.GetRightEdge())>1.e5 or abs(k_pulse.GetWidth())>1.e5:
								#print(str(k_pulse.GetTime())+'     '+str(k_pulse.GetCharge())+'     '+str(k_pulse.GetPeak())+'     '+str(k_pulse.GetPeakCharge())+'     '+str(k_pulse.GetLeftEdge())+'     '+str(k_pulse.GetRightEdge())+'      '+str(k_pulse.GetWidth()))
								this_pmt_is_bad = True
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
								'''
								h_time.Fill(k_pulse.GetTime())
								h_charge.Fill(k_pulse.GetCharge())
								h_peak.Fill(k_pulse.GetPeak())
								h_peakC.Fill(k_pulse.GetPeakCharge())
								h_lE.Fill(k_pulse.GetLeftEdge())
								h_rE.Fill(k_pulse.GetRightEdge())
								h_width.Fill(k_pulse.GetWidth())
								'''
						else:
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
					
					feature_list.append(mc_pmt_npe)
					
					if this_pmt_is_bad==True:
						for q in range(len(feature_list)):
							feature_list[q]=0
					
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
		#if no_event_processed_in_this_file==10:
		#	break
	print('.................................')
	
	np_ev_pmt_features_array = np.array(event_list,dtype=np.float32)
	print(str(type(np_ev_pmt_features_array)))
	new_np_ev_pmt_features_array = np.swapaxes(np_ev_pmt_features_array,0,1)
	print('shape of new_np_ev_pmt_features_array '+str(new_np_ev_pmt_features_array.shape))
	pmt_specific_event_list = new_np_ev_pmt_features_array.tolist()
	index = int(index)
	
	for p in range(92):
		with h5py.File('/people/bhat731/HomeRAT/PhotonCounting/TestHDF5/PulseData_lAr_neutron_PMT_'+str(p)+'_'+str(index)+'.hdf5', 'w') as hf:
		#with h5py.File('/pic/projects/miniclean/kb/rat2282/Ntuples/gAr/PEcounting/gAr_beta/PulseData_gAr_beta_PMT_'+str(p)+'_'+str(index)+'.hdf5', 'w') as hf:
			hf.create_dataset('dataList',data=pmt_specific_event_list[p])
		#shutil.move('/people/bhat731/HomeRAT/PhotonCounting/PulseData_alpha_PMT_'+str(p)+'_'+str(index)+'.hdf5','/pic/projects/miniclean/kb/rat2282/Ntuples/gAr/PEcounting/Alpha_training/PulseData_alpha_PMT_'+str(p)+'_'+str(index)+'.hdf5')
	
	'''
	rootFile = 'Pulse_features_alpha11.root'
	tempFile = root.TFile(rootFile,"RECREATE")
	tempFile.WriteTObject(h_nPulse)
	tempFile.WriteTObject(h_time)
	tempFile.WriteTObject(h_charge)
	tempFile.WriteTObject(h_peak)
	tempFile.WriteTObject(h_peakC)
	tempFile.WriteTObject(h_lE)
	tempFile.WriteTObject(h_rE)
	tempFile.WriteTObject(h_width)
	tempFile.Close()
	'''

if __name__ == '__main__':
	infile= sys.argv[1]
	index = sys.argv[2]
	
	tic = time.time()
	read_data_into_h5(infile,index)
	toc = time.time()
	print(str(toc-tic))
