#!/usr/bin/env python
from __future__ import print_function
import sys, os, glob

#file_list = glob.glob("output_file_over_full_training_set_batch_size32_epoch20_*_bN1.txt")
file_list = glob.glob("lAr_beta_predictions_for_nPE_for_PMT_*_bN.txt")
file_list.sort(key=lambda f: int(filter(str.isdigit, f)))
n = 0
labels = []
feature= []
for i in file_list:
	#print(i)
	#n = n+1
	
	labels.append([1.0*float(x.split('     ')[0]) for x in open(i).readlines()])
	feature.append([1.0*float(x.split('     ')[1]) for x in open(i).readlines()])
	
label_totals = [sum(x) for x in zip(*labels)]
feature_totals = [sum(x) for x in zip(*feature)]

res = "\n".join("{}     {}".format(x, y) for x, y in zip(label_totals, feature_totals))
print(str(res))
