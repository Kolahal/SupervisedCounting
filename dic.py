import re
from itertools import izip
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
from pylab import genfromtxt;

f0 = open("pmt0.txt", 'r')
answer = {}
VA0 = []
for line in f0:
	if 'val_loss' not in line:
		continue
	L = line.split('-')
	VA0.append(float(re.findall("\d+\.\d+",L[5])[0]))
print(str(VA0))

print('.........')
f1 = open("pmt11.txt", 'r')
VA1 = []
for line in f1:
	if 'val_loss' not in line:
		continue
	L = line.split('-')
	#print(L[5].strip())
	VA1.append(float(re.findall("\d+\.\d+",L[5])[0]))
print(str(VA1))
'''
plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot(VA0)
plt.subplot(212)             # the second subplot in the first figure
plt.plot(VA1)
'''

plt.plot(VA0,c='r')
plt.plot(VA1,c='b')

#plt.title('Comparison of test accuracy in various PMTs')
plt.ylabel('test accuracy')
plt.xlabel('epoch')
plt.legend(['PMT 0', 'PMT 11'], loc='lower right')

plt.show()
plt.savefig('va_comp.png')
