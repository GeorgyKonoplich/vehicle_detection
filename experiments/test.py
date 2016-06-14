# -*- coding: utf-8 -*-
import codecs
import numpy as np
import matplotlib.pyplot as plt
import sys

print(sys.getdefaultencoding())
'''
path_to_project = "C:/workspace/ml/graduate_work/vehicle_detection/" #windows"

accs = np.load(path_to_project + "data/processed/acc200_128.npy")

his1 = np.load(path_to_project + "data/processed/history200_5.npy")

print(len(his1))
plt.plot(his1)
plt.axis([0, len(his1), -0.5, 2])
plt.show()
ans = 0
for x in accs:
	ans += x
print("accuracy-cv: ", ans/5)
print(accs)
'''
xx = []
x = np.array([14.5, 6.2, 3.5, 2.1, 1.6])
x1 = np.array([87.6, 59.2, 43.0, 32.8, 24.5])
x2 = np.array([91.6, 65.3, 49.1, 40.1, 31.9])
x3 = np.array([18.5, 7.0, 4.1, 2.2, 1.8])
x4 = np.array([23.2, 12.1, 7.3, 4.9, 3.7])
y = np.array([95, 90, 85, 80, 75])
xx.append(x)
xx.append(x1)
xx.append(x2)
xx.append(x3)
xx.append(x4)

arr = ["DNN", "LBP+SVM", "Adaboost", "HDNN", "SimpleDNN"]

class Foo(object):
	def __init__(self, name):
		self.name = name
  
	def __str__(self):
		return 'str: %s' % self.name
  
	def __unicode__(self):
		return 'uni: %s' % self.name.decode('utf-8')

	def __repr__(self):
		return 'repr: %s' % self.name


plt.xlabel("recall rate")
plt.ylabel("false alarm rate")
#plt.title('')
plt.grid(True)
plt.axis([75, 98, 0, 100])
for i in range(0, 5):
    plt.plot(y, xx[i], label=arr[i])
plt.legend(loc='best')
plt.show()