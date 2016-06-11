import numpy as np
import matplotlib.pyplot as plt



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