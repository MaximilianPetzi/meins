import numpy as np
from matplotlib import pyplot as plt 
import matplotlib
matplotlib.use("TkAgg")
import os
os.system("rm ../bilder/plit_temp/*")
nrt=25
am = []
for i in range(nrt):
    print(i+1)
    try:
        t = np.load(str(i+1)+'error.npy')
        am.append(t)
    except:
        print("--"+str(i+1)+"error.npy is missing--")
Am=np.average(am,axis=0)
plt.plot(Am)
plt.savefig("../bilder/plit_temp/this")
plt.show()