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
        t = np.load("../error_h/"+str(i+1)+'error.npy')
        am.append(t)
    except:
        print("--"+str(i+1)+"error.npy is missing--")
am=np.array(am)

minl=10000000
for i in range(len(am)):
    minl=np.min([minl,len(am[i])])
Am=np.zeros((len(am),minl))
for i in range(len(am)):
    amm=am[i]
    Am[i,:]=amm[:minl]
print("data-shape before avg:",np.shape(Am))
Am=np.average(Am,axis=0)

plt.plot(Am)
plt.savefig("../bilder/plit_temp/this")
plt.show()

