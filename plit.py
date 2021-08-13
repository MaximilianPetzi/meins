import numpy as np
from matplotlib import pyplot as plt 
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
plt.show()