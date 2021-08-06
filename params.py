##
print("params started")
import numpy as np
import os
import time

ara=[.3,.5,1]
arb=[4,8,16]
#set last entry to 1
Paramarr=np.array([.7,ara[0],1.5,200,arb[0],1])
np.save('../meins/paramarr',Paramarr)
os.system("echo 0 > ../meins/cancel.txt")
#change this 0 in the txt file to stop the simulations
for i in range(3):
	for j in range(3):
		print(i,j)
		while True:
			Paramarr=np.load("../meins/paramarr.npy")
			if Paramarr[-1]==1:break
			print("waiting")
			time.sleep(.1)#wait till its 1
		Paramarr=np.array([.7,ara[i],1.5,200,arb[j],0])
#last entry=1 means file is writable, 0 means its not yet read by minconi.py so it has to wait before its changed again.
		np.save('../meins/paramarr',Paramarr)
		os.system("python3 ../meins/meta.py &")
		
		
