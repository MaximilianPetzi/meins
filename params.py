#just use one params.py at a time
#give max_trials in meta.py, otherwise its hard to stop it:

print("params started")
import numpy as np
import os
import time

os.system("rm ../bilder/temporary/*")
#ara=[.3,.5,1]
#arb=[4,8,16]
ara=[.000500,0.00501,.000502,.000503]
arb=[1.2,1.4]
#set last entry to 1 
Paramarr=np.array([.7111,.4111,1.5111,200111,4111,1])
np.save('../paramarr',Paramarr)
os.system("echo 0 > ../cancel.txt")
#change this 0 in the txt file to stop the simulations
for i in range(len(ara)):
	for j in range(len(arb)):
		print(i,j)
		while True:
			Paramarr=np.load("../paramarr.npy")
			if Paramarr[-1]==1:break
			print("waiting")
			time.sleep(.1)#wait till its 1
		Paramarr=np.array([0,ara[i],arb[j],200,0,0])
#last entry=1 means file is writable, 0 means its not yet read by minconi.py so it has to wait before its changed again.
		np.save('../paramarr',Paramarr)
		os.system("python3 ../meins/meta.py --m 1&")
os.system("done")
		
