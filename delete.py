import numpy as np
def fdbencode(coord,sig,Nc,rangea,rangeb):
    arenc=np.zeros(Nc)
    for i in range(Nc):
        ii=i/(Nc-1)*(rangeb-rangea)+rangea
        arenc[i]=np.exp(-(ii-coord)**2/(2*sig**2))
    return arenc

print(fdbencode(-40,4,21,-95.5, 8))