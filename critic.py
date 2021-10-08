import numpy as np
import copy
def constantF(x):
    return np.array([1])
def linF(x):#length 5
    x=np.array(x)
    x=np.append([1],x)
    return x
def squareF(x):
    lengt=len(x)
    lx=linF(x)
    features=np.copy(lx)
    for i in range(lengt):
        for j in range(i,lengt):
            features=np.append(features,lx[i]*lx[j])
    return np.array(features)
def scaledgaus(x,idx,sig,Nc,rangea,rangeb):
    realsig=sig*0.5*(rangeb-rangea)/(Nc-1)
    ii=idx/(Nc-1)*(rangeb-rangea)+rangea
    return (np.exp(-(ii-x)**2/(2*realsig**2)),ii)   #return value of gaussian at x, and origin of gaussian (ii)

def gaussianF(x):
    features=[1]#baseline->init weight as -1
    origins=[]
    n_n=3
    n_i=n_n
    n_j=n_n
    n_k=n_n
    n_l=n_n
    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):
                for l in range(n_l):
                    (v1,o1)=scaledgaus(x[0],i,2,n_i,0,1)#0,160)
                    (v2,o2)=scaledgaus(x[1],j,2,n_j,0,1)#-95.5, 8)
                    (v3,o3)=scaledgaus(x[2],k,2,n_k,0,1)#-32, 80)
                    (v4,o4)=scaledgaus(x[3],l,2,n_l,0,1)#15, 106)
                    features.append(v1*v2*v3*v4)
                    origins.append([o1,o2,o3,o4])
    return np.array(features)#,np.array(origins)

def kernel(x,kernelstr=1):
    if kernelstr=="constantF":
        return constantF(x)
    if kernelstr=="linF":
        return linF(x)
    if kernelstr=="squareF":
        return squareF(x)
    if kernelstr=="gaussianF":
        return gaussianF(x)
    print(kernelstr,": kernel doesnt exist")
    return 666
    



class model:
    
    def __init__(self,kernelstr):
        self.kernelstr=kernelstr
        testfeat=kernel([0,0,0,0],self.kernelstr)
        self.w=np.zeros(len(testfeat))#polynome 0deg, 1deg(x,y,z,w)
        self.w[0]=0   #baseline position
        self.me=[0]
        self.weights=np.zeros((0,len(self.w)))
        self.meAlpha=0

    def predict(self,x):
        features=kernel(x,self.kernelstr)
        rhat=self.w@features
        return rhat
    
    def learnstep(self,x,r,eta):
        features=kernel(x,self.kernelstr)
        rhat=self.predict(x)
        self.w=self.w-eta*(rhat-r)*features #delta learing rule
        lastme=np.array(self.me)[-1]
        self.me.append(lastme*self.meAlpha+abs((r-rhat))**2*(1-self.meAlpha))
        self.weights=np.concatenate((self.weights,np.array([self.w])),axis=0)

#mycrit=model("gaussianF")
#print(mycrit.predict([1,2,3,4]))
#for i in range(5):
#    mycrit.learnstep(x=[1,2,3,4],r=-1.1,eta=.02)
#    print(mycrit.predict([1,2,3,4]))
#    print(mycrit.w)
