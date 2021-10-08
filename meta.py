import numpy as np
import os
#just use one meta.py at a time, except with params.py
#terminal auf CPG_iCub
usekm=True      #use kinematic model or not (simulator, not on ssh)
skipcpg=False   #just use scaled minconi output after last timechunk as final angels instead of using the CPG. 
multiple_rewards=True
use_feedback=True
randinit=True
popcode=False
figname2=str(np.random.randint(100000))
savepa="../bilder/temporary/"+figname2+"/"
os.system("mkdir "+savepa)
#picture usually saved in bilder/temporary/, and this folder is always cleared before
showplot=False
doplot2=False
nchunks=1
max_trials=3000
chunktime1=200   #also change var_f inversely
chunktime2=200
d_execution=200   #average over last d_execution timesteps
learnstart=100
import critic
mycrit=critic.model("gaussianF")
crit2=critic.model("constantF")
crit3=critic.model("linF")
crit4=critic.model("squareF")
import time
from termcolor import colored
import sys
import matplotlib.pyplot as plt
import matplotlib
#3D stuff:
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
#matplotlib.use("TkAgg")
sys.path.append("../CPG_iCub")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--m", help="set to eg. 1, if meta is called by params")    #optional argument
parser.add_argument("--s", help="number of the simulation")
args = parser.parse_args()
sim=args.s

if not args.m:
    #os.system("rm ../bilder/temporary/*")
    Paramarr=np.zeros((6))
    default="y"#input("use default?(y/n)")
    if default=="y":
        var_f=9
        var_eta=0.6
        var_g=1.5
        var_N=200
        var_A=20
    else:
        var_f=float(input("f="))
        var_eta=float(input("eta="))
        var_g=float(input("g="))
        var_N=int(input("N="))
        var_A=float(input("A="))
    Paramarr[0]=var_f
    Paramarr[1]=var_eta
    Paramarr[2]=var_g
    Paramarr[3]=var_N
    Paramarr[4]=var_A
    np.save('../paramarr',Paramarr)


if usekm==False:
    import readr
if usekm==True:
    import kinematic_model
import minconi

#start simulator:
path="../iCub_simulator_tools/iCubSim_environment/"
if usekm==False:
    os.system("bash "+path+"start_simulator.sh")
    time.sleep(5)
if usekm==True:
    pass
'''
1 = LShoulderRoll, 26
2 = LElbow, 28
3 = LShoulderPitch, 25
4 = LShoulderYaw, 27
'''

njoints_max=4
mynet=minconi.net(n_out=6*njoints_max) #mynet has many class variables

# mynet.init_w = minconi.net.Wrec.w #instance variable implicitly created

if usekm==False:
    myreadr=readr.readr()
    #print(colored("icub started "+str(myreadr.read()),"yellow"))

# Compute the mean reward per trial
R_mean = np.zeros((2))
r_mean = np.zeros((2))
alpha = 0.6 # 0.33
if multiple_rewards:
    alpha=alpha**(1/nchunks) #to slow down R_mean again to the previous value
import CPG
#print("__________________________qui: CPG imported")

def scaling(x):
    return x+1+0.001#(-.5-1/(x-1))*3
def scaling2(x):
    return 5*(x+1+0.001)
def scalingICUR(x):
    return 8*x
def fdbencode(coord,sig,Nc,rangea,rangeb):
    arenc=np.zeros(Nc)
    for i in range(Nc):
        ii=i/(Nc-1)*(rangeb-rangea)+rangea
        arenc[i]=np.exp(-(ii-coord)**2/(2*sig**2))-np.exp(-(ii-coord)**2/((2*sig)**2))
    return arenc
####################################################

weightlist=np.zeros((0,38))
def trial_simulation(trial,first,R_mean,trialtype,initseed):#trialtype "normal" or sample
    print(trialtype)
    traces = []
    #start cpg:
    mycpg=CPG.cpg()

    #move to starting position:
    if usekm==False:
        mycpg.move_to_init(randinit=randinit)#noch machen (wie bei move_to_init2)
    if usekm==True:
        mycpg.move_to_init2(randinit=randinit,initseed=initseed)
    #time.sleep(2.5)  # wait until robot finished motion
    
    #rbpy, mycpg.Angles are radians
    def give_position(a=False,b=False,c=False,d=False):
        if a==False:
            a=mycpg.Angles[mycpg.LShoulderRoll]
            b=mycpg.Angles[mycpg.LElbow]
            c=mycpg.Angles[mycpg.LShoulderPitch]
            d=mycpg.Angles[mycpg.LShoulderYaw]
        if usekm==False:
            position=myreadr.read()
        if usekm==True:
            if skipcpg:
                the1=(parr[0,0]+1)*80
            if not skipcpg:
                the1=a
            the2=b
            if skipcpg:
                the3=(parr[1,0]+1)/2*(8+95.5)-95.5
            if not skipcpg:
                the3=c
            the4=d
            position=kinematic_model.wrist_position([the4,the3,the1,the2])[0:3]#yprb
        return position
    
    
    initposi=give_position()# needed only for plotting
    trajectory=np.array([initposi])#array of 3D coordinates for total trial movement

    #print(CPG.cpg.myCont[0].RG.E.q)
    #print(mycpg.myCont[0].RG.E.V)
    
    
    #print(mynet.Wi.w[0][0:10])
    recz=[]
    Ahist=[]
    Phist=[]
    #compute target
    if usekm==False:
        target = targetA if first == 0 else targetB
    if usekm==True:
        target= targetA if first == 0 else targetB
    
    errors=[]
    learnerrors=[]
    
    ############################################# minitrials:
    
    for timechunk in range(nchunks):
        #reset network activities:
        mynet.reinit()
        #reset cpg diff equation (global variable myCont) (FUNKTIONERT BEIM 1. TRIAL NICHT RICHTIG(sieht man bei konstanten parameter))
        for i in range(0, len(CPG.myCont)):
            CPG.myCont[i].RG.E.V=0
            CPG.myCont[i].RG.E.q=0
            CPG.myCont[i].RG.F.V=0
            CPG.myCont[i].RG.F.q=0
        #reset initpos of cpg and initiate cpc:
        mycpg.init_updates()

        ##################### simulate the net with minconis structure:
        mynet.inp[first].r=1.0   #if not randinit ... eigentlich=1.0
        if use_feedback:
            oldangles=np.degrees(np.array(mycpg.Angles)[[26,25,27,28]])#(r p y b)
            oldposition=give_position()
            if popcode:
                mynet.fdbx[:].r=fdbencode(oldangles[0],6,mynet.Nx,0,160)   #sigma ändern->minconi.wis ändern
                mynet.fdby[:].r=fdbencode(oldangles[1],4,mynet.Ny,-95.5, 8)
                mynet.fdbz[:].r=fdbencode(oldangles[2],5,mynet.Nz,-32, 80)
                mynet.fdbw[:].r=fdbencode(oldangles[3],5,mynet.Nw,15, 106)
            if not popcode:
                mynet.fdbx[0].r=oldangles[0]/160
                mynet.fdby[0].r=(oldangles[1]+95.5)/(8+95.5)
                mynet.fdbz[0].r=(oldangles[2]+32)/(32+80)
                mynet.fdbw[0].r=(oldangles[3]-15)/(106+15)
                mynet.fdbx[1].r=1-mynet.fdbx[0].r
                mynet.fdby[1].r=1-mynet.fdby[0].r
                mynet.fdbz[1].r=1-mynet.fdbz[0].r
                mynet.fdbw[1].r=1-mynet.fdbw[0].r
        minconi.simulate(chunktime1)
        #für später(critic) merken:
        #feedback_vector=np.append(mynet.fdbx[0].r,mynet.fdby[0].r)#delete those
        #feedback_vector=np.append(feedback_vector,mynet.fdbz[:].r)#
        #feedback_vector=np.append(feedback_vector,mynet.fdbw[:].r)#
        feedback_vector=np.array([oldangles[0]/160,(oldangles[1]+95.5)/(8+95.5),(oldangles[2]+32)/(32+80),(oldangles[3]-15)/(106+15)])
        mynet.inp[0:2].r = 0.0
        mynet.fdbx[:].r=0.0
        mynet.fdby[:].r=0.0
        mynet.fdbz[:].r=0.0
        mynet.fdbw[:].r=0.0

        minconi.simulate(chunktime2)
        #####################

        rec = mynet.m.get()
        recz.append(rec)

        ########################### set CPG params with the network outputs:

        #output = rec['r'][-int(d_execution):, output_neuron] # neuron 100 over the last 200 ms
        output = rec['r'][-int(d_execution):, minconi.net.begin_out:minconi.net.begin_out+mynet.n_out]
        #print("______________OUTPUTSSS: ",output)
        output=np.array(output)
        output=np.average(output,axis=0)
        

        parr=[]
        for i in range(njoints_max):
            parr.append(output[i*6:i*6+6])
        #parrr[:,0] = np.clip( (1+parrr[:,0])*(5.0/2.0),1,5)  #,0.001,5)
        #parrr[:,1] = np.clip( (1+parrr[:,1])*(5.0/2.0),0.001,5) 
        #parrr[:,2] = np.clip(parrr[:,2]*4,-4,4)  
        #parrr[:,3] = np.clip( (1+parrr[:,3])*(10.0/2.0),0.001,10)  
        #parrr[:,4] = np.clip( (1+parrr[:,4]),0.5,2.0)   #.001
        #parrr[:,5] = np.clip( (1+parrr[:,5]),0.01,2.0) 
        parrr=np.array(parr)
        parrr[:,0] = np.clip( (1+parrr[:,0])*(5.0/2.0),.001,5)  #,0.001,5)
        parrr[:,1] = np.clip( (1+parrr[:,1])*(5.0/2.0),0.001,5) 
        parrr[:,2] = np.clip(parrr[:,2]*4,-4,4)  #-4
        parrr[:,3] = np.clip( (1+parrr[:,3])*(10.0/2.0),0.001,10)  
        parrr[:,4] = np.clip( (1+parrr[:,4]),0.001,2.0)   #.001
        parrr[:,5] = np.clip( (1+parrr[:,5]),0.01,2.0) 

        
        ###alpha,theta,Sf,Ss,InjCMF,TM
        #mycpg.set_patterns([.25], [0],  [5] ,[0.1] ,[1], [0.1])
        #mycpg.set_patterns([.25,.1], [0,0],  [5,5] ,[.1,0.1] ,[1,1], [.1,0.1])
        #mycpg.set_patterns([scaling(1.3),.1], [scaling(0.03),0],  [scaling2(1.9),5] ,[scaling2(6.1),0.1] ,[scalingICUR(2.9),1], [scaling(0.4),0.1])
        #mycpg.set_patterns(.1, 0,  5 ,0.1 ,1, 0.1)
        #print(output[0:6])
        mycpg.set_patterns(parrr[:,0]*0+.25,parrr[:,1]*0,parrr[:,2]*0+5,parrr[:,3]*0+.1,parrr[:,4]/2-0.4,parrr[:,5]*0+.3)
        #mycpg.set_patterns(parrr[:,0],parrr[:,1],parrr[:,2],parrr[:,3],parrr[:,4],parrr[:,5])
        
        #mycpg.set_patterns(scaling(parr[:,0]),scaling(parr[:,1]),scaling2(parr[:,2]),scaling2(parr[:,3]),scalingICUR(parr[:,4]),scaling(parr[:,5]))
        
        #########################   move:
        mycpg.loop_move(timechunk)
        trajecto=np.array(mycpg.trajectori)
        lentra=np.shape(trajecto)[0]
        trajecto2=np.zeros((lentra,3))
        for tr_i in range(lentra):
            curpos9=give_position(a=trajecto[tr_i,0],b=trajecto[tr_i,1],c=trajecto[tr_i,2],d=trajecto[tr_i,3])    #stimmt reihenfolge?
            trajecto2[tr_i]=np.array(curpos9)
        trajectory=np.concatenate((trajectory,trajecto2),axis=0)                                    #und stimmt deg/rad?

        #print(mycpg.Angles)
        Ahist.append(np.degrees(np.array(mycpg.Angles)[[26,25,27,28]]))#r p y b 
        
        #print("_______________là: loop-moved")

        ###########################calculate error and then learn:
        position=give_position()
        Phist.append(position)
        #haut nur beim ersten mal hin
        #mycpg.plot_layers()
        error = (np.linalg.norm(np.array(target) - np.array(position)))**1
        learnerror=error-(np.linalg.norm(np.array(target) - np.array(oldposition)))
        #learnerror=min(0,learnerror)
        penalty=-1*min(0,np.linalg.norm(np.array(position-oldposition))-0.2)   #doesnt work at all
        #print(np.linalg.norm(np.array(position-oldposition)),penalty,learnerror)
        #learnerror=0*learnerror+penalty
        error=penalty
        if multiple_rewards:
            learncond=True
        else:
            learncond=timechunk==9
        if trial > learnstart and learncond and trialtype=="normal":
            errors.append(error)
            learnerrors.append(learnerror)
            global weightlist
            weightlist=np.concatenate((weightlist,[mynet.Wrec.w[0][0:np.shape(weightlist)[1]]]),axis=0)
            #learn minconi net (actor)
            mynet.Wrec.learning_phase = 1.0
            mynet.Wrec.error = learnerror
            #error_predictor=1
            #error_predictor=R_mean[first]
            #use critic:
            
            error_predictor=mycrit.predict(feedback_vector)
            mynet.Wrec.mean_error = error_predictor

            # Learn minconi net for one step (actor)
            minconi.step()
            # Reset the traces
            mynet.Wrec.learning_phase = 0.0
            mynet.Wrec.trace = 0.0
            _ = mynet.m.get() # to flush the recording of the last step
        if trialtype=="normal":
            #learn critic (from the beginning)
            mycrit.learnstep(x=feedback_vector,r=-learnerror,eta=.03)
            crit2.learnstep(x=feedback_vector,r=-learnerror,eta=.06)
            crit3.learnstep(x=feedback_vector,r=-learnerror,eta=.02)
            crit4.learnstep(x=feedback_vector,r=-learnerror,eta=.02)
            # Update the mean reward
            R_mean[first] = alpha * R_mean[first] + (1.- alpha) * learnerror
            r_mean[first] = alpha * r_mean[first] + (1.- alpha) * error
        
    return position,recz ,traces, r_mean, R_mean, initposi, Ahist, Phist, error,learnerror,errors,learnerrors, trajectory

####################################### now do the simulation:
if usekm==True:
    targetA=[ 0.07145494,  0.36328722, -0.04980991]#[0.04399564,0.22961777,0.25192178]
    targetB=[-0.35862834, -0.09199801, -0.04854833]#[0.11422334,0.15786776,0.26741575]
if usekm==False:
    targetA=[-0.31723,-0.110844,0.20111076]
    targetB=[-0.244523,0.087608,0.204415]   #komma war eins weiter links (2.445), war sicher falsch. bild 3 in fotos sagt das gleiche
try:
    posis1=[]
    posis2=[]
    R_means1=[]
    R_means2=[]
    r_means1=[]
    r_means2=[]
    recordsAz=[]
    recordsBz=[]
    TRIAL=0
    os.system("echo 0 > ../cancel.txt")
    AhistA_ar=[]
    AhistB_ar=[]
    AhistA_arl=[]
    AhistB_arl=[]
    PhistA_ar=[]
    PhistB_ar=[]
    PhistA_arl=[]
    PhistB_arl=[]
    error_history=[]
    errorz=[]
    learnerrorz=[]
    seedlist=np.random.randint(10000,size=(1000))
    for trial in range(max_trials):
        cancel_content = open("../cancel.txt", "r")
        Cancel=str(cancel_content.read())
        if Cancel[0]!="0":break

        if trial%150==10:
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            nr_s_external=10
            nr_s_internal=1
            for kkk in range(nr_s_external):
                intextseed=seedlist[kkk]
                for kkkk in range(nr_s_internal):
                    posi110,recordsA10,tracesA10,r_mean10,R_mean10,initposi10,AhistA10,PhistA10,error110,learnerror110,errors110,learnerrors110,trajectory= trial_simulation(trial, 0, R_mean, "not_normal",intextseed)
                    xtraj=trajectory[:,0]
                    ytraj=trajectory[:,1]
                    ztraj=trajectory[:,2]
                    colorval=(kkk+1)/(nr_s_external)
                    colorvalinternal=(kkkk+1)/(nr_s_internal)/2+.3
                    ax.plot(xtraj, ytraj, ztraj,color=[colorval,0,1-colorval,colorvalinternal],linewidth=.5)
                ax.plot([xtraj[0],xtraj[0]+0.001],[ytraj[0],ytraj[0]+0.001],[ztraj[0],ztraj[0]+0.001],linewidth=2,color="black")
            ax.plot([targetA[0],targetA[0]+0.001],[targetA[1],targetA[1]],[targetA[2],targetA[2]],linewidth=10)          
            plt.title("3D_Tr"+str(TRIAL))
            plt.savefig(savepa+"_3D_Tr"+str(TRIAL))
            #plt.legend()

        print('Trial', trial)
        posi1,recordsA,tracesA,r_mean,R_mean,initposi,AhistA,PhistA,error1,learnerror1,errors1,learnerrors1,trajectory= trial_simulation(trial, 0, R_mean, "normal","noseed")
        (posi2, recordsB, tracesB, r_mean, R_mean, initposi, AhistB, PhistB, error2, learnerror2, errors2, learnerrors2)= (posi1, recordsA, tracesA, r_mean, R_mean, initposi, AhistA, PhistA, error1, learnerror1, errors1,learnerrors1)#trial_simulation(trial, 0, R_mean)
        
        if trial == 0:
            recordsA_first=recordsA
            recordsB_first=recordsB
            AhistA_first=np.array(AhistA)
            AhistB_first=np.array(AhistB)
            PhistA_first=np.array(PhistA)
            PhistB_first=np.array(PhistB)

        if trial <20:
            AhistA_ar.append(np.array(AhistA))
            AhistB_ar.append(np.array(AhistB))
            PhistA_ar.append(np.array(PhistA))
            PhistB_ar.append(np.array(PhistB))
        #nicht überschneiden lassen
        startrecA=max_trials-40
        if trial == startrecA:
            recordsA_first=recordsA
            recordsB_first=recordsB
            AhistA_first=np.array(AhistA)
            AhistB_first=np.array(AhistB)
            PhistA_first=np.array(PhistA)
            PhistB_first=np.array(PhistB)
        if trial>=startrecA and trial<startrecA+40:
            AhistA_arl.append(np.array(AhistA))
            AhistB_arl.append(np.array(AhistB))
            PhistA_arl.append(np.array(PhistA))
            PhistB_arl.append(np.array(PhistB))
        posis1.append(posi1)
        posis2.append(posi2)
        R_means1.append(R_mean[0])
        R_means2.append(R_mean[1])
        r_means1.append(r_mean[0])
        r_means2.append(r_mean[1])
        #R_means1.append(learnerror1)
        #R_means2.append(learnerror2)
        #r_means1.append(error1)
        #r_means2.append(error2)
        if trial>learnstart:
            errorz.append((np.array(errors1)+np.array(errors2))/2)
            learnerrorz.append((np.array(learnerrors1)+np.array(learnerrors2))/2)
        error_history.append((error1+error2)/2)
        TRIAL+=1
except KeyboardInterrupt:
    pass
posis1=np.array(posis1)
posis2=np.array(posis2)
errorz=np.array(errorz)
learnerrorz=np.array(learnerrorz)

print("____________________")
figname="f"+str(minconi.var_f).replace(".","-")+"_"+"eta"+str(minconi.var_eta).replace(".","-")+"_g"+str(minconi.var_g).replace(".","-")+"_N"+str(minconi.var_N).replace(".","-")+"_A"+str(minconi.var_A).replace(".","-")
print("var_f,var_eta,var_g,var_N: ",figname)
####
rAf_t = []
for i in range(nchunks):  # 10=nr of chunks
    rAf_t.append(recordsA_first[i]['r'].T)
rAf_t = np.array(rAf_t)
raftsh = np.shape(rAf_t)
rAf_t = np.transpose(rAf_t, (1, 0, 2))
rAf_t = np.reshape(rAf_t, (raftsh[1], raftsh[0]*raftsh[2]))
#######
#delete:
rBf_t = []
for i in range(nchunks):  # 10=nr of chunks
    rBf_t.append(recordsB_first[i]['r'].T)
rBf_t = np.array(rBf_t)
rbftsh = np.shape(rBf_t)
rBf_t = np.transpose(rBf_t, (1, 0, 2))
rBf_t = np.reshape(rBf_t, (rbftsh[1], rbftsh[0]*rbftsh[2]))
#######
####
rAl_t = []
for i in range(nchunks):  # 10=nr of chunks
    rAl_t.append(recordsA[i]['r'].T)  # last records
rAl_t = np.array(rAl_t)
raltsh = np.shape(rAl_t)
rAl_t = np.transpose(rAl_t, (1, 0, 2))
rAl_t = np.reshape(rAl_t, (raltsh[1], raltsh[0]*raltsh[2]))
####
ehname="../error_h/"+str(sim)+"error.npy"
np.save(ehname,error_history)
print("saved as",ehname)
plt.savefig(savepa+"_3D_Tr"+str(TRIAL)+"_end")
plt.figure()
plt.subplot(1,2,1)
plt.plot(weightlist)
plt.subplot(1,2,2)
plt.imshow((weightlist-weightlist[1,:]).T,cmap="seismic",aspect="auto", interpolation='none')
plt.colorbar()
plt.savefig(savepa+"_weights")
plt.figure()
plt.plot(mycrit.me,color="black",linewidth=.8,label="critic-error (gaussian)")
plt.plot(crit2.me,label="const",linewidth=.8)
#plt.plot(crit3.me,label="lin",linewidth=.8)
#plt.plot(crit4.me,label="squared")
print(sum(mycrit.me))
print(sum(crit2.me))
print(sum(crit3.me))
print(sum(crit4.me))
plt.legend()
plt.savefig(savepa+"_critics")
plt.figure()

quarter=int((max_trials-learnstart)/4)
if doplot2:
    plt.plot((errorz.T)[:,0:quarter],color=(0,0,0))
    plt.plot((errorz.T)[:,quarter:2*quarter],color=(.25,.25,.25))
    plt.plot((errorz.T)[:,2*quarter:3*quarter],color=(.5,.5,.5))
    plt.plot((errorz.T)[:,3*quarter:],color=(.75,.75,.75))
    plt.show()
    plt.plot(np.average(errorz,axis=0),color=(0,0,0))
    plt.plot(np.average(learnerrorz,axis=0),color=(.5,.5,.5))
    plt.show()
try: 
    fig=plt.figure(figsize=(20, 20))
except:
    print(colored("don't use ssh -X with screen because it doesn't work for some reason.","red"))

ax = plt.subplot(241)
ax.imshow(rAf_t, aspect='auto', origin='lower')
ax.set_title('ALL 10 100ms-chunks of PopulationA of FIRST trial')
    
    
ax = plt.subplot(242)
AhistA_ar=np.array(PhistA_ar)
PhistA_ar=np.array(PhistA_ar)
for i in range(len(AhistA_ar)):
    alphap=(i+1)/(1+len(AhistA_ar))
    ax.plot(AhistA_ar[i,:,0],linewidth=.6,c=(1,0,0,alphap**2),label="1")
    ax.plot(AhistA_ar[i,:,1],linewidth=.6,c=(0,0,1,alphap**2),label="2")
    ax.plot(AhistA_ar[i,:,2],linewidth=.6,c=(0,1,0,alphap**2),label="3")
    #ax.plot(AhistA_ar[i,:,3],linewidth=.6,c=(1,0,1,alphap**2))
ax.set_title("A Angle development in FIRST "+str(len(AhistA_ar))+" trials")

ax=plt.subplot(246)
AhistA_arl=np.array(PhistA_arl)
PhistA_arl=np.array(PhistA_arl)
for i in range(len(AhistA_arl)):
    alphap=(i+1)/(1+len(AhistA_arl))
    ax.plot(AhistA_arl[i,:,0],linewidth=.6,c=(1,0,0,alphap**2),label="1")
    ax.plot(AhistA_arl[i,:,1],linewidth=.6,c=(0,0,1,alphap**2),label="2")
    ax.plot(AhistA_arl[i,:,2],linewidth=.6,c=(0,1,0,alphap**2),label="3")
    #ax.plot(AhistA_arl[i,:,3],linewidth=.6,c=(1,0,1,alphap**2))
ax.set_title("A Angle development in LATER "+str(len(AhistA_arl))+" trials")


ax = plt.subplot(245)
ax.imshow(rAl_t, aspect='auto', origin='lower', interpolation='none')#rAl_t
ax.set_title('ALL 10 100ms-chunks of PopulationA of last trial')



ax = plt.subplot(243)
#ax.plot(initialA[:, mynet.begin_out], label='before')

ax.plot(posis1[:,0]-targetA[0],  label='end pos x',color="blue")
#ax.plot((initposi[0]-targetA[0])*np.ones(len(posis1)), label='init x',color='blue',linewidth=.5)   #if not randinit
ax.plot(posis1[:,1]-targetA[1],  label='end pos y',color="red")
#ax.plot((initposi[1]-targetA[1])*np.ones(len(posis1)), label='init y',color='red',linewidth=.5)
ax.plot(posis1[:,2]-targetA[2],  label='end pos z',color="green")
#ax.plot((initposi[2]-targetA[2])*np.ones(len(posis1)), label='init z',color='green',linewidth=.5)
if usekm==False:
    ax.plot(0*np.ones(len(posis1)), label='target =0')
if usekm==True:
    ax.plot(0*np.ones(len(posis1)), label='target x,y,z=0',color='black')
ax.set_ylim((-0.5, 0.5))
ax.set_title('target A trials')

ax = plt.subplot(244)
#ax.plot(initialB[:, mynet.begin_out:mynet.begin_out+1], label='before')

ax.plot(posis2[:,0]-targetB[0],  label='end pos x',color="blue")
ax.plot((initposi[0]-targetB[0])*np.ones(len(posis2)), label='init x',color='blue',linewidth=.5)
ax.plot(posis2[:,1]-targetB[1],  label='end pos y',color="red")
ax.plot((initposi[1]-targetB[1])*np.ones(len(posis2)), label='init y',color='red',linewidth=.5)
ax.plot(posis2[:,2]-targetB[2],  label='end pos z',color="green")
ax.plot((initposi[2]-targetB[2])*np.ones(len(posis2)), label='init z',color='green',linewidth=.5)
if usekm==False:
    ax.plot(0*np.ones(len(posis1)), label='target =0')
if usekm==True:
    ax.plot(0*np.ones(len(posis1)), label='target x,y,z=0',color='black')
    #ax.plot(0*np.ones(len(posis1)), label='targety')
ax.set_ylim((-0.5,0.5))
ax.set_title('target B trials')
#ax.legend()

ax = plt.subplot(247)
ax.plot(R_means1,label='mean_learnerror 1')
ax.plot(r_means1,label='mean_penalty1')
#ax.set_ylim((-0.3,0.12))
ax.legend()

ax = plt.subplot(248)
ax.plot(R_means2,label='mean_learned_error 2')
ax.plot(r_means2,label='mean_error 2')
#ax.set_ylim((-0.3,0.12))
ax.legend()

fig.suptitle("params:"+figname, fontsize=14)
plt.savefig(savepa+"__")
savefiginp="y"#input("save the figure? (y/n)")
if savefiginp=="y":
    savepa="../bilder/temporary/"
    #plt.savefig(savepa+figname+"_Tr"+str(TRIAL))	
    #print("saved in "+savepa)
else:
    print("not saved")
if showplot:
    plt.show()
