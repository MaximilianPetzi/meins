#just use one meta.py at a time, except with params.py
#terminal auf CPG_iCub
usekm=True      #use kinematic model or not (simulator, not on ssh)
skipcpg=False   #just use scaled minconi output after last timechunk as final angels instead of using the CPG. 
multiple_rewards=True
use_feedback=True
#picture usually saved in bilder/temporary/, and this folder is always cleared before
showplot=False
doplot2=False
nchunks=1
max_trials=1000
chunktime=200   #also change var_f inversely
d_execution=200  #average over last d_execution timesteps
learnstart=20
import os
import time
from termcolor import colored
import numpy as np
import sys
sys.path.append("../CPG_iCub")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--m", help="set to eg. 1, if meta is called by params")    #optional argument
parser.add_argument("--s", help="number of the simulation")
args = parser.parse_args()
sim=args.s

if not args.m:
    os.system("rm ../bilder/temporary/*")
    Paramarr=np.zeros((6))
    default="y"#input("use default?(y/n)")
    if default=="y":
        var_f=9
        var_eta=.6
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
alpha = 0.75 # 0.33
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

####################################################
def trial_simulation(trial,first,R_mean):
    traces = []
    #start cpg:
    mycpg=CPG.cpg()
    
    def give_position():
        if usekm==False:
            position=myreadr.read()
        if usekm==True:
            if skipcpg:
                the1=(parr[0,0]+1)*80
            if not skipcpg:
                the1=mycpg.Angles[mycpg.LShoulderRoll]
            the2=mycpg.Angles[mycpg.LElbow]
            if skipcpg:
                the3=(parr[1,0]+1)/2*(8+95.5)-95.5
            if not skipcpg:
                the3=mycpg.Angles[mycpg.LShoulderPitch]
            the4=mycpg.Angles[mycpg.LShoulderYaw]
            position=kinematic_model.wrist_position([the4,the3,the1,the2])[0:3]
        return position

    #move to starting position:
    if usekm==False:
        mycpg.move_to_init()
    if usekm==True:
        mycpg.move_to_init2()
    #time.sleep(2.5)  # wait until robot finished motion


    initposi=give_position()# needed only for plotting
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
        mynet.inp[first].r=1.0
        if use_feedback:
            oldposition=give_position()
            mynet.fdb[:].r=4.5*np.array(oldposition)#faktor 2.5, um die auswirkung groß genug zu halten
        
        minconi.simulate(chunktime)
        
        mynet.inp[0:2].r = 0.0
        mynet.fdb[:].r=0.0

        minconi.simulate(chunktime)
        #####################

        rec = mynet.m.get()
        recz.append(rec)

        ########################### set CPG params with the network outputs:

        #output = rec['r'][-int(d_execution):, output_neuron] # neuron 100 over the last 200 ms
        output = rec['r'][-int(d_execution):, minconi.net.begin_out:minconi.net.begin_out+mynet.n_out]
        #print("______________OUTPUTSSS: ",output)
        output=np.array(output)
        output=np.average(output,axis=0)

        ###alpha,theta,Sf,Ss,InjCMF,TM
        #mycpg.set_patterns([.25], [0],  [5] ,[0.1] ,[1], [0.1])
        #mycpg.set_patterns([.25,.1], [0,0],  [5,5] ,[.1,0.1] ,[1,1], [.1,0.1])
        #mycpg.set_patterns([scaling(1.3),.1], [scaling(0.03),0],  [scaling2(1.9),5] ,[scaling2(6.1),0.1] ,[scalingICUR(2.9),1], [scaling(0.4),0.1])
        #mycpg.set_patterns(.1, 0,  5 ,0.1 ,1, 0.1)
        #print(output[0:6])
        
        parr=[]
        for i in range(njoints_max):
            parr.append(output[i*6:i*6+6])
        parrr=np.array(parr)
        parrr[:,0] = np.clip( (1+parrr[:,0])*(5.0/2.0),0.001,5)  
        parrr[:,1] = np.clip( (1+parrr[:,1])*(5.0/2.0),0.001,5) 
        parrr[:,2] = np.clip(parrr[:,2]*4,-4,4)  
        parrr[:,3] = np.clip( (1+parrr[:,3])*(10.0/2.0),0.001,10)  
        parrr[:,4] = np.clip( (1+parrr[:,4]),0.01,2.0)  
        parrr[:,5] = np.clip( (1+parrr[:,5]),0.01,2.0) 
        
        mycpg.set_patterns(parrr[:,0],parrr[:,1],parrr[:,2],parrr[:,3],parrr[:,4],parrr[:,5])
        #mycpg.set_patterns(scaling(parr[:,0]),scaling(parr[:,1]),scaling2(parr[:,2]),scaling2(parr[:,3]),scalingICUR(parr[:,4]),scaling(parr[:,5]))
        
        #########################   move:
        mycpg.loop_move(timechunk)

        #print(mycpg.Angles)
        Ahist.append(np.degrees(np.array(mycpg.Angles)[[26,25,27,28]]))
        
        #print("_______________là: loop-moved")

        ###########################calculate error and then learn:
        position=give_position()
        Phist.append(position)
        #haut nur beim ersten mal hin
        #mycpg.plot_layers()
        error = (np.linalg.norm(np.array(target) - np.array(position)))**1
        learnerror=error/(np.linalg.norm(np.array(target) - np.array(oldposition)))
        
        if multiple_rewards:
            learncond=True
        else:
            learncond=timechunk==9
        if trial > learnstart and learncond:
            errors.append(error)
            learnerrors.append(learnerror)
            # Apply the learning rule
            mynet.Wrec.learning_phase = 1.0
            mynet.Wrec.error = learnerror
            mynet.Wrec.mean_error = R_mean[first]
            # Learn for one step
            minconi.step()
            # Reset the traces
            mynet.Wrec.learning_phase = 0.0
            mynet.Wrec.trace = 0.0
            _ = mynet.m.get() # to flush the recording of the last step

        # Update the mean reward
        R_mean[first] = alpha * R_mean[first] + (1.- alpha) * learnerror
        r_mean[first] = alpha * r_mean[first] + (1.- alpha) * error

    return position,recz ,traces, r_mean, R_mean, initposi, Ahist, Phist, error,learnerror,errors,learnerrors

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
    for trial in range(max_trials):
        cancel_content = open("../cancel.txt", "r")
        Cancel=str(cancel_content.read())
        if Cancel[0]!="0":break
        print('Trial', trial)
        posi1, recordsA, tracesA, r_mean, R_mean, initposi, AhistA, PhistA, error1, learnerror1, errors1,learnerrors1= trial_simulation(trial, 0, R_mean)
        posi2, recordsB, tracesB, r_mean, R_mean, initposi, AhistB, PhistB, error2, learnerror2, errors2, learnerrors2= trial_simulation(trial, 1, R_mean)
        
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
        #R_means1.append(R_mean[0])
        #R_means2.append(R_mean[1])
        #r_means1.append(r_mean[0])
        #r_means2.append(r_mean[1])
        R_means1.append(learnerror1)
        R_means2.append(learnerror2)
        r_means1.append(error1)
        r_means2.append(error2)
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

print("___errorzzrzzrzz_",np.shape(errorz),np.shape(learnerrorz))
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("TkAgg")
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
    alpha=(i+1)/(1+len(AhistA_ar))
    ax.plot(AhistA_ar[i,:,0],linewidth=.6,c=(1,0,0,alpha**2),label="1")
    ax.plot(AhistA_ar[i,:,1],linewidth=.6,c=(0,0,1,alpha**2),label="2")
    ax.plot(AhistA_ar[i,:,2],linewidth=.6,c=(0,1,0,alpha**2),label="3")
    #ax.plot(AhistA_ar[i,:,3],linewidth=.6,c=(1,0,1,alpha**2))
ax.set_title("A Angle development in FIRST "+str(len(AhistA_ar))+" trials")

ax=plt.subplot(246)
AhistA_arl=np.array(PhistA_arl)
PhistA_arl=np.array(PhistA_arl)
for i in range(len(AhistA_arl)):
    alpha=(i+1)/(1+len(AhistA_arl))
    ax.plot(AhistA_arl[i,:,0],linewidth=.6,c=(1,0,0,alpha**2),label="1")
    ax.plot(AhistA_arl[i,:,1],linewidth=.6,c=(0,0,1,alpha**2),label="2")
    ax.plot(AhistA_arl[i,:,2],linewidth=.6,c=(0,1,0,alpha**2),label="3")
    #ax.plot(AhistA_arl[i,:,3],linewidth=.6,c=(1,0,1,alpha**2))
ax.set_title("A Angle development in LATER "+str(len(AhistA_arl))+" trials")


ax = plt.subplot(245)
ax.imshow(rBf_t, aspect='auto', origin='lower')#rAl_t
ax.set_title('ALL 10 100ms-chunks of PopulationA of last trial')



ax = plt.subplot(243)
#ax.plot(initialA[:, mynet.begin_out], label='before')

ax.plot(posis1[:,0]-targetA[0],  label='end pos x',color="blue")
ax.plot((initposi[0]-targetA[0])*np.ones(len(posis1)), label='init x',color='blue',linewidth=.5)
ax.plot(posis1[:,1]-targetA[1],  label='end pos y',color="red")
ax.plot((initposi[1]-targetA[1])*np.ones(len(posis1)), label='init y',color='red',linewidth=.5)
ax.plot(posis1[:,2]-targetA[2],  label='end pos z',color="green")
ax.plot((initposi[2]-targetA[2])*np.ones(len(posis1)), label='init z',color='green',linewidth=.5)
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
ax.plot(R_means1,label='mean_learnerror')
ax.plot(r_means1,label='mean_error')
#ax.set_ylim((-0.3,0.12))


ax = plt.subplot(248)
ax.plot(R_means2,label='mean_learned_error')
ax.plot(r_means2,label='mean_error')
#ax.set_ylim((-0.3,0.12))
ax.legend()
fig.suptitle("params:"+figname, fontsize=14)
savefiginp="y"#input("save the figure? (y/n)")
if savefiginp=="y":
    savepa="../bilder/temporary/"
    plt.savefig(savepa+figname+"_Tr"+str(TRIAL))	
    print("saved in "+savepa)
else:
    print("not saved")
if showplot:
    plt.show()
