#just use one meta.py at a time, except with params.py
#terminal auf CPG_iCub
usekm=True      #use kinematic model or not (simulator, not on ssh)
showall=True    #plot whole trial activities or not (just first and last 100ms)
skipcpg=False   #just use scaled minconi output after last timechunk as final angels instead of using the CPG. 
#picture usually saved in bilder/temporary/, and this folder is always cleared before
showplot=True
max_trials=10
chunktime=200   #also change var_f inversely
d_execution=200  #average over last d_execution timesteps
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
        var_f=5
        var_eta=.5
        var_g=1.5
        var_N=200
        var_A=2
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
##

if usekm==False:
    import readr
if usekm==True:
    import kinematic_model
    #print(kinematic_model.wrist_position([0,0,0,0])[0:3])
import minconi
#import tlib
#import tlib_ordner.tlib2
#print(tl1.a,tl2.a)
#ade=tlib.mensch("dolf\n")


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
alpha = 0.75 # 0.33
import CPG
#print("__________________________qui: CPG imported")



def scaling(x):
    return x+1+0.001#(-.5-1/(x-1))*3
def scaling2(x):
    return 5*(x+1+0.001)
def scalingICUR(x):
    return 8*x

def trial_simulation(trial,first,R_mean):
    traces = []
    mynet.reinit()
    
    #move robot to point using the parameters
    #start cpg:
    mycpg=CPG.cpg()
    
    #reset cpg diff equation (global variable myCont) (FUNKTIONERT BEIM 1. TRIAL NICHT RICHTIG(sieht man bei konstanten parameter))
    for i in range(0, len(CPG.myCont)):
        CPG.myCont[i].RG.E.V=0
        CPG.myCont[i].RG.E.q=0
        CPG.myCont[i].RG.F.V=0
        CPG.myCont[i].RG.F.q=0
    
    if len(traces) == 0 and trial == 0:
        global tmovetoinit, tsimulate
        tmovetoinit = 0
        tsimulate = 0
    #move to starting position:
    tspre=time.time()
    if usekm==False:
        mycpg.move_to_init()
    if usekm==True:
        mycpg.move_to_init2()
    tmovetoinit+=time.time()-tspre
    #time.sleep(2.5)  # wait until robot finished motion


    #initiate:
    mycpg.init_updates()
    

    
    if usekm==False:
        initposi=myreadr.read()
        #print(colored("initpos "+str(initposi),"yellow"))
    if usekm==True:
        the1=mycpg.Angles[mycpg.LShoulderRoll]
        the2=mycpg.Angles[mycpg.LElbow]
        the3=mycpg.Angles[mycpg.LShoulderPitch]
        the4=mycpg.Angles[mycpg.LShoulderYaw]
        initposi=kinematic_model.wrist_position([the4,the3,the1,the2])[0:3]
        #print(colored("initpos "+str(initposi),"yellow"))
    #print("init u und v: ", CPG.cpg.__dict__)
    #print("\n\n..mycpg: ",mycpg.__dict__)
    #print(CPG.cpg.myCont[0].RG.E.q)
    #print(mycpg.myCont[0].RG.E.V)
    
    
    mynet.inp[first].r = 1.0
    
    #mynet.Wrec.eta*=np.exp(1/200*np.log(101))#x^200=101
    #print(mynet.Wi.w[0][0:10])
    recz=[]
    Ahist=[]
    for timechunk in range(2):
        #print("\n::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        #print(colored("\ntimechunk "+str(timechunk),"green"))
        tspre=time.time()
        
        minconi.simulate(chunktime)
        tsimulate+=time.time()-tspre
        # Read the output parameters(robot inputs)#
        
        #remove input after first 100ms
        mynet.inp[0:2].r = 0.0


        rec = mynet.m.get()
        recz.append(rec)
        #output = rec['r'][-int(d_execution):, output_neuron] # neuron 100 over the last 200 ms
        output = rec['r'][-int(d_execution):, minconi.net.begin_out:minconi.net.begin_out+mynet.n_out]
        #print("______________OUTPUTSSS: ",output)
        output=np.array(output)
        output=np.average(output,axis=0)

        ###alpha,theta,Sf,Ss,InjCMF,TM
        #mycpg.set_patterns(.25, 0,  5 ,0.1 ,1, 0.1)
        #mycpg.set_patterns([.25,.1], [0,0],  [5,5] ,[.1,0.1] ,[1,1], [.1,0.1])
        #mycpg.set_patterns([scaling(1.3),.1], [scaling(0.03),0],  [scaling2(1.9),5] ,[scaling2(6.1),0.1] ,[scalingICUR(2.9),1], [scaling(0.4),0.1])
        #mycpg.set_patterns(.1, 0,  5 ,0.1 ,1, 0.1)
        #print(output[0:6])
        parr=[]
        for i in range(njoints_max):
            parr.append(output[i*6:i*6+6])
        parr=np.array(parr)
        mycpg.set_patterns(scaling(parr[:,0]),scaling(parr[:,1]),scaling2(parr[:,2]),scaling2(parr[:,3]),scalingICUR(parr[:,4]),scaling(parr[:,5]))
        if timechunk==5:
            #print("patterns set to:",scaling(parr[0,0]),scaling(parr[0,1]),scaling2(parr[0,2]),scaling2(parr[0,3]),scalingICUR(parr[0,4]),scaling(parr[0,5]))
            pass
        #moove:
         
        mycpg.loop_move(timechunk)
        #print(mycpg.Angles)
        Ahist.append(np.degrees(np.array(mycpg.Angles)[[26,25]]))
        #print("_______________là: loop-moved")
    
    mynet.inp[first].r = 0.0
    
    if usekm==False:
        position=myreadr.read()
    if usekm==True:
        if skipcpg:
            the1=(parr[0,0]+1)*80
            #print(the1)
        if not skipcpg:
            the1=mycpg.Angles[mycpg.LShoulderRoll]
        the2=mycpg.Angles[mycpg.LElbow]
        if skipcpg:
            the3=(parr[1,0]+1)/2*(8+95.5)-95.5
            #print(the3)
        if not skipcpg:
            the3=mycpg.Angles[mycpg.LShoulderPitch]
        the4=mycpg.Angles[mycpg.LShoulderYaw]
        position=kinematic_model.wrist_position([the4,the3,the1,the2])[0:3]
          
    #haut nur beim ersten mal hin
    #mycpg.plot_layers()
    
    #compute target 1:hand right 2:hand left
    if usekm==False:
        target = targetA if first == 0 else targetB
    if usekm==True:
        target= targetA if first == 0 else targetB
    error = (np.linalg.norm(np.array(target) - np.array(position)))**1
    #print(colored("target: "+str(first)+", "+str(target)+" \nposition : "+str(position),"white"))
    #print(colored("error: "+str(error)+" Mean: "+str(R_mean[first]),"red"))
    
    #print('Target:', target, '\tOutput:', position, '\tError:',  "%0.3f" % error, '\tMean:', "%0.3f" % R_mean[first])
    #print("WEIGHT: ",mynet.Wrec[3].w[1:3])
    #print(mynet.Wrec.w)
    if trial > 50:
        # Apply the learning rule
        mynet.Wrec.learning_phase = 1.0
        mynet.Wrec.error = error
        mynet.Wrec.mean_error = R_mean[first]
        # Learn for one step
        minconi.step()
        # Reset the traces
        mynet.Wrec.learning_phase = 0.0
        mynet.Wrec.trace = 0.0
        _ = mynet.m.get() # to flush the recording of the last step

    # Update the mean reward
    R_mean[first] = alpha * R_mean[first] + (1.- alpha) * error

    return position,recz ,traces, R_mean, initposi, Ahist, error
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
    ttotal=time.time()
    recordsAz=[]
    recordsBz=[]
    TRIAL=0
    os.system("echo 0 > ../cancel.txt")
    AhistA_ar=[]
    AhistB_ar=[]
    error_history=[]
    for trial in range(max_trials):
        cancel_content = open("../cancel.txt", "r")
        Cancel=str(cancel_content.read())
        if Cancel[0]!="0":break
        print('Trial', trial)
        posi1, recordsA, tracesA, R_mean, initposi, AhistA, error= trial_simulation(trial, 0, R_mean)
        posi2, recordsB, tracesB, R_mean, initposi, AhistB, error= trial_simulation(trial, 1, R_mean)
        if trial == 0:
            recordsA_first=recordsA
            recordsB_first=recordsA
            AhistA_first=np.array(AhistA)
            AhistB_first=np.array(AhistB)
        if trial <20:
            AhistA_ar.append(np.array(AhistA))
            AhistB_ar.append(np.array(AhistB))
        posis1.append(posi1)
        posis2.append(posi2)
        R_means1.append(R_mean[0])
        R_means2.append(R_mean[1])


        error_history.append(error)
        TRIAL+=1
except KeyboardInterrupt:
    pass
posis1=np.array(posis1)
posis2=np.array(posis2)
print("____________________")
print("total time: ",time.time()-ttotal, "davon simulate(chunktime): ", tsimulate, "davon movetoinit:", tmovetoinit)
figname="f"+str(minconi.var_f).replace(".","-")+"_"+"eta"+str(minconi.var_eta).replace(".","-")+"_g"+str(minconi.var_g).replace(".","-")+"_N"+str(minconi.var_N).replace(".","-")+"_A"+str(minconi.var_A).replace(".","-")
print("var_f,var_eta,var_g,var_N: ",figname)
####
rAf_t = []
for i in range(2):  # 10=nr of chunks
    rAf_t.append(recordsA_first[i]['r'].T)
rAf_t = np.array(rAf_t)
raftsh = np.shape(rAf_t)
rAf_t = np.transpose(rAf_t, (1, 0, 2))
rAf_t = np.reshape(rAf_t, (raftsh[1], raftsh[0]*raftsh[2]))
####
rAl_t = []
for i in range(2):  # 10=nr of chunks
    rAl_t.append(recordsA[i]['r'].T)  # last records
rAl_t = np.array(rAl_t)
raltsh = np.shape(rAl_t)
rAl_t = np.transpose(rAl_t, (1, 0, 2))
rAl_t = np.reshape(rAl_t, (raltsh[1], raltsh[0]*raltsh[2]))
####
ehname="../error_h/"+str(sim)+"error.npy"
np.save(ehname,error_history)
print("saved as",ehname)

import matplotlib.pyplot as plt

import matplotlib
#matplotlib.use("TkAgg")

try: 
    fig=plt.figure(figsize=(20, 20))
except:
    print(colored("don't use ssh -X with screen because it doesn't work for some reason.","red"))

ax = plt.subplot(241)
if not showall:
    ax.imshow(recordsA_first[0]['r'].T, aspect='auto', origin='lower')
    ax.set_title('FIRST 100ms PopulationA of FIRST trial')
if showall:
    ax.imshow(rAf_t, aspect='auto', origin='lower')
    ax.set_title('ALL 10 100ms-chunks of PopulationA of FIRST trial')
    
    ax = plt.subplot(242)
    ax.plot(AhistA_first[:,0],c="r",label="LShoulderRoll")
    ax.plot(AhistA_first[:,1],c="b",label="LShoulderPitch")
    ax.set_title("Angle development in first trial")
    ax.legend()
    ax = plt.subplot(246)
    AhistA_ar=np.array(AhistA_ar)
    for i in range(len(AhistA_ar)):
        alpha=(i+1)/(1+len(AhistA_ar))
        ax.plot(AhistA_ar[i,:,0],linewidth=.6,c=(1,0,0,alpha**2))
        ax.plot(AhistA_ar[i,:,1],linewidth=.6,c=(0,0,1,alpha**2))
    ax.set_title("Angle development in FIRST "+str(len(AhistA_ar))+" trials")
if not showall:
    ax = plt.subplot(242)
    ax.imshow(recordsA_first[-1]['r'].T, aspect='auto', origin='lower')
    ax.set_title('LAST 100ms PopulationA FIRST trial')

ax = plt.subplot(245)
if not showall:
    ax.imshow(recordsA[0]['r'].T, aspect='auto', origin='lower')
    ax.set_title('FIRST 100ms PopulationA of last trial')
if showall:
    ax.imshow(rAl_t, aspect='auto', origin='lower')
    ax.set_title('ALL 10 100ms-chunks of PopulationA of last trial')

if not showall:
    ax = plt.subplot(246)
    ax.imshow(recordsA[-1]['r'].T, aspect='auto', origin='lower')
    ax.set_title('LAST 100ms PopulationA last trial')

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
ax.plot(R_means1,label='mean_error')
#ax.set_ylim((-0.3,0.12))


ax = plt.subplot(248)
ax.plot(R_means2,label='mean_error')
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
