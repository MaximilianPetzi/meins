usekm=True
#terminal auf CPG_iCub
import sys
import os
sys.path.append("../CPG_iCub")

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

import numpy as np
import os
import time
from termcolor import colored
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

njoints_max=2
mynet=minconi.net(n_out=6*njoints_max) #mynet has many class variables
# mynet.init_w = minconi.net.Wrec.w #instance variable implicitly created

if usekm==False:
    myreadr=readr.readr()
    print(colored("icub started "+str(myreadr.read()),"yellow"))

# Compute the mean reward per trial
R_mean = np.zeros((2))
alpha = 0.75 # 0.33
import CPG
print("__________________________qui: CPG imported")



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
        print(colored("initpos "+str(initposi),"yellow"))
    if usekm==True:
        the1=mycpg.Angles[mycpg.LShoulderRoll]
        the2=mycpg.Angles[mycpg.LElbow]
        the3=mycpg.Angles[mycpg.LShoulderPitch]
        the4=mycpg.Angles[mycpg.LShoulderYaw]
        initposi=kinematic_model.wrist_position([the4,the3,the1,the2])[0:3]
        print(colored("initpos "+str(initposi),"yellow"))
    #print("init u und v: ", CPG.cpg.__dict__)
    #print("\n\n..mycpg: ",mycpg.__dict__)
    #print(CPG.cpg.myCont[0].RG.E.q)
    #print(mycpg.myCont[0].RG.E.V)
    
    
    mynet.inp[first].r = 1.0
    
    #mynet.Wrec.eta*=np.exp(1/200*np.log(101))#x^200=101
    #print(mynet.Wi.w[0][0:10])
    recz=[]
    for timechunk in range(10):
        #print("\n::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        #print(colored("\ntimechunk "+str(timechunk),"green"))
        tspre=time.time()
        
        minconi.simulate(1000)
        tsimulate+=time.time()-tspre
        # Read the output parameters(robot inputs)#
        
        #remove input after first 100ms
        mynet.inp[0:2].r = 0.0


        rec = mynet.m.get()
        recz.append(rec)
        output = rec['r'][-1, minconi.net.begin_out:minconi.net.begin_out+mynet.n_out]
        #print("______________OUTPUTSSS: ",output)

        ###alpha,theta,Sf,Ss,InjCMF,TM
        #mycpg.set_patterns(.25, 0,  5 ,0.1 ,1, 0.1)
        #mycpg.set_patterns(.1, 0,  5 ,0.1 ,1, 0.1)
        #print(output[0:6])
        parr=[]
        for i in range(njoints_max):
            parr.append(output[i*6:i*6+6])
        parr=np.array(parr)
        mycpg.set_patterns(scaling(parr[:,0]),scaling(parr[:,1]),scaling2(parr[:,2]),scaling2(parr[:,3]),scalingICUR(parr[:,4]),scaling(parr[:,5]))
        if timechunk==5:
            print("patterns set to:",scaling(parr[0,0]),scaling(parr[0,1]),scaling2(parr[0,2]),scaling2(parr[0,3]),scalingICUR(parr[0,4]),scaling(parr[0,5]))

        #moove:
         
        mycpg.loop_move(timechunk)
        #print("_______________là: loop-moved")
    
    mynet.inp[first].r = 0.0
    
    if usekm==False:
        position=myreadr.read()
    if usekm==True:
        the1=mycpg.Angles[mycpg.LShoulderRoll]
        the2=mycpg.Angles[mycpg.LElbow]
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
    error = (np.linalg.norm(np.array(target) - np.array(position)))**2
    print(colored("target: "+str(first)+", "+str(target)+" \nposition : "+str(position),"white"))
    print(colored("error: "+str(error)+" Mean: "+str(R_mean[first]),"red"))
    
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

    return position,recz ,traces, R_mean, initposi
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
    os.system("echo 0 > ../meins/cancel.txt")
    for trial in range(50000):
        cancel_content = open("../meins/cancel.txt", "r")
        Cancel=str(cancel_content.read())
        if Cancel[0]!="0":break
        print('Trial', trial)
        posi1, recordsA, tracesA, R_mean, initposi = trial_simulation(trial, 0, R_mean)

        posi2, recordsB, tracesB, R_mean, initposi = trial_simulation(trial, 1, R_mean)
        if trial == 0:
            recordsA_first=recordsA
            recordsB_first=recordsA
        posis1.append(posi1)
        posis2.append(posi2)
        R_means1.append(R_mean[0])
        R_means2.append(R_mean[1])
        TRIAL+=1
except KeyboardInterrupt:
    pass
posis1=np.array(posis1)
posis2=np.array(posis2)
print("____________________")
print("total time: ",time.time()-ttotal, "davon simulate(100): ", tsimulate, "davon movetoinit:", tmovetoinit)
figname="f"+str(minconi.var_f).replace(".","-")+"_"+"eta"+str(minconi.var_eta).replace(".","-")+"_g"+str(minconi.var_g).replace(".","-")+"_N"+str(minconi.var_N).replace(".","-")+"_A"+str(minconi.var_A).replace(".","-")
print("var_f,var_eta,var_g,var_N: ",figname)

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 20))
ax = plt.subplot(241)
#recordsA[timechunk]
ax.imshow(recordsA_first[0]['r'].T, aspect='auto', origin='lower')
ax.set_title('FIRST 100ms PopulationA of FIRST trial')
ax = plt.subplot(242)
ax.imshow(recordsA_first[-1]['r'].T, aspect='auto', origin='lower')
ax.set_title('LAST 100ms PopulationA FIRST trial')

ax = plt.subplot(245)
ax.imshow(recordsA[0]['r'].T, aspect='auto', origin='lower')
ax.set_title('FIRST 100ms PopulationA of last trial')
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

savefiginp="y"#input("save the figure? (y/n)")
if savefiginp=="y":
    plt.savefig("../bilder/delete/"+figname+"_Tr"+str(TRIAL))	
    print("saved")
else:
    print("not saved")
plt.show()
