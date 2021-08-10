import numpy as np
Paramarr=np.load("../paramarr.npy")
#set flag to writable:
Paramarr[-1]=1
np.save('../paramarr',Paramarr)

var_f=Paramarr[0]
var_eta=Paramarr[1]
var_g=Paramarr[2]
var_N=int(Paramarr[3])
var_A=Paramarr[4] 
print("PARAMETERS MINCONI: ", Paramarr)

#var_f=float(input("f="))
#var_eta=float(input("eta="))
#var_g=float(input("g="))
#var_N=int(input("N="))
#var_A=float(input("A="))

import sys
#sys.path.append("/home/maxim/Documents/programme/CPG_iCub")
import time
from ANNarchy import *
clear()
setup(dt=1.0)

####

class net:
    def __init__(self, n_out):
        self.n_out=n_out
    begin_out=0
    neuron = Neuron(
        parameters = """
            tau = 30 : (default 30) population # Time constant
            constant = 0.0 # The four first neurons have constant rates
            alpha = 0.05 : population # To compute the sliding mean
            f = """+str(var_f)+""" : population # Frequency of the perturbation (war 3)
            A = """+str(var_A)+""" : (default 16) population # Perturbation amplitude. dt*A/tau should be 0.5...
        """,

        equations="""
            # Perturbation
            perturbation = if Uniform(0.0, 1.0) < f/1000.: 1.0 else: 0.0 
            noise = if perturbation > 0.5: A*Uniform(-1.0, 1.0) else: 0.0

            # ODE for x
            x += dt*(sum(in) + sum(exc) - x + noise)/tau

            # Output r
            rprev = r
            r = if constant == 0.0: tanh(x) else: tanh(constant)

            # Sliding mean
            delta_x = x - x_mean
            x_mean = alpha * x_mean + (1 - alpha) * x
        """
    )

    synapse = Synapse(
        parameters="""
            eta = """+str(var_eta)+""" : projection # Learning rate (vorher 0.5)
            learning_phase = 0.0 : projection # Flag to allow learning only at the end of a trial
            error = 0.0 : projection # Reward received
            mean_error = 0.0 : projection # Mean Reward received
            max_weight_change = 0.0005 : projection # Clip the weight changes
        """,
        equations="""
            # Trace
            trace += if learning_phase < 0.5:
                        power(pre.rprev * (post.delta_x), 3)
                    else:
                        0.0

            # Weight update only at the end of the trial
            delta_w = if learning_phase > 0.5:
                    eta * trace * (mean_error) * (error - mean_error)
                else:
                    0.0 : min=-max_weight_change, max=max_weight_change
            w -= if learning_phase > 0.5:
                    delta_w
                else:
                    0.0
        """
    )

    
    # Input population
    inp = Population(2, Neuron(parameters="r=0.0"))

    # Recurrent population
    N = var_N
    pop = Population(N, neuron)
    pop[40].constant = 1.0
    pop[41].constant = 1.0
    pop[42].constant = -1.0
    pop.x = Uniform(-0.1, 0.1)
    

    # Input weights
    Wi = Projection(inp, pop, 'in')
    Wi.connect_all_to_all(weights=Uniform(-1.0, 1.0))

    # Recurrent weights
    g = var_g #default 1.5
    Wrec = Projection(pop, pop, 'exc', synapse)
    Wrec.connect_all_to_all(weights=Normal(0., g/np.sqrt(N)), allow_self_connections=True)
    m = Monitor(pop, ['r'])
    compile()
    ####
    
    # Compute the mean reward per trial
    ##R_mean = np.zeros((2))
    ##alpha = 0.75 # 0.33

    
    def reinit(self):
        #setup(seed=1111)
        self.pop.x = Uniform(-0.1, 0.1).get_values(self.N)
        self.pop.r = np.tanh(self.pop.x)
        self.pop[1].r = np.tanh(1.0)
        self.pop[10].r = np.tanh(1.0)
        self.pop[11].r = np.tanh(-1.0)
        #print("(reinit)")
    
    

def dnms_trial(trial, first, second, R_mean):
    traces = []
    # Reinit network
    pop.x = Uniform(-0.1, 0.1).get_values(N)
    pop.r = np.tanh(pop.x)
    pop[1].r = np.tanh(1.0)
    pop[10].r = np.tanh(1.0)
    pop[11].r = np.tanh(-1.0)
    
    # First input
    inp[first].r = 1.0
    simulate(d_stim)
    # Delay
    inp.r = 0.0
    simulate(d_delay)
    # Second input
    inp[second].r = 1.0
    simulate(d_stim)
    # Relaxation
    inp.r = 0.0
    simulate(d_response)
    # Read the output
    rec = m.get()
    # traces = n.get('trace')
    # Compute the target
    target = 0.98 if first != second else -0.98
    # Response if over the last 200 ms
    output = rec['r'][-int(d_execution):, begin_out:begin_out+n_out] # neuron 100 over the last 200 ms
    # Compute the error
    error = np.mean(np.abs(target - output))
    print('Target:', target, '\tOutput:', "%0.3f" % np.mean(output), '\tError:',  "%0.3f" % error, '\tMean:', "%0.3f" % R_mean[first])
    # The first 25 trial do not learn, to let R_mean get realistic values
    if trial > 25:
        # Apply the learning rule
        Wrec.learning_phase = 1.0
        Wrec.error = error
        Wrec.mean_error = R_mean[first]
        # Learn for one step
        step()
        # Reset the traces
        Wrec.learning_phase = 0.0
        Wrec.trace = 0.0
        _ = m.get() # to flush the recording of the last step

    # Update the mean reward
    R_mean[first, second] = alpha * R_mean[first, second] + (1.- alpha) * error

    return rec, traces, R_mean


    # Initial weights
#init_w = Wrec.w

# Many trials of each type
'''try:
    for trial in range(2500):
        print('Trial', trial)
        recordsAA, tracesAA, R_mean = dnms_trial (trial, 0, 0, R_mean)
        recordsAB, tracesAB, R_mean = dnms_trial (trial, 0, 1, R_mean)
        recordsBA, tracesBA, R_mean = dnms_trial (trial, 1, 0, R_mean)
        recordsBB, tracesBB, R_mean = dnms_trial (trial, 1, 1, R_mean)
        if trial == 0:
            initialAA = recordsAA['r']
            initialAB = recordsAB['r']
            initialBA = recordsBA['r']
            initialBB = recordsBB['r']
except KeyboardInterrupt:
    pass

# Final weights
final_w = Wrec.w


import matplotlib.pyplot as plt
plt.figure(figsize=(20, 20))
ax = plt.subplot(231)
ax.imshow(recordsAA['r'].T, aspect='auto', origin='lower')
ax.set_title('Population')
ax = plt.subplot(232)
ax.plot(np.mean(initialAA[:, output_neuron:output_neuron+1], axis=1), label='before')
ax.plot(np.mean(recordsAA['r'][:, output_neuron:output_neuron+1], axis=1), label='after')
ax.set_ylim((-1., 1.))
ax.legend()
ax.set_title('Output AA -1')
ax = plt.subplot(233)
ax.plot(np.mean(initialBA[:, output_neuron:output_neuron+1], axis=1), label='before')
ax.plot(np.mean(recordsBA['r'][:, output_neuron:output_neuron+1], axis=1), label='after')
ax.set_ylim((-1., 1.))
ax.legend()
ax.set_title('Output BA +1')
ax = plt.subplot(235)
ax.plot(np.mean(initialAB[:, output_neuron:output_neuron+1], axis=1), label='before')
ax.plot(np.mean(recordsAB['r'][:, output_neuron:output_neuron+1], axis=1), label='after')
ax.set_ylim((-1., 1.))
ax.set_title('Output AB +1')
ax = plt.subplot(236)
ax.plot(np.mean(initialBB[:, output_neuron:output_neuron+1], axis=1), label='before')
ax.plot(np.mean(recordsBB['r'][:, output_neuron:output_neuron+1], axis=1), label='after')
ax.set_ylim((-1., 1.))
ax.set_title('Output BB -1')
plt.show()



'''


