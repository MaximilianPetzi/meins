use_feedback=True
import time
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

    
    
    # Recurrent population
    N = var_N
    pop = Population(N, neuron)
    pop[40].constant = 1.0
    pop[41].constant = 1.0
    pop[42].constant = -1.0
    pop.x = Uniform(-0.1, 0.1)
    
    if use_feedback:
        NN=10
        Nx=NN;Ny=NN;Nz=NN
        wis=2/(Nx+Ny+Nz)
        fdbx = Population(Nx, Neuron(parameters="r=0.0"))
        Wfx=Projection(fdbx, pop, 'in')
        Wfx.connect_all_to_all(weights=Uniform(-1.5*wis, 1.5*wis))

        fdby = Population(Ny, Neuron(parameters="r=0.0"))
        Wfy=Projection(fdby, pop, 'in')
        Wfy.connect_all_to_all(weights=Uniform(-1.5*wis, 1.5*wis))

        fdbz = Population(Nz, Neuron(parameters="r=0.0"))
        Wfz=Projection(fdbz, pop, 'in')
        Wfz.connect_all_to_all(weights=Uniform(-1.5*wis, 1.5*wis))

    
    # Input population
    inpx = Population(2, Neuron(parameters="r=0.0"))
    # Input weights'
    Wi = Projection(inp, pop, 'in')
    #setup(seed=5)
    Wi.connect_all_to_all(weights=Uniform(-1.5, 1.5))
    

    # Recurrent weights
    g = var_g #default 1.5
    Wrec = Projection(pop, pop, 'exc', synapse)
    #setup(seed=68)
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
    
    
