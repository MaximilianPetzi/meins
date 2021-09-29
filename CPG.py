usekm=True
njoints=4
#+
from CPG_lib.MLMPCPG.SetTiming import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.MLMPCPG import *
import CPG_lib.parameter as params
#import CPG_lib.iCub_connect.iCub_connect as robot_connect
import numpy as np
from termcolor import colored
import time
import importlib
import sys

#sys.path.append("/scratch/petzi/remote/CPG_iCub")


sys.path.append('CPG_lib/MLMPCPG')
sys.path.append('CPG_lib/icubPlot')


######################################
# import CPG_lib.icubPlot.icubplot as plot_arm
iCubMotor = importlib.import_module(params.iCub_joint_names)
######################################


class cpg:
    global All_Command
    global All_Joints_Sensor
    global angles, myT, myCont
    if usekm==True:
        pass#global Angles
    
    LShoulderRoll = 26
    LElbow = 28
    LShoulderPitch = 25
    LShoulderYaw = 27

    All_Command = []
    All_Joints_Sensor = []

    RG_Layer_E = []
    RG_Layer_F = []
    PF_Layer_E = []
    PF_Layer_F = []
    MN_Layer_E = []
    MN_Layer_F = []

    #######################################################
    myT = fSetTiming()

    # Create list of CPG objects
    myCont = fnewMLMPcpg(params.number_cpg)

    # Instantiate the CPG list with iCub robot data
    myCont = fSetCPGNet(myCont, params.my_iCub_limits,
                        params.positive_angle_dir)
    #######################################################

    # Setup iCub control devices
    if usekm==False:
        iCub_robot = robot_connect.iCub(params.used_parts)

    # Print the CPG paprameters for a joint
    # myCont[joint1].fPrint()
    # myCont[joint2].fPrint()

    RG_Layer_E = []
    RG_Layer_F = []
    PF_Layer_E = []
    PF_Layer_F = []
    MN_Layer_E = []
    MN_Layer_F = []

    # Read the robot angles
    if usekm==False:
        angles = iCub_robot.iCub_get_angles()
    # print(np.array(angles)*360/2/np.pi)
    # time.sleep(10)
    if usekm==True:
        angles = np.zeros(params.number_cpg)

    def set_patterns(self, alpha, theta, Sf, Ss, InjCMF, TM):  # vllt nur set_patterns???
        # Initiate PF and RG patterns for the joints
        if usekm==False:
            self.joint1 = iCubMotor.LShoulderRoll
            self.joint2 = iCubMotor.LShoulderPitch
            self.joint3 = iCubMotor.LShoulderYaw
            self.joint4 = iCubMotor.LElbow
        if usekm==True:
            self.joint1 = self.LShoulderRoll
            self.joint2 = self.LShoulderPitch
            self.joint3 = self.LShoulderYaw
            self.joint4 = self.LElbow
        
        
        Pattern1 = RG_Patterns(Sf[0], Ss[0], InjCMF[0], TM[0])
        PFPat1 = PF_Patterns(alpha[0], theta[0])

        myCont[self.joint1].fSetPatternPF(PFPat1)
        myCont[self.joint1].fSetPatternRG(Pattern1)
        
        if njoints>1:
            Pattern2 = RG_Patterns(Sf[1], Ss[1], InjCMF[1], TM[1])
            PFPat2 = PF_Patterns(alpha[1], theta[1])

            myCont[self.joint2].fSetPatternPF(PFPat2)
            myCont[self.joint2].fSetPatternRG(Pattern2)

        if njoints>2:
            Pattern3 = RG_Patterns(Sf[2], Ss[2], InjCMF[2], TM[2])
            PFPat3 = PF_Patterns(alpha[2], theta[2])

            myCont[self.joint3].fSetPatternPF(PFPat3)
            myCont[self.joint3].fSetPatternRG(Pattern3)

        if njoints>3:
            Pattern4 = RG_Patterns(Sf[3], Ss[3], InjCMF[3], TM[3])
            PFPat4 = PF_Patterns(alpha[3], theta[3])

            myCont[self.joint4].fSetPatternPF(PFPat4)
            myCont[self.joint4].fSetPatternRG(Pattern4)
        #######################################################

    def init_updates(self):
        # Update CPG initial position (reference position)
        if usekm==False:
            angles = self.iCub_robot.iCub_get_angles()
        if usekm==True:
            angles = self.Angles   
        for i in range(0, len(myCont)):
            myCont[i].fUpdateInitPos(angles[i])
        

        #############################
        # Update all joints CPG, it is important to update all joints
        # at least one time, otherwise, non used joints will be set to
        # the default init position in the CPG which is 0
        for i in range(0, len(myCont)):
            myCont[i].fUpdateLocomotionNetwork(myT, angles[i])
    
    def move_to_init(self):
        # move to init position at begining of each trial
        angles = np.zeros(params.number_cpg)
        lsp=-20;lsr=60;lsy=80;leb=65
        angles[iCubMotor.LShoulderPitch] = lsp
        angles[iCubMotor.LShoulderRoll] = lsr
        angles[iCubMotor.LShoulderYaw] = lsy
        angles[iCubMotor.LElbow] = leb
        angles[iCubMotor.LWristProsup] = -67.
        angles[iCubMotor.LWristPitch] = 0.
        angles[iCubMotor.LWristYaw] = 0.
        angles[iCubMotor.LHandFinger] = 60.
        angles[iCubMotor.LThumbOppose] = 80.
        angles[iCubMotor.LThumbProximal] = 15.
        angles[iCubMotor.LThumbDistal] = 30.
        angles[iCubMotor.LIndexProximal] = 30.
        angles[iCubMotor.LIndexDistal] = 45.
        angles[iCubMotor.LMiddleProximal] = 30.
        angles[iCubMotor.LMiddleDistal] = 45.
        angles[iCubMotor.LPinky] = 90.

        # convert angles to radians
        angles = np.radians(angles)
        self.init_angles = np.copy(angles)
        #print("\nPosition Command:")
        #print(myCont[joint1].description + ": ", np.round(np.degrees(angles[joint1]), 2))
        #print(myCont[joint2].description + ": ", np.round(np.degrees(angles[joint2]), 2))

        # Move robot to initial position
        # print("___before_set___")
        self.iCub_robot.iCub_set_angles(angles)
        # print("___after_set___")

    def move_to_init2(self,randinit,initseed):
        if randinit:
            #for random init positions:
            np.random.seed(initseed)
            randangles=np.random.rand(4)
            [joint11,joint21,joint31,joint41]=randangles
            
            joint11*=160
            joint21=joint21*(106-15)+15
            joint31=joint31*(8+95.5)-95.5
            joint41=joint41*(80+32)-32
            (lsp,lsr,lsy,leb)=(joint31,joint11,joint41,joint21)
        else:
            #for one init position only
            lsp=-7.6931;lsr=15.3936;lsy=13.1915;leb=38.7954
        # move to init position at begining of each trial
        #15.393610227562071 38.795440830509705 -7.693182229450187 13.191547981283598
        self.Angles = np.zeros(params.number_cpg)
        self.Angles[self.LShoulderRoll] = lsr#60.#90
        self.Angles[self.LElbow] = leb#65.
        self.Angles[self.LShoulderPitch] = lsp #-20.#-80
        self.Angles[self.LShoulderYaw] = lsy#80.

        # convert angles to radians
        self.Angles = np.radians(self.Angles)
        self.init_angles = np.copy(self.Angles)
    trajectori=[]
    def loop_move(self, timechunk):
        self.trajectori=[]
        #############################
        # MAIN LOOP
        I = 0

        time1 = time.time()
        tt = time1

        #myT.T1 = 1.0
        #myT.T2 = myT.T1 + myT.signal_pulse_width

        #myT.T3 = 1.0
        # myT.T4 = myT.T3 + myT.signal_pulse_width#signal_p_w..==0.2

        MAT_Iinj = []
        # for I in range(0,int(myT.N_Loop/2)):

        Imax = 5
        while I < Imax:  # (time.time() - tt) < maxt:
            I += 1
            t = timechunk*Imax*myT.T + I * myT.T  # myT.T==0.05
            # if I % 1 == 0:
            #    print(colored("I="+str(I),"blue"))
            ExtInjCurr = 1
            ExtInjCurr2 = 1
            ExtInjCurr3 = 1
            ExtInjCurr4 = 1
            #remove input after first 100ms
            #if I==2:# or timechunk!=0:   
            #    ExtInjCurr = 0
            #    ExtInjCurr2 = 0


            #print(ExtInjCurr)
            # if t >= myT.T1 and t < myT.T2:
            #    ExtInjCurr = 1
            # else:
            #    ExtInjCurr = 0

            # if t >= myT.T3 and t < myT.T4:
            #    ExtInjCurr2 = 1
            # else:
            #    ExtInjCurr2 = 0

            MAT_Iinj.append(ExtInjCurr)

            JointList1 = [self.joint1]
            for ii in JointList1:
                myCont[ii].RG.F.InjCurrent_value = +1*ExtInjCurr * \
                    myCont[ii].RG.F.InjCurrent_MultiplicationFactor
                myCont[ii].RG.E.InjCurrent_value = -1*ExtInjCurr * \
                    myCont[ii].RG.E.InjCurrent_MultiplicationFactor
            
            
            JointList2 = [self.joint2]
            if njoints>1:
                for ii in JointList2:
                    myCont[ii].RG.F.InjCurrent_value = +1*ExtInjCurr2 * \
                        myCont[ii].RG.F.InjCurrent_MultiplicationFactor
                    myCont[ii].RG.E.InjCurrent_value = -1*ExtInjCurr2 * \
                        myCont[ii].RG.E.InjCurrent_MultiplicationFactor
            
            
            JointList3 = [self.joint3]
            if njoints>2:    
                for ii in JointList3:
                    myCont[ii].RG.F.InjCurrent_value = +1*ExtInjCurr3 * \
                        myCont[ii].RG.F.InjCurrent_MultiplicationFactor
                    myCont[ii].RG.E.InjCurrent_value = -1*ExtInjCurr3 * \
                        myCont[ii].RG.E.InjCurrent_MultiplicationFactor

            
            JointList4 = [self.joint4]
            if njoints>3:
                for ii in JointList4:
                    myCont[ii].RG.F.InjCurrent_value = +1*ExtInjCurr4 * \
                        myCont[ii].RG.F.InjCurrent_MultiplicationFactor
                    myCont[ii].RG.E.InjCurrent_value = -1*ExtInjCurr4 * \
                        myCont[ii].RG.E.InjCurrent_MultiplicationFactor


            if usekm==False:
                angles = self.iCub_robot.iCub_get_angles()
            if usekm==True:
                angles=self.Angles
            self.AllJointList = JointList1 + JointList2 + JointList3 + JointList4

            # Update sensor neurons first before update the CPG
            ic = 0
            for i in self.AllJointList:
                ic += 1
                if ic <=njoints:  # NUR JOINT1 und joint2
                    myCont[i].fUpdateLocomotionNetwork(myT, angles[i])
            RG_Layer_E_tmp = []
            RG_Layer_F_tmp = []
            PF_Layer_E_tmp = []
            PF_Layer_F_tmp = []
            MN_Layer_E_tmp = []
            MN_Layer_F_tmp = []

            for idx, controller in enumerate(myCont):
                iCubMotor.MotorCommand[idx] = controller.joint.joint_motor_signal
                RG_Layer_E_tmp.append(controller.RG.E.out)
                RG_Layer_F_tmp.append(controller.RG.F.out)
                PF_Layer_E_tmp.append(controller.PF.E.o)
                PF_Layer_F_tmp.append(controller.PF.F.o)
                MN_Layer_E_tmp.append(controller.MN.E.o)
                MN_Layer_F_tmp.append(controller.MN.F.o)
            #print("wait now-...")
            #print("initial angles:\n",self.init_angles*360/2/np.pi)
            # time.sleep(5)
            #angl = self.iCub_robot.iCub_get_angles()
            #print("angles before setting",np.array(angl)*360/2/np.pi)
            # tspre=time.time()
            if usekm==False:
                self.iCub_robot.iCub_set_angles(iCubMotor.MotorCommand)
            if usekm==True:
                self.Angles = iCubMotor.MotorCommand
            # print(time.time()-tspre)
            self.RG_Layer_E.append(RG_Layer_E_tmp)
            self.RG_Layer_F.append(RG_Layer_F_tmp)
            self.PF_Layer_E.append(PF_Layer_E_tmp)
            self.PF_Layer_F.append(PF_Layer_F_tmp)
            self.MN_Layer_E.append(MN_Layer_E_tmp)
            self.MN_Layer_F.append(MN_Layer_F_tmp)

            # Update all commands and all sensors
            All_Command.append(iCubMotor.MotorCommand[:])
            All_Joints_Sensor.append(angles)

            self.trajectori.append(np.array(self.Angles)[[26,28,25,27]])#26,28,25,27 r b p y
            ######################################

            time2 = time.time()
            #print(t, time2-time1)
            while (time2 - time1) < .00:
                time2 = time.time()
            #print('time diff: ', time2 - time1)
            time1 = time2

        #print("\nFinished loop.. ")

    def plot_layers(self):
        # delete robot:
        #del self.iCub_robot
        # self.iCub_robot=None
        joint_names = []
        for i in self.AllJointList:
            joint_names.append(myCont[i].description)
        # plot all Layer Activities
        fPlotCPG_Layer((self.RG_Layer_E, self.RG_Layer_F), (self.PF_Layer_E, self.PF_Layer_F), (
            self.MN_Layer_E, self.MN_Layer_F), (All_Command, All_Joints_Sensor), self.AllJointList, joint_names)

        # plot motor patterns of used joints

        # for i in self.AllJointList:
        #      fPlotJointCommandSensor(All_Command, All_Joints_Sensor, i, myCont[i].description)

        # plot trajectory in workspace
        # ArmTrajX,ArmTrajY = plot_arm.fPlot2DArm(All_Joints_Sensor, 0,'arm_off', "left") # Drawing the arm :arm_off / arm_on

        # try:
        #    np.savetxt('All_Command.out', All_Command)
        #    print("=====================")
        #    print("All_Command has been saved to: \"All_Command.out\" ")
        #    print("To import it use: \"np.loadtxt('All_Command.out\')\"")
        #    print("=====================")
        #
        # except:
        #    print("All_Command CANNOT BE SAVED to: \"All_Command.out\"!!! ")
        #
        #
        # try:
        #    np.savetxt('All_Joints_Sensor.out', All_Joints_Sensor)
        #    print("=====================")
        #    print("All_Joints_Sensor has been saved to: \"All_Joints_Sensor.out\" ")
        #    print("To import it use: \"np.loadtxt('All_Joints_Sensor.out\')\"")
        #    print("=====================")
        #
        # except:
        #    print("All_Joints_Sensor CANNOT BE SAVED to: \"All_Joints_Sensor.out\"!!! ")
