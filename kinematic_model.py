import numpy as np


def G(a, d, alph, theta):

    t1 = np.array([[np.cos(theta), - np.sin(theta), 0,  0],
                  [np.sin(theta), np.cos(theta), 0,  0], [0, 0, 1, 0], [0, 0, 0, 1]])
    bt = np.c_[np.identity(3), np.array([0, 0, d])]
    t2 = np.vstack([bt, [0, 0, 0, 1]])

    ct = np.c_[np.identity(3), np.array([a, 0, 0])]
    t3 = np.vstack([ct, [0, 0, 0, 1]])

    t4 = np.array([[1, 0, 0, 0], [0, np.cos(alph), -np.sin(alph), 0],
                  [0, np.sin(alph), np.cos(alph), 0], [0, 0, 0, 1]])

    return np.matmul(np.matmul(np.matmul(t1, t2), t3), t4)


def wrist_position(a):
    L1 = 0.05
    L2 = 0.22
    L3 = 0.16
    theta2 = a[0]  # np.radians(a1)#Shoulder yaw
    theta1 = a[1]  # np.radians(a2)#Shoulder pitch
    theta3 = a[2]  # np.radians(a3)#Shoulder roll
    theta4 = a[3]  # np.radians(a4)#Elbow
    G_34 = G(L3, 0, 0, np.pi/2+theta4)
    G_23 = G(0, L2, np.pi/2, theta3+np.pi/2)
    G_12 = G(L1, 0, -np.pi/2, theta2+np.pi/2)
    G_01 = G(0, 0, -np.pi/2, theta1)

    G_02 = np.matmul(G_01, G_12)
    G_03 = np.matmul(G_02, G_23)
    G_04 = np.matmul(G_03, G_34)
    return G_04.dot(np.array([0, 0, 0, 1]).T)


"""
Usage example:

angles = np.zeros(params.number_cpg)
angles[iCubMotor.LShoulderPitch] = 40
angles[iCubMotor.LElbow] = -10
angles = np.radians(angles)

joint1 = iCubMotor.LShoulderRoll
joint2 = iCubMotor.LElbow
joint3 = iCubMotor.LShoulderPitch
joint4 = iCubMotor.LShoulderYaw
joints = [joint4,joint3,joint1,joint2]
AllJointList = joints


initial_position = wrist_position(angles[joints])[0:3]

#The function gives you the wrist position given the angles of the 4 joint in the arm
#Be careful with the order in which you give the angles, it matters
#it should always be like in the example

"""
'''
#joint1 = 20
#joint2 = 50
#joint3 = -25
#joint4 = 0

#orte testen:
#alte initpos (1joint):
joint1 = 00
joint2 = 65
joint3 = -80
joint4 = 80

joint1 = 140 #[0,160]
joint2 = 65 #[15,106]
joint3 = -20 #[-95.5,8]
joint4 = 40 #[-32,80]
joints = [joint4,joint3,joint1,joint2]


prevdist=0
for i in range(20000):
    
    [joint11,joint21,joint31,joint41]=np.random.rand(4)
    joint11*=160
    joint21=joint21*(106-15)+15
    joint31=joint31*(8+95.5)-95.5
    joint41=joint41*(80+32)-32
    joints = [joint41,joint31,joint11,joint21]
    joints= np.radians(joints)
    AllJointList = joints
    pos1 = wrist_position(AllJointList)[0:3]

    [joint12,joint22,joint32,joint42]=np.random.rand(4)
    joint12*=160
    joint22=joint21#joint22*(106-15)+15
    joint32=joint32*(8+95.5)-95.5
    joint42=joint41#joint42*(80+32)-32
    joints = [joint42,joint32,joint12,joint22]
    joints= np.radians(joints)
    AllJointList = joints
    pos2 = wrist_position(AllJointList)[0:3]

    [joint13,joint23,joint33,joint43]=np.random.rand(4)
    joint13*=160
    joint23=joint21#joint23*(106-15)+15
    joint33=joint33*(8+95.5)-95.5
    joint43=joint41#joint43*(80+32)-32
    joints = [joint43,joint33,joint13,joint23]
    joints= np.radians(joints)
    AllJointList = joints
    pos3 = wrist_position(AllJointList)[0:3]


    dist12=np.linalg.norm(np.array(pos1) - np.array(pos2))
    dist23=np.linalg.norm(np.array(pos2) - np.array(pos3))
    dist31=np.linalg.norm(np.array(pos3) - np.array(pos1))

    newdist=dist12**.5+(1+dist23)**2+dist31**.5
    if newdist>prevdist:
        prevdist=newdist
        print(newdist)
    if newdist>3.8:
        break

print("end distance:",prevdist,"dist12,23,31=",dist12,dist23,dist31)
print("""order of angles:\njoint1 = iCubMotor.LShoulderRoll,
joint2 = iCubMotor.LElbow,
joint3 = iCubMotor.LShoulderPitch,
joint4 = iCubMotor.LShoulderYaw""")

print("pos1 and its angles:")
print(pos1)
print(joint11,joint21,joint31,joint41)
print("pos2")
print(pos2)
print(joint12,joint22,joint32,joint42)
print("pos3")
print(pos3)
print(joint13,joint23,joint33,joint43)
print("\n choose targeta and targetb as the furthes away from each other")
print("dist23 will be found to be the highest")

'''