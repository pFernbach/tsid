import pinocchio as se3
import tsid
import numpy as np
from numpy.linalg import norm as norm
import os

import gepetto.corbaserver
import time
import commands

from configs.talos_nav import *
import numpy as np
import timeopt
from locomote import WrenchCone,SOC6,ControlType,IntegratorType,ContactPatch, ContactPhaseHumanoid, ContactSequenceHumanoid

from utils import trajectories as traj

def SE3toVec(M):
    v = np.matrix(np.zeros((12, 1)))
    for j in range(3):
        v[j] = M.translation[j]
        v[j + 3] = M.rotation[j, 0]
        v[j + 6] = M.rotation[j, 1]
        v[j + 9] = M.rotation[j, 2]
    return v

def MotiontoVec(M):
    v = np.matrix(np.zeros((6, 1)))
    for j in range(3):
        v[j] = M.linear[j]
        v[j + 3] = M.angular[j]
    return v

np.set_printoptions(precision=3, linewidth=200)

print "".center(100, '#')
print " Test Task Space Inverse Dynamics ".center(100, '#')
print "".center(100, '#'), '\n'

lxp = 0.18  # foot length in positive x direction
lxn = 0.097  # foot length in negative x direction
lyp = 0.089  # foot length in positive y direction
lyn = 0.089  # foot length in negative y direction
lz = 0.0 #105  # foot sole height with respect to ankle joint
mu = 0.3  # friction coefficient
fMin = 5.0  # minimum normal force
fMax = 1000.0  # maximum normal force
rf_frame_name = "leg_right_6_joint"  # right foot frame name
lf_frame_name = "leg_left_6_joint"  # left foot frame name
contactNormal = np.matrix([0., 0., 1.]).T  # direction of the normal to the contact surface
w_com = 1.0                     # weight of center of mass task
w_posture = 1.0                # weight of joint posture task
w_forceRef = 1e-5               # weight of force regularization task
w_RF = 1.0                      # weight of right foot motion task
w_LF = 1.0  # weight of left foot motion task
kp_contact = 30.0               # proportional gain of contact constraint
kp_com = 3000.0                   # proportional gain of center of mass task
kp_posture = 30.0               # proportional gain of joint posture task
kp_RF = 3000.0                    # proportional gain of right foot motion task
kp_LF = 3000.0  # proportional gain of left foot motion task
REMOVE_CONTACT_N = 100  # remove right foot contact constraint after REMOVE_CONTACT_N time steps
CONTACT_TRANSITION_TIME = 1.0  # duration of the contact transition (to smoothly get a zero contact force before removing a contact constraint)
DELTA_COM_Y = 0.1  # distance between initial and desired center of mass position in y direction
DELTA_FOOT_Z = 0.1  # desired elevation of right foot in z direction
dt = 0.001  # controller time step
PRINT_N = 500  # print every PRINT_N time steps
DISPLAY_N = 25  # update robot configuration in viwewer every DISPLAY_N time steps
N_SIMULATION = 4000  # number of time steps simulated

filename = str(os.path.dirname(os.path.abspath(__file__)))
path = filename + '/../models/talos_data'
urdf = path + '/urdf/talos_reduced.urdf'
vector = se3.StdVec_StdString()
vector.extend(item for item in path)
robot = tsid.RobotWrapper(urdf, vector, se3.JointModelFreeFlyer(), False)
srdf = path + '/srdf/talos.srdf'

# for gepetto viewer .. but Fix me!!
robot_display = se3.RobotWrapper(urdf, [path, ], se3.JointModelFreeFlyer())

l = commands.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
if int(l[1]) == 0:
    os.system('gepetto-gui &')
time.sleep(1)
cl = gepetto.corbaserver.Client()
gui = cl.gui
robot_display.initDisplay(loadModel=True)

#load Contact Sequence and Trajectory from time-optimization
CONTACT_SEQUENCE_XML_TAG = "ContactSequence"
cs = ContactSequenceHumanoid(0)
cs.loadFromXML(filename + "/data/contact_sequence.xml", CONTACT_SEQUENCE_XML_TAG)
cs_result = ContactSequenceHumanoid(0)
cs_result.loadFromXML(filename + "/data/Result.xml", CONTACT_SEQUENCE_XML_TAG)

q = cs.contact_phases[0].reference_configurations[0].copy()
v = np.matrix(np.zeros(robot.nv)).transpose()

robot_display.displayCollisions(False)
robot_display.displayVisuals(True)
robot_display.display(q)
viewer = robot_display.viewer

gui = viewer.gui
path = path + '/urdf/plateforme_surfaces.urdf'
import rospkg
rospack = rospkg.RosPack()
packagePath = rospack.get_path ("talos_data")
packagePath += '/urdf/' + 'plateforme_surfaces.urdf'

gui.addUrdfObjects("planning",  packagePath, "", False)
gui.addToGroup ("planning", "world")
gui.refresh()

assert robot.model().existFrame(rf_frame_name)
assert robot.model().existFrame(lf_frame_name)

t = 0.0  # time
invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
invdyn.computeProblemData(t, q, v)
data = invdyn.data()
contact_Point = np.matrix(np.ones((3, 4)) * lz)
contact_Point[0, :] = [-lxn, -lxn, lxp, lxp]
contact_Point[1, :] = [-lyn, lyp, -lyn, lyp]

contactRF = tsid.Contact6d("contact_rfoot", robot, rf_frame_name, contact_Point, contactNormal, mu, fMin, fMax,
                           w_forceRef)
contactRF.setKp(kp_contact * np.matrix(np.ones(6)).transpose())
contactRF.setKd(2.0 * np.sqrt(kp_contact) * np.matrix(np.ones(6)).transpose())
H_rf_ref = robot.position(data, robot.model().getJointId(rf_frame_name))
contactRF.setReference(H_rf_ref)

invdyn.addRigidContact(contactRF)

contactLF = tsid.Contact6d("contact_lfoot", robot, lf_frame_name, contact_Point, contactNormal, mu, fMin, fMax,
                           w_forceRef)
contactLF.setKp(kp_contact * np.matrix(np.ones(6)).transpose())
contactLF.setKd(2.0 * np.sqrt(kp_contact) * np.matrix(np.ones(6)).transpose())
H_lf_ref = robot.position(data, robot.model().getJointId(lf_frame_name))
contactLF.setReference(H_lf_ref)
invdyn.addRigidContact(contactLF)

comTask = tsid.TaskComEquality("task-com", robot)
comTask.setKp(kp_com * np.matrix(np.ones(3)).transpose())
comTask.setKd(2.0 * np.sqrt(kp_com) * np.matrix(np.ones(3)).transpose())
invdyn.addMotionTask(comTask, w_com, 1, 0.0)

postureTask = tsid.TaskPosture("task-posture", robot)
postureTask.setKp(kp_posture * np.matrix(np.ones(robot.nv)).transpose())

postureTask.setKd(2.0 * np.sqrt(kp_posture) * np.matrix(np.ones(robot.nv)).transpose())
masks = np.matrix(np.zeros(robot.nv)).T
masks[3:6] += 1
masks[18:] += 1

postureTask.mask(masks)
invdyn.addMotionTask(postureTask, w_posture, 1, 0.0)

rightFootTask = tsid.TaskSE3Equality("task-right-foot", robot, rf_frame_name)
rightFootTask.setKp(kp_RF * np.matrix(np.ones(6)).transpose())
rightFootTask.setKd(2.0 * np.sqrt(kp_RF) * np.matrix(np.ones(6)).transpose())
rightFootTraj = tsid.TrajectorySE3Constant("traj-right-foot", H_rf_ref)

leftFootTask = tsid.TaskSE3Equality("task-left-foot", robot, lf_frame_name)
leftFootTask.setKp(kp_RF * np.matrix(np.ones(6)).transpose())
leftFootTask.setKd(2.0 * np.sqrt(kp_RF) * np.matrix(np.ones(6)).transpose())
leftFootTraj = tsid.TrajectorySE3Constant("traj-left-foot", H_lf_ref)

com_ref = robot.com(data)
trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)

q_ref = q
trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)

solver = tsid.SolverHQuadProg("qp solver")
solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)

# time check
control_time = cs_result.contact_phases[7].time_trajectory[len(cs_result.contact_phases[7].time_trajectory)-1]
N_SIMULATION = int(control_time / dt)
state = 0
state_phase = []
contact_break = False
state_change = True
com_desired =[]
foot_traj_RF =[]
foot_traj_LF =[]

for i in range(0, N_SIMULATION):
    time_start = time.time()
    phase = cs_result.contact_phases[state]
    phase_next = cs_result.contact_phases[state + 1]
    if i * dt > phase.time_trajectory[len(phase.time_trajectory)-1]:
        state = state + 1
        state_change = True
        print "state change"

    phase = cs_result.contact_phases[state]
    phase_next = cs_result.contact_phases[state+1]
    if phase.RF_patch.active and phase.LF_patch.active and state_change:
        invdyn.removeTask("task-right-foot", 0.0)
        invdyn.removeTask("task-left-foot", 0.0)

        if state_phase == "SSP(LF)":
            contactRF = tsid.Contact6d("contact_rfoot", robot, rf_frame_name, contact_Point, contactNormal, mu, fMin,
                                       fMax,
                                       w_forceRef)
            contactRF.setKp(kp_contact * np.matrix(np.ones(6)).transpose())
            contactRF.setKd(2.0 * np.sqrt(kp_contact) * np.matrix(np.ones(6)).transpose())
            data = invdyn.data()
            H_rf_ref = robot.position(data, robot.model().getJointId(rf_frame_name))
            H_rf_ref = phase_next.RF_patch.placement
	    contactRF.setReference(H_rf_ref)
            invdyn.addRigidContact(contactRF)

        elif state_phase == "SSP(RF)":
            contactLF = tsid.Contact6d("contact_lfoot", robot, lf_frame_name, contact_Point, contactNormal, mu, fMin,
                                       fMax,
                                       w_forceRef)
            contactLF.setKp(kp_contact * np.matrix(np.ones(6)).transpose())
            contactLF.setKd(2.0 * np.sqrt(kp_contact) * np.matrix(np.ones(6)).transpose())
            data = invdyn.data()
            H_lf_ref = robot.position(data, robot.model().getJointId(lf_frame_name))
            H_lf_ref = phase_next.LF_patch.placement
            contactLF.setReference(H_lf_ref)
            invdyn.addRigidContact(contactLF)

        if i==0:
            transition_time = phase.time_trajectory[len(phase.time_trajectory) - 1]
        else:
            transition_time = phase.time_trajectory[len(phase.time_trajectory) - 1] - cs_result.contact_phases[state-1].time_trajectory[len(cs_result.contact_phases[state-1].time_trajectory) - 1]

        if phase_next.RF_patch.active:
            print "\nTime %.3f Start breaking contact %s\n" % (t, contactLF.name)
            invdyn.removeRigidContact(contactLF.name, transition_time)
        elif phase_next.LF_patch.active:
            print "\nTime %.3f Start breaking contact %s\n" % (t, contactRF.name)
            invdyn.removeRigidContact(contactRF.name, transition_time)
        else:
            assert False
        state_phase = "DSP"

    elif cs_result.contact_phases[state].LF_patch.active and not cs_result.contact_phases[state].RF_patch.active and state_change:
        state_phase = "SSP(LF)"
        invdyn.addMotionTask(rightFootTask, w_RF, 1, 0.0)

        time_interval = []
        t0 = i * dt
        t1 = phase.time_trajectory[len(phase.time_trajectory)-1]

        time_interval.append(t0)
        time_interval.append(t1)

        foot_placement = []
        foot_init = robot.position(data, robot.model().getJointId(rf_frame_name))
        foot_final = phase_next.RF_patch.placement
        foot_placement.append(foot_init)
        foot_placement.append(foot_final)
        foot_traj_RF = traj.SmoothedFootTrajectory(time_interval, foot_placement)
    elif cs_result.contact_phases[state].RF_patch.active and not cs_result.contact_phases[state].LF_patch.active and state_change:
        state_phase = "SSP(RF)"
        invdyn.addMotionTask(leftFootTask, w_LF, 1, 0.0)

        time_interval = []
        t0 = i * dt
        t1 = phase.time_trajectory[len(phase.time_trajectory)-1]

        time_interval.append(t0)
        time_interval.append(t1)

        foot_placement = []
        foot_init = robot.position(data, robot.model().getJointId(lf_frame_name))
        foot_final = phase_next.LF_patch.placement
        foot_placement.append(foot_init)
        foot_placement.append(foot_final)
        foot_traj_LF = traj.SmoothedFootTrajectory(time_interval, foot_placement)


    if state_change:
        com_init = np.matrix(np.zeros((9, 1)))
        com_init[0:3, 0] = robot.com(invdyn.data())
        com_traj = traj.SmoothedCOMTrajectory("com_smoothing", cs_result.contact_phases[state], com_init, 0.001)
        state_change = False

    sampleCom = trajCom.computeNext()
    sampleCom.pos(com_traj(i * dt)[0].T)
    sampleCom.vel(com_traj(i * dt)[1].T)

    com_desired = com_traj(i * dt)[0].T

    comTask.setReference(sampleCom)
    samplePosture = trajPosture.computeNext()
    postureTask.setReference(samplePosture)

    if state_phase == "SSP(LF)":
        sampleRightFoot = rightFootTraj.computeNext()
        sampleRightFoot.pos(SE3toVec(foot_traj_RF(i * dt)[0]))
        sampleRightFoot.vel(MotiontoVec(foot_traj_RF(i * dt)[1]))
        rightFootTask.setReference(sampleRightFoot)
    elif state_phase == "SSP(RF)":
        sampleLeftFoot = leftFootTraj.computeNext()
        sampleLeftFoot.pos(SE3toVec(foot_traj_LF(i * dt)[0]))
        sampleLeftFoot.vel(MotiontoVec(foot_traj_LF(i * dt)[1]))
        leftFootTask.setReference(sampleLeftFoot)

    HQPData = invdyn.computeProblemData(t, q, v)
    if i == 0:
        HQPData.print_all()

    sol = solver.solve(HQPData)
    tau = invdyn.getActuatorForces(sol)
    dv = invdyn.getAccelerations(sol)

    if i % PRINT_N == 0:
        print "Time %.3f" % (t)

        if invdyn.checkContact(contactRF.name, sol):
            f = invdyn.getContactForce(contactRF.name, sol)
            print "\tnormal force %s: %.1f" % (contactRF.name.ljust(20, '.'), contactRF.getNormalForce(f))

        if invdyn.checkContact(contactLF.name, sol):
            f = invdyn.getContactForce(contactLF.name, sol)
            print "\tnormal force %s: %.1f" % (contactLF.name.ljust(20, '.'), contactLF.getNormalForce(f))
        
        print "\ttracking err %s: %.3f" % (comTask.name.ljust(20, '.'), norm(comTask.position_error, 2))
        print "\ttracking err %s: %.3f" % (rightFootTask.name.ljust(20, '.'), norm(rightFootTask.position_error, 2))
        print "\ttracking err %s: %.3f" % (leftFootTask.name.ljust(20, '.'), norm(leftFootTask.position_error, 2))
        print "\t||v||: %.3f\t ||dv||: %.3f" % (norm(v, 2), norm(dv))
        
    v_mean = v + 0.5 * dt * dv
    v += dt * dv
    q = se3.integrate(robot.model(), q, dt * v_mean)
    t += dt

    if i % DISPLAY_N == 0:
        robot_display.display(q)

    time_spent = time.time() - time_start
    if (time_spent < dt):
        time.sleep(dt - time_spent)

    assert norm(dv) < 1e6
    assert norm(v) < 1e6

print "\nFinal COM Position  ", robot.com(invdyn.data()).T
print "Desired COM Position", com_desired.T

