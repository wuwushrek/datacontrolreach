from jax.config import config
config.update("jax_enable_x64", True)

import jax

import numpy as np
import jax.numpy as jnp
import datacontrolreach.jumpy as jp

from datacontrolreach import interval as itvl
from datacontrolreach.interval import Interval as iv

from datacontrolreach import dynsideinfo
from datacontrolreach.dynsideinfo import DynamicsWithSideInfo

from datacontrolreach.utils import synthTraj

from tqdm.auto import tqdm

# Define a seed for reproductibility
np.random.seed(2339) # 3626 923 sexy traj: (3963 c_vmax =  1.0 c_wmax =  1.5 c_rot = 5)

# Sampling time -> Delta t
sampling_time = 0.1

# Define the initial state
initial_state = np.array([-2, -2.5, np.pi/2])

# Number of data point in random trajectory
n_data_max = 15

# Input bounds \mathcal{U}
v_max = 3
w_max = 0.5 * (2*np.pi)
v_min = -v_max
w_min = -w_max
input_lb = np.array([v_min, w_min])
input_ub = np.array([v_max, w_max])

# Generate random input sequence
v_seq = -1 *(np.random.rand(n_data_max,1) - 0) * v_max 
w_seq = 2 * (np.random.rand(n_data_max,1) - 0.5) * w_max

# We ensure sufficient excitations in all control directions to have a 'rich' trajectory
# Number of data points used while applying such excitations
nSep = int(n_data_max /2)

# Basically randomly pick data point in the trajectory with v = w  = 0, v=0 and w = random, w =0 and v = random 
sepIndexes = np.random.choice([ i for i in range(n_data_max)], nSep, replace=False)
for i in range(1,nSep):
    zero_or_no = np.random.choice([0,1], p=[0.2,0.8])
    if zero_or_no == 0:
        w_seq[sepIndexes[i],0] = 0
        v_seq[sepIndexes[i],0] = 0
        continue
    v_or_theta = np.random.choice([0,1], p=[0.4,0.6])
    if v_or_theta == 0: # pick v
        w_seq[sepIndexes[i],0] = 0
    else: # pick theta
        v_seq[sepIndexes[i],0] = 0

# Randomly generated control sequence.
rand_control = np.hstack((v_seq,w_seq))
print ('Random control applied at each time step: ')
print(rand_control)

# Input signal to evaluate a trajectory of the system
c_vmax =  1.0
c_wmax =  1.
c_rot = 6
t_end = (n_data_max-1) * sampling_time

# Build a function to output the corresponding control value at a given time t \in [0, 4]
# t \in [0, 1.5] corresponds to the trajectory T_15
@jax.jit
def uOver(t, pert=iv(0.)):
    # print(t)
    t  = t if isinstance(t, iv) else iv(t)
    uval = itvl.block_array([iv(c_vmax), jp.cos(c_rot * (t-t_end))* c_wmax])
    valid_indx = jnp.asarray(jnp.where(t.lb < t_end+(1.0*sampling_time)-1e-6, t.lb / sampling_time, 0), dtype=int)
    rcontrol = jnp.asarray(rand_control)
    uval = jp.where(t.lb < t_end+(1.0*sampling_time)-1e-6, iv(rcontrol[valid_indx]), uval+pert)
    return uval

@jax.jit
def uOverDer(t, pert=iv(0.)):
    t  = t if isinstance(t, iv) else iv(t)
    uval = itvl.block_array([iv(0.), -c_rot * jp.sin(c_rot * (t-t_end)) * c_wmax])
    uval = jp.where(t.lb < t_end+(1.0*sampling_time)-1e-6, itvl.zeros_like(uval), uval+pert)
    return uval


# Scalar based control signal for ode solving
_uOver =  lambda t : jnp.array(uOver(t).lb)

############################## System dynamics ###############################3
# The actual dynamics of the system --> Decomposed in known and unknown terms
def _known_terms(x, u):
    # The value of the control is used in the composed function and for contraction
    return {'u' : u}

### ALL RETURN DICTIONARY MUST HAVE THE SAME TYPE ####
def _composed_function(knTerm, unkTerm):
    # unkTerm are uncertain values and  knTerm is possibly uncertain
    zl = jp.zeros_like(unkTerm['G11'])
    G = jp.block_array([[unkTerm['G11'], zl],[unkTerm['G21'], zl], [zl, unkTerm['G32']] ])
    f = jp.block_array([unkTerm['f1'], unkTerm['f2'], unkTerm['f3']])
    return f + G @ knTerm['u']

def value_unknown(x):
    zl = jp.zeros_like(x[0])
    ol = jp.ones_like(x[1])
    # True value of the unknown term for ground true comparisons
    return {'G11' : jp.cos(x[2]), 'G21' : jp.sin(x[2]), 'G32' : ol, 'f1' : zl, 'f2' : zl, 'f3' : zl}

# The known dynamics of the system
@jax.jit
def m_dyn(x,u):
    return _composed_function(_known_terms(x, u), value_unknown(x))


# Generate a more precise trajectory with smaller sample time based on the control
# uFun by solving numerically the ODE with the true dynamics
nPoint = 81 # Precise the trajectory length (including T_15)
dt = 0.05 # Delta time of the computed trajectory

# Create the time tables
tVal_1 = np.array([ i*dt for i in range(nPoint+1)])
tMeas = np.array([i*sampling_time for i in range(n_data_max)])

# Obtain the trajectory via ode solver
tVal_1, traj, trajDot = synthTraj(m_dyn, _uOver, initial_state, tVal_1, atol=1e-10, rtol=1e-10)
tVal = tVal_1[:-1] # Obtain the list of time before the last time

# Create a table with the control values applied at each time step dt
u_values = np.empty((nPoint, rand_control.shape[1]), dtype=np.float64)
for i in range(u_values.shape[0]):
    u_values[i,:] = _uOver(tVal[i])

# Construct the state evolution and the state derivative evolution
x_values = traj[:,:]
xdot_values = trajDot[:-1,:]
print(x_values.shape, xdot_values.shape, u_values.shape)

# Generate the trajectory corresponding to random input sequence
rand_init_traj_vec = np.zeros((n_data_max + 1, initial_state.size), dtype=np.float64)
rand_init_traj_der_vec = np.zeros((n_data_max, initial_state.size), dtype=np.float64)
rand_init_input_vec = np.zeros((n_data_max, rand_control.shape[1]), dtype=np.float64)
repeat_val = int(sampling_time/dt)
for i in range(0, repeat_val*rand_control.shape[0], repeat_val):
    rand_init_input_vec[int(i/repeat_val),:] = u_values[i,:] 
    rand_init_traj_der_vec[int(i/repeat_val),:] = xdot_values[i,:]
    rand_init_traj_vec[int(i/repeat_val),:] = x_values[i,:]
rand_init_traj_vec[-1,:] = x_values[repeat_val*rand_control.shape[0],:]


import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

# Preview the trajectory
# plt.style.use('dark_background')
lwidth = 2
mSize = 5
markerTraj = 'bs'

# plt.figure()
# plt.plot(tVal, xdot_values[:,0], "green", linewidth=lwidth, label='$\dot{p}_x$')
# plt.plot(tMeas, rand_init_traj_der_vec[:,0], markerTraj, markersize=mSize, label=r"$\mathscr{T}_{"+ str(tMeas.shape[0])+r"}$")
# plt.xlabel('Time (s)')
# plt.ylabel('$\dot{p}_x$')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# # plt.show()

# plt.figure()
# plt.plot(tVal, xdot_values[:,1], "green", linewidth=lwidth, label='$\dot{p}_y$')
# plt.plot(tMeas, rand_init_traj_der_vec[:,1], markerTraj, markersize=mSize, label=r'$\mathscr{T}_{'+ str(tMeas.shape[0])+r'}$')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$\dot{p}_y$')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# # plt.show()

# plt.figure()
# plt.plot(tVal, xdot_values[:,2], "green", linewidth=lwidth, label=r'$\dot{\Theta}$')
# plt.plot(tMeas, rand_init_traj_der_vec[:,2], markerTraj, markersize=mSize, label=r'$\mathscr{T}_{'+ str(tMeas.shape[0])+r'}$')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$\dot{\Theta}$')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# # plt.show()

# plt.figure()
# plt.plot(x_values[:-1,0], x_values[:-1,1], 'tab:cyan', linewidth=lwidth, label='$\mathrm{p_x-p_y \ trajectory}$')
# plt.plot(rand_init_traj_vec[:-1,0], rand_init_traj_vec[:-1,1], markerTraj, markersize=mSize, label='$\mathscr{T}_{'+ str(tMeas.shape[0])+'}$')    
# plt.xlabel('$p_x$')
# plt.ylabel('$p_y$')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# # plt.show()

# plt.figure()
# plt.plot(tVal, x_values[:-1,0], "tab:cyan", linewidth=lwidth, label='$p_x$')
# plt.plot(tMeas, rand_init_traj_vec[:-1,0], markerTraj, markersize=mSize, label='$\mathscr{T}_{'+ str(tMeas.shape[0])+'}$')
# plt.xlabel('Time (s)')
# plt.ylabel('$p_x$')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# # plt.show()

# plt.figure()
# plt.plot(tVal, x_values[:-1,1], "tab:cyan", linewidth=lwidth, label='$p_y$')
# plt.plot(tMeas, rand_init_traj_vec[:-1,1], markerTraj, markersize=mSize, label='$\mathscr{T}_{'+ str(tMeas.shape[0])+'}$')
# plt.xlabel('Time (s)')
# plt.ylabel('$p_y$')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# # plt.show()

# plt.figure()
# plt.plot(tVal, x_values[:-1,2], "tab:cyan", linewidth=lwidth, label=r'$\Theta$')
# plt.plot(tMeas, rand_init_traj_vec[:-1,2], markerTraj, markersize=mSize, label='$\mathscr{T}_{'+ str(tMeas.shape[0])+'}$')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$\Theta$')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# # plt.show()

# plt.figure()
# plt.plot(tVal, u_values[:,0], "magenta", linewidth = lwidth, label='$\mathrm{Speed}$')
# plt.plot(tMeas, rand_init_input_vec[:,0], markerTraj, markersize=mSize, label='$\mathscr{T}_{'+ str(tMeas.shape[0])+'}$')
# plt.xlabel('Time (s)')
# plt.ylabel('$\mathrm{Speed}$ $(v)$')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# # plt.show()

# plt.figure()
# plt.plot(tVal, u_values[:,1], "magenta", linewidth = lwidth, label='$\mathrm{Turning\ rate}$')
# plt.plot(tMeas, rand_init_input_vec[:,1], markerTraj, markersize=mSize, label='$\mathscr{T}_{'+ str(tMeas.shape[0])+'}$')
# plt.xlabel('Time (s)')
# plt.ylabel('$\mathrm{Turning\ rate}$ $(\omega)$')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

######## Build the differential inclusion ##########
from datacontrolreach.dynsideinfo import DynamicsWithSideInfo, lipinfo_builder, hc4revise_lin_eq

from datacontrolreach import reach
from datacontrolreach.reach import build_overapprox, DaTaReach


lipinfo = {}
nS, nC = 3, 2
lipinfo['G11'] = lipinfo_builder(Lip=1.1, vDep=[2], weightLip=None, n=nS,
                    bound=(-10,10), gradBound=None)
lipinfo['G21'] = lipinfo_builder(Lip=1.1, vDep=[2], weightLip=None, n=nS,
                    bound=(-10,10), gradBound=None)
lipinfo['G32'] = lipinfo_builder(Lip=0.01, vDep=[2], weightLip=None, n=nS,
                    bound=(-10,10), gradBound=None)
lipinfo['f1'] = lipinfo_builder(Lip=0.01, vDep=[2], weightLip=None, n=nS,
                    bound=(-10,10), gradBound=None)
lipinfo['f2'] = lipinfo_builder(Lip=0.01, vDep=[2], weightLip=None, n=nS,
                    bound=(-10,10), gradBound=None)
lipinfo['f3'] = lipinfo_builder(Lip=0.01, vDep=[2], weightLip=None, n=nS,
                    bound=(-10,10), gradBound=None)


# return {'G11' : jp.cos(x[2]), 'G21' : jp.sin(x[2]), 'G32' : ol, 'f1' : zl, 'f2' : zl, 'f3' : zl}
@jax.tree_util.register_pytree_node_class
class Unicycle(DynamicsWithSideInfo):
    def __init__(self, *args):
        super().__init__(*args)

    @staticmethod
    def known_terms(x, u):
        return _known_terms(x, u)

    @staticmethod
    def composed_dynamics(kTerms, unkTerms):
        return _composed_function(kTerms, unkTerms)

    @staticmethod
    def contraction(unkTerms, kTerms, noisyXdot):
        # First axis constraint -> dot{x}
        res = {}
        uval = kTerms['u']
        x1_unk = hc4revise_lin_eq(noisyXdot[0], jp.array([1., uval[0]]), jp.block_array([unkTerms['f1'], unkTerms['G11']]))
        res['f1'], res['G11'] = x1_unk[0], x1_unk[1]
        x2_unk = hc4revise_lin_eq(noisyXdot[1], jp.array([1., uval[0]]), jp.block_array([unkTerms['f2'], unkTerms['G21']]))
        res['f2'], res['G21'] = x2_unk[0], x2_unk[1]
        x3_unk = hc4revise_lin_eq(noisyXdot[2], jp.array([1., uval[1]]), jp.block_array([unkTerms['f3'], unkTerms['G32']]))
        res['f3'], res['G32'] = x3_unk[0], x3_unk[1]
        return {key : res[key] for key in unkTerms}

feas_state_input = (rand_init_traj_vec[0], rand_init_input_vec[0])

print('BUILDING ARRAY')
jax_jit = jax.jit(build_overapprox, static_argnums=(0,6,))
# jax_jit = build_overapprox
sidyn = jax_jit(Unicycle, feas_state_input, lipinfo, xTraj = rand_init_traj_vec,
                xDotTraj = rand_init_traj_der_vec, uTraj = rand_init_input_vec, 
                max_data_size = 30)


# dt_reach = 0.02
# jax.jit(reach.apriori_enclosure, static_argnums=(6,7,))(sidyn, rand_init_traj_vec[-1], rand_init_input_vec[-1], dt_reach,
#                         fixpointWidenCoeff=0.2, zeroDiameter=1e-5, containTol=1e-2, maxFixpointIter=10)

# Compute the differential inclusion
diff_inclu, _ = jax.vmap(lambda x, u : dynsideinfo.dynamics(sidyn, x, u))(x_values[:u_values.shape[0]], u_values)
xdot_lb, xdot_ub = diff_inclu.lb, diff_inclu.ub
# print(res.__dict__)

# # First plot the state evolution
# plt.figure()
# plt.plot(tVal, xdot_values[:,0], "orange", linewidth=1, label='$f(x) + G(x) u$')
# plt.plot(tMeas, rand_init_traj_der_vec[:,0], markerTraj, markersize=5, label='$\mathscr{T}_{'+ str(tMeas.shape[0])+'}$')
# plt.fill_between(tVal, xdot_lb[:,0], xdot_ub[:,0], alpha=0.8, facecolor="green", edgecolor= "darkgreen",
#                  linewidth=1, label=r'$\mathrm{f}(x) + \mathrm{G}(x) u$')
# plt.xlabel('Time (s)')
# plt.ylabel('$\dot{p}_x$')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()

# plt.figure()
# plt.plot(tVal, xdot_values[:,1], "orange", linewidth = 1, label='$f(x) + G(x) u$')
# plt.plot(tMeas, rand_init_traj_der_vec[:,1], markerTraj, markersize=5, label='$\mathscr{T}_{'+ str(tMeas.shape[0])+'}$')
# plt.fill_between(tVal, xdot_lb[:,1], xdot_ub[:,1], alpha=0.8, facecolor="green", edgecolor= "darkgreen",
#                  linewidth=1, label=r'$\mathrm{f}(x) + \mathrm{G}(x) u$')
# plt.xlabel('Time (s)')
# plt.ylabel('$\dot{p}_y$')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# plt.figure()
# plt.plot(tVal, xdot_values[:,2], "orange", linewidth = 1, label='$f(x) + G(x) u$')
# plt.plot(tMeas, rand_init_traj_der_vec[:,2], markerTraj, markersize=5, label='$\mathscr{T}_{'+ str(tMeas.shape[0])+'}$')
# plt.fill_between(tVal, xdot_lb[:,2], xdot_ub[:,2], alpha=0.8, facecolor="green", edgecolor= "darkgreen", 
#                  linewidth=1, label=r'$\mathrm{f}(x) + \mathrm{G}(x) u$')
# plt.xlabel('Time (s)')
# plt.ylabel(r'$\dot{\Theta}$')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()

# plt.show()

# Define the pertubation set
pert_lb = np.array([-0.1,-0.01])
pert_ub = np.array([0.1, 0.01])

# Define the uncertain control signals
uOverPert =  uOver # lambda t : uOver(t, iv(pert_lb, pert_ub))
uOverDerPert =  uOverDer # lambda t : uOverDer(t)


# time step for for the reachable sets computation
dt_reach = 0.02
repeat_val = int(sampling_time/dt_reach)
nPoint = int((tVal[-1]) / dt_reach)
tValOver = np.array([ i*dt_reach for i in range(nPoint+1)])

# Discretize the pertubation
nTile = 100
v_tiling = np.linspace(pert_lb[0], pert_ub[0], nTile)
w_tiling = np.linspace(pert_lb[1], pert_ub[1], nTile)

def trueReachableSet(discreteTime, stateInit):
    """ Compute the true reachable set based on the Subdivion by nTile"""
    res_lb = np.zeros((discreteTime.shape[0], stateInit.shape[0]), dtype=float)
    res_ub = np.zeros((discreteTime.shape[0], stateInit.shape[0]), dtype=float)
    for i in tqdm(range(nTile)):
        # @jax.jit
        def uFun(t):
            return uOverPert(t,  pert=iv(jp.array([v_tiling[i], w_tiling[i]]))).lb
        _, traj, trajDot = synthTraj(m_dyn, uFun, stateInit, discreteTime, atol=1e-10, rtol=1e-10)
        # print(t)
        if i == 0:
            res_lb = traj
            res_ub = traj
            continue
        res_lb = np.minimum(res_lb, traj)
        res_ub = np.maximum(res_ub, traj)
    return res_lb, res_ub

reach_lb, reach_ub = trueReachableSet(np.array([ t_end+sampling_time+i*dt_reach for i in range(tValOver.shape[0]-repeat_val*rand_init_input_vec.shape[0])]), rand_init_traj_vec[-1,:])
exit

_x0 = rand_init_traj_vec[-1,:]
_t0 = t_end+sampling_time
_nPoint = tValOver.shape[0]-1-repeat_val*rand_init_input_vec.shape[0]
_uover = lambda t : uOver(t, iv(pert_lb, pert_ub))
_uder = uOverDer
import time

jit_datareach = jax.jit(DaTaReach, static_argnums=(3,5,6,10,))
_ = jit_datareach(sidyn, _x0, _t0, _nPoint, dt_reach, _uover, _uder, 
                fixpointWidenCoeff=0.2, zeroDiameter=1e-5, containTol=1e-3, maxFixpointIter=10)

c_t = time.time()
datareach_1, integ_d = jit_datareach(sidyn, _x0, _t0, _nPoint, dt_reach, _uover, _uder,
                fixpointWidenCoeff=0.2, zeroDiameter=1e-5, 
                containTol=1e-3, maxFixpointIter=10)
diff_t = time.time()-c_t
datareach_d_lb, datareach_d_ub = datareach_1.lb, datareach_1.ub
print('Compute Time = {}'.format(diff_t))



# Plot the over-approximation of the reachable set
# plt.style.use('dark_background')

# First plot the state evolution
plt.figure()
plt.plot(tMeas, rand_init_traj_vec[:-1,0], markerTraj, markersize=mSize, label='$\mathscr{T}_{'+ str(tMeas.shape[0])+'}$')
# plt.fill_between(tValOver, datareach_nd_lb[:,0], datareach_nd_ub[:,0], alpha=0.8, facecolor="coral", edgecolor= "darksalmon",
#                  linewidth=1, label='$\mathrm{DaTaReach,\ case\ (a)}$')
plt.fill_between(integ_d, datareach_d_lb[:,0], datareach_d_ub[:,0], alpha=0.8, facecolor="green", edgecolor= "darkgreen",
                 linewidth=1, label='$\mathrm{DaTaReach,\ case\ (b)}$')
# plt.fill_between(tValOver, datareach_dt_lb[:,0], datareach_dt_ub[:,0], alpha=0.8, facecolor="purple",edgecolor= "darkmagenta",
#                  linewidth=1, label='$\mathrm{DaTaReach,\ case\ (c)}$')
plt.fill_between(integ_d, reach_lb[:,0], reach_ub[:,0], alpha=0.8, facecolor="tab:cyan", edgecolor= "darkcyan",
                 linewidth=1, label='$\mathrm{Reachable\ set}$')
plt.xlabel('Time (s)')
plt.ylabel('$p_x$')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
# plt.show()

plt.figure()
plt.plot(tMeas, rand_init_traj_vec[:-1,1], markerTraj, markersize=5, label='$\mathscr{T}_{'+ str(tMeas.shape[0])+'}$')
# plt.fill_between(tValOver, datareach_nd_lb[:,1], datareach_nd_ub[:,1], alpha=0.8, facecolor="coral", edgecolor= "darksalmon",
#                  linewidth=1, label='$\mathrm{DaTaReach,\ case\ (a)}$')
plt.fill_between(integ_d, datareach_d_lb[:,1], datareach_d_ub[:,1], alpha=0.8, facecolor="green", edgecolor= "darkgreen",
                 linewidth=1, label='$\mathrm{DaTaReach,\ case\ (b)}$')
# plt.fill_between(tValOver, datareach_dt_lb[:,1], datareach_dt_ub[:,1], alpha=0.8, facecolor="purple",edgecolor= "darkmagenta",
#                  linewidth=1, label='$\mathrm{DaTaReach,\ case\ (c)}$')
plt.fill_between(integ_d, reach_lb[:,1], reach_ub[:,1], alpha=0.8, facecolor="tab:cyan", edgecolor= "darkcyan",
                 linewidth=1, label='$\mathrm{Reachable\ set}$')
plt.xlabel('Time (s)')
plt.ylabel('$p_y$')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
# plt.show()

plt.figure()
plt.plot(tMeas, rand_init_traj_vec[:-1,2], markerTraj, markersize=5, label='$\mathscr{T}_{'+ str(tMeas.shape[0])+'}$')
plt.plot(tVal_1, x_values[:,-1], label='true', color='red')
# plt.fill_between(tValOver, datareach_nd_lb[:,2], datareach_nd_ub[:,2], alpha=0.8, facecolor="coral", edgecolor= "darksalmon",
#                  linewidth=1, label='$\mathrm{DaTaReach,\ case\ (a)}$')
plt.fill_between(integ_d, datareach_d_lb[:,2], datareach_d_ub[:,2], alpha=0.8, facecolor="green", edgecolor= "darkgreen",
                 linewidth=1, label='$\mathrm{DaTaReach,\ case\ (b)}$')
# plt.fill_between(tValOver, datareach_dt_lb[:,2], datareach_dt_ub[:,2], alpha=0.8, facecolor="purple",edgecolor= "darkmagenta",
#                  linewidth=1, label='$\mathrm{DaTaReach,\ case\ (c)}$')
plt.fill_between(integ_d, reach_lb[:,2], reach_ub[:,2], alpha=0.8, facecolor="tab:cyan", edgecolor= "darkcyan", 
                 linewidth=1, label='$\mathrm{Reachable\ set}$')
plt.xlabel('Time (s)')
plt.ylabel('$\Theta$')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
# plt.show()


plt.figure()
plt.plot(x_values[:(rand_init_input_vec.shape[0]*int(sampling_time/dt)),0], 
        x_values[:(rand_init_input_vec.shape[0]*int(sampling_time/dt)),1], 'red', label='$\mathrm{Unknown\ trajectory}$')
plt.plot(rand_init_traj_vec[:-1,0], rand_init_traj_vec[:-1,1], markerTraj, label='$\mathscr{T}_{'+ str(tMeas.shape[0])+'}$')

# first = True
# for i in range(tValOver.shape[0]):
#     x1_d,x2_d,y1_d,y2_d = datareach_nd_lb[i,0], datareach_nd_ub[i,0], datareach_nd_lb[i,1], datareach_nd_ub[i,1]
#     if first:
#         plt.fill([x1_d,x2_d,x2_d,x1_d], [y1_d,y1_d,y2_d,y2_d],\
#                 facecolor='coral', edgecolor='darksalmon', alpha=0.8,\
#                 label='$\mathrm{DaTaReach,\ case\ (a)}$')
#         first = False
    
#     plt.fill([x1_d,x2_d,x2_d,x1_d], [y1_d,y1_d,y2_d,y2_d],\
#         alpha=0.8,facecolor='coral', edgecolor='darksalmon')

first = True
for i in range(integ_d.shape[0]):
    x1_d,x2_d,y1_d,y2_d = datareach_d_lb[i,0], datareach_d_ub[i,0], datareach_d_lb[i,1], datareach_d_ub[i,1]
    if first:
        plt.fill([x1_d,x2_d,x2_d,x1_d], [y1_d,y1_d,y2_d,y2_d],\
                facecolor='green', edgecolor='darkgreen', alpha=0.8,\
                label='$\mathrm{DaTaReach,\ case\ (b)}$')
        first = False
    
    plt.fill([x1_d,x2_d,x2_d,x1_d], [y1_d,y1_d,y2_d,y2_d],\
        alpha=0.8,facecolor='green', edgecolor='darkgreen')

# first = True
# for i in range(tValOver.shape[0]):
#     x1_d,x2_d,y1_d,y2_d = datareach_dt_lb[i,0], datareach_dt_ub[i,0], datareach_dt_lb[i,1], datareach_dt_ub[i,1]
#     if first:
#         plt.fill([x1_d,x2_d,x2_d,x1_d], [y1_d,y1_d,y2_d,y2_d],\
#                 facecolor='purple', edgecolor='darkmagenta', alpha=0.8,\
#                 label='$\mathrm{DaTaReach,\ case\ (c)}$')
#         first = False
    
#     plt.fill([x1_d,x2_d,x2_d,x1_d], [y1_d,y1_d,y2_d,y2_d],\
#         alpha=0.8,facecolor='purple', edgecolor='darkmagenta')
    
first = True
for i in range(integ_d.shape[0]):
    x1,x2,y1,y2 = reach_lb[i,0], reach_ub[i,0], reach_lb[i,1], reach_ub[i,1]
    if first:
        plt.fill([x1,x2,x2,x1], [y1,y1,y2,y2],\
                facecolor='tab:cyan', edgecolor='darkcyan', alpha=0.8,\
                label='$\mathrm{Reachable\ set}$')
        first = False
    plt.fill([x1,x2,x2,x1], [y1,y1,y2,y2],\
        alpha=0.8, facecolor='tab:cyan', edgecolor='darkcyan')
    
plt.xlabel('$p_x$')
plt.ylabel('$p_y$')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# for i, t in enumerate(l):
#     DaTaReach(sidyn, rand_init_traj_vec[i], t, 1, sampling_time, _uover, _uder, 
#                 fixpointWidenCoeff=0.2, zeroDiameter=1e-5, 
#                 containTol=1e-4, maxFixpointIter=10)

# print(DaTaReach(sidyn, rand_init_traj_vec[0], 0., rand_init_input_vec.shape[0], sampling_time, _uover, _uder, 
#                 fixpointWidenCoeff=0.2, zeroDiameter=1e-5, 
#                 containTol=1e-3, maxFixpointIter=10))


# print(rand_init_traj_vec)
# print(rand_init_input_vec)
# print(rand_init_traj_der_vec)