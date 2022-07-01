import jax
import jax.numpy as jnp
import datacontrolreach.jumpy as jp

from collections import namedtuple
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

from datacontrolreach import interval as itvl
from datacontrolreach.interval import Interval as iv

from functools import partial

# Provide regularity information on the unknwon terms
# L is the Lipschitz constant
# vDep is the set of variable that the function depends on
# weightLip is the relative importance of each input to another one
# bound is an upper bound on the range of the function
# gradBound is an upper bound on the range of the gradient 
LipInfo = namedtuple("LipInfo", ["L", "vDep", "weightLip", "bound", "gradBound"])

# Utility function to provide information on unknown terms
def lipinfo_builder(Lip=0.0, vDep=[], weightLip=[], n=0, bound=(-1e5,1e5), gradBound=None):
    """ Utils function to build the regularity side information
        :param Lip          : Lipschitz constant
        :param vDep         : A set of indexes
        :param weightLip    : An array of positive coefficients
        :param n            : Number of state 
        :param bound        : A tuple (lower, upper)
        :param gradBound    : A list of tuple (lower, upper) of the same dimension as weightLip
    """
    assert n > 0 and Lip >= 0, 'n = {} <=0 | Lip = {} < 0'.format(n, Li)

    # Get the dependency variable of the system
    s_vDep = set(vDep)
    vDep =  tuple([int(vp) for vp in s_vDep])

    # Define the weigh vector
    if weightLip is None or len(weightLip) == 0:
        weightLip = [1.0 for _ in s_vDep]

    assert jnp.amin(jnp.array(weightLip)) > 0 , 'weightLip must be all positive'
    assert len(vDep) == len(weightLip), 'WeightLip must be the same dimension as vDep'

    # Weight for the relative importance of each input of the unknown terms
    weightLip_ = jnp.array(weightLip, dtype=float)

    # Gradient information
    lipIndex = { i : Lip*w for i,w in zip(s_vDep, weightLip)} # Gradient is bounded by Lipschitz * relative importance
    if gradBound is None: # If no extra gradient bound is given
        gradBound = [ (-lipIndex[i], lipIndex[i]) if i in vDep else (0.0,0.0) for i in range(n)]
    else: # If gradient bound extra gradient bounds are given, intersect them with the Lipschitz bounds
        # Store the 
        dIndex = { i : v for i,v in zip(s_vDep, gradBound)}
        gradBound = [ (max(-lipIndex[i],dIndex[i][0]), min(lipIndex[i], dIndex[i][1])) if i in vDep else (0.0,0.0) \
                        for i in range(n)]
        for i, (lb, ub) in enumerate(gradBound):
            assert lb <= ub, 'Gradient bound are not sound L={}, i={}, vDep={}, weightLip={}, lb= {}, ub = {}'.format(Lip, i, vDep, weightLip_, lb, ub)

    return LipInfo(L=float(Lip), vDep=vDep, weightLip=weightLip_, bound=iv(lb=float(bound[0]), ub=float(bound[1])), 
                    gradBound=iv(lb=jnp.array([l for (l,u) in gradBound], dtype=float), ub=jnp.array([u for (l,u) in gradBound], dtype=float))
                  )


@jax.tree_util.register_pytree_node_class
class DynamicsWithSideInfo:
    """ Define a class representing the dynamics of an unknown system and several 
        a priori side information about the dynamics in terms of compositional function 
        of known terms and unknown terms of the dynamics and possible constraints on the
        internal state (known and unknown terms)
    """
    def __init__(self, unkLipDict, xTraj, xDotTraj, unkAtXtraj, contHistory, dataIndx, dataNum):
        """ The unknown term of the dynamics are described with regularity information
            and a dataset of fixed size containing over-approximation of the 
            unknown term at given point
        """
        self.unkLipDict = unkLipDict

        assert hasattr(xTraj, 'shape') and xTraj.ndim == 2, 'Trajectory should be a 2D array'
        self.xTraj = xTraj

        # Uncertain xDot measurement (include noise bound)
        assert isinstance(xDotTraj, iv) and xDotTraj.ndim == 2, 'xDotTraj must be 2D intervals'
        self.xDotTraj = xDotTraj

        assert self.xDotTraj.shape == self.xTraj.shape, 'Measurements must agree in size'

        # Overapproximation of the unknown term at each point of the trajectory
        # TODO : Check if the key of the dict matches with unkLipDict 
        self.unkAtXtraj = unkAtXtraj

        # COntraction argument history for each data point (dictionary with ket similar to the key given by known terms)
        self.contHistory = contHistory

        # Save the current number of data and data indx of the latest added data
        # This simulate a circular array
        self.dataIndx = jnp.array(dataIndx, dtype=int)
        self.dataNum = jnp.array(dataNum, dtype=int)

    def __repr__(self):
        m_str = '=====================================================\n'
        m_str += 'Number data = {} | Index latest = {} | Max data = {}\n\n'.format(self.dataNum, self.dataIndx, self.xTraj.shape[0])
        m_str += '---     State Evolution        ---\n{}\n\n'.format(self.xTraj[:self.dataNum])
        m_str += '--- State derivative Evolution ---\n{}\n\n'.format(self.xDotTraj[:self.dataNum])
        m_str += '---  Unknown Terms Overapprox  ---\n{}\n'.format('\n'.join( '{} : \n{}\n'.format(key, v[:self.dataNum]) for key,v in self.unkAtXtraj.items()))
        m_str += '---        History Data        ---\n{}\n'.format('\n'.join( '{} : \n{}\n'.format(key, v[:self.dataNum]) for key,v in self.contHistory.items()))
        return m_str
    
    @staticmethod
    def known_terms(x, u):
        """ This function computes a bundle of known terms of the dynamics and return them
            as a dictionary with specific name associated to each term or an empty dictionary

            [Example: ] If the vector field \dot{x} = f(x)*h(x) + G(x)u, where f and G are known
            and h is unknown. Then, a possible implementation of this function is:
            def known_terms(x):
                return {'f' : f(x), 'G' : G(x)*u}

            :param x : The state of the system
            :param u : The control input of the system
        """
        raise NotImplementedError

    @staticmethod
    def composed_dynamics(kTerms, unkTerms):
        """ This function return the vector field of the dynamics by combining the known terms
            and unknown terms. It assumes that kTerms and unkTerms are evaluated at the same
            state x and input u

            [Example: ] If the vector field \dot{x} = f(x)*h(x) + G(x)u, where f and G are known
            and h is unknown. Then, a possible implementation of known_terms function is:
            def known_terms(x):
                return {'f' : f(x), 'Gu' : G(x)*u}

            AND an implementation of this function is given by:
            def composed_dynamics(kTerms, unkTerms):
                return kTerms['f']*unkTerms['h'] + kTerms['Gu']

            :param kTerms    : The known terms of the dynamics as resulting from self.known_terms
            :param unkTerms  : The unknown terms of the dynamics 
        """
        raise NotImplementedError


    @staticmethod
    def contraction(unkTerms, kTerms, noisyXdot):
        """ Contract the overapproximation of unkTerms according to the dynamics given by self.dynamics
            This function return a dictionary similar to unkTerms such that each key of unkTerms corresponds
            to interval values that are contained inside the original argument unkTerms.
            Basically, this function applies a contraction algorithm based on the constraint
                noisy_xdot = self.dynamics(kTerms, )
            We provide as utilities of this class some contraction algorithm for 
            special case such as linear constraints but future commit will generalize the contraction algorithm
            
            :param unkTerms   : An initial approximation of the unknown terms of the dynamics
            :param kTerms     : The known terms of the dynamics as given by self.known_terms
            :param noisyXdot  : A noisy measurement of the derivative of the states

        Returns
        -------
        A dictionary containing contracted value of unkTerms

        """
        return unkTerms
        # raise NotImplementedError

    @staticmethod
    def sideinfo_state(xover):
        """ Contract an uncertain state value with respect to state-only constraints 

            :param xover : An uncertain (interval) state

        Returns
        -------
        A scalar contracted state of the same shape and type as x 
        """
        return xover

    def sideinfo_vectorfield(self, unkTerms, unkGradTerms, x):
        """ This function provides a contraction scheme on side infomration 
            constraints involving the unknown unkTerms. It is principally used
            prior to computing the reachable sets and return a contracted unkTerms

            :param unkTerms     : A dictionary providing an initial overapproximation of the unknown 
                                    terms of the dynamics evaluated at x
            :param unkGradTerms : A dictionary providing an initial overapproximation of the gradient of
                                    the unkown terms at the current state x
            :param x            : The current state of the system

        Returns
        -------
        A dictionary containing contracted value of unkTerms and unkGradTerms and of the same shape and type as above
        """
        return unkTerms, unkGradTerms

    def tree_flatten(self):
        return ((self.unkLipDict, self.xTraj, self.xDotTraj, self.unkAtXtraj, self.contHistory, self.dataIndx, self.dataNum), None)

    @classmethod
    def tree_unflatten(cls, _, args):
        return cls(*args)

############################################ Dynamics function #############################################

def dynamics(dyn, x, u):
    """ Define the dynamics associated to this system. This function is based on known_terms
        , composed_dynamics, sideinfo_vectorfield, and the regularity side information
        
        :param dyn  : The object representing the dynamics and its side information
        :param x    : The current state of the system
        :param u    : The current control input applied to the system
    """
    # First compute the known terms of the dynamics and save them
    kTerms = type(dyn).known_terms(x, u)

    # Compute the unknown terms of the dynamics according the regularity side information
    unkTerms = overapprox(dyn, x)

    # Compose the known and unknown terms
    vField = type(dyn).composed_dynamics(kTerms, unkTerms)
    return vField, (kTerms, unkTerms)
    

@partial(jax.custom_jvp, nondiff_argnums=(0,))
def overapprox(dyn, x):
    """ Compute an over-approximation of all the unknown terms of the dynamics
        using the regularity information and the available side information on the 
        vector field
        :param dyn  : The object representing the dynamics and its side information
        :param x    : The current state of the system
    """
    return _overapprox(dyn, x)


@overapprox.defjvp
def overapprox_jvp(dyn, primal, tangent):
    """ Provide a way to compute jacobian vector product of the over-approximation terms
        :param dyn       : The object representing the dynamics and its side information
        :param primal    : The primal space
        :param tangent   : The tangent space
    """
    # Get the primal and the tangent
    x, = primal
    xdot, = tangent
    unkTerms = { key : lipoverapprox(x, dyn.xTraj, dyn.unkAtXtraj[key], _lipinfo) for key, _lipinfo in dyn.unkLipDict.items()}
    unkTerms, unkGrad = dyn.sideinfo_vectorfield(unkTerms, {_key : _lipinfo.gradBound for _key, _lipinfo in dyn.unkLipDict.items()}, x)
    # print(unkGrad)
    return unkTerms, { key : v @ xdot for key, v in unkGrad.items() }


def _overapprox(dyn, x, closestp: jp.ndarray = None):
    """ Compute an over-approximation of all the unknown terms of the dynamics
        using the regularity information and the available side information on the 
        vector field
        :param dyn  : The object representing the dynamics and its side information
        :param x    : The current state of the system
        :param closestp : The closest point from the dataset if available
    """
    # Compute all the unknown terms from 
    unkTerms = { key : lipoverapprox(x, dyn.xTraj, dyn.unkAtXtraj[key], _lipinfo, closestp) for key, _lipinfo in dyn.unkLipDict.items()}

    # Apply the side information (if available) to reduce the over-approximation of the unknown terms
    unkTerms, _ = dyn.sideinfo_vectorfield(unkTerms, {_key : _lipinfo.gradBound for _key, _lipinfo in dyn.unkLipDict.items()}, x)
    return unkTerms

# Function to compute the Lipschitz overapproximation
# TODO add an argument for the contracted gradient for autodifferentiation (maybe faster than creating a new LipInfo)
def lipoverapprox(x : jp.ndarray, xTraj : jp.ndarray, fover: jp.ndarray, lipinfo : LipInfo, closestp: Optional[jp.ndarray] = None):
    """ Function to over-approximate a real-valued function at the given
        input x based on the Lipschitz constants of the function, and
        interval-based overapproixmations of the values of such function

        :param x        : The point (or interval vector) at which to evaluate the function based on data
        :param xTraj    : The data set of state pas trajectories
        :param fover    : A data set of overapproximation of the function at point in xTraj
        :param lipinfo  : Provide regularity information on the unknown function
        :param closestp : The closest point from the dataset if available

        Returns
        -------
        A scalar value over-approximating the unknown term
    """
    if closestp is not None:
        closestx, foverClosestx = xTraj[closestp, lipinfo.vDep], fover[closestp]
    else:
        midx = x[lipinfo.vDep].mid if isinstance(x, iv) else x[lipinfo.vDep]
        depXtraj = xTraj[:,lipinfo.vDep]
        # Use jp here to account for the fact that it can be an interval
        indxClosest = jp.argmin(jp.norm( (midx - depXtraj) * lipinfo.weightLip, axis=1))
        closestx, foverClosestx = depXtraj[indxClosest], fover[indxClosest]
    # DO the intersection with the a priori bound on the range of the function
    return (foverClosestx + iv(-lipinfo.L, lipinfo.L) * jp.norm((x[lipinfo.vDep]-closestx)*lipinfo.weightLip)) & lipinfo.bound


############################### Define some contraction rule as utility function for the user #############################

def hc4revise_lin_eq(rhs : jp.ndarray, coeffs : jp.ndarray, unk : iv):
    """ Apply HC4Revise for the constraint coeffs @ vars = rhs and return the contracted vars
        This function assumes that coeffs are not zeros
        :param rhs          : The right hand side of the hc4revise
        :param coeffs       : The coefficient of the constraints
        :param vars         : Initial overapproximation of the range of the variables in the constraints
    """
    # Save temporary higher order sums
    _S = unk * coeffs
    _S_lb, _S_ub = jnp.cumsum(_S.lb[::-1]), jnp.cumsum(_S.ub[::-1])
    _S = iv(lb=_S_lb, ub=_S_ub)
    Csum = _S[-1] & rhs # The sum and the rhs are equals

    # Perform contraction along a given variable and propagate the contracted sum
    def op(carry, extra):
        _unk, _c, _csum = extra
        # Trick to avoid division by 0
        zero_notin_c = jp.logical_or(_c > 0, _c < 0)
        _ct = jp.where(zero_notin_c, _c, 1. if not isinstance(coeffs, iv) else iv(1.))
        Cunk = jp.where(zero_notin_c, ((carry - _csum) & (_c * _unk))/_ct, _unk)
        # Use the newly Cunk to provide a contraction of the remaining sum
        carry = (carry - (Cunk * _c)) & _csum
        return carry, Cunk

    _, cunk = jax.lax.scan(op, Csum, (unk, coeffs, itvl.concatenate((_S[::-1], iv(0.)))[1:]) )
    return cunk

def hc4revise_lin_leq(rhs : Union[jp.ndarray, iv], coeffs : jp.ndarray, unk : iv):
    """ Apply HC4Revise for the constraint coeffs @ vars <= rhs and return the contracted vars
        This function assumes that coeffs are not zeros
        :param rhs          : The right hand side of the hc4revise
        :param coeffs       : The coefficient of the constraints
        :param vars         : Initial overapproximation of the range of the variables in the constraints
    """
    # Save temporary higher order sums
    _S = unk * coeffs
    _S_lb, _S_ub = jnp.cumsum(_S.lb[::-1]), jnp.cumsum(_S.ub[::-1])
    _S = iv(lb=_S_lb, ub=_S_ub)
    Csum = _S[-1] & iv(lb=-jp.inf, ub=rhs.lb if isinstance(rhs, iv) else rhs) # The sum and the rhs are equals

    # Perform contraction along a given variable and propagate the contracted sum
    def op(carry, extra):
        _unk, _c, _csum = extra
        # Trick to avoid division by 0
        zero_notin_c = jp.logical_or(_c > 0, _c < 0)
        _ct = jp.where(zero_notin_c, _c, 1. if not isinstance(coeffs, iv) else iv(1.))
        Cunk = jp.where(zero_notin_c, ((carry - _csum) & (_c * _unk))/_ct, _unk)
        # Use the newly Cunk to provide a contraction of the remaining sum
        carry = (carry - (Cunk * _c)) & _csum
        return carry, Cunk

    _, cunk = jax.scan(op, Csum, (unk, coeffs, itvl.concatenate((_S[::-1], iv(0.)))[1:]) )
    return cunk

def hc4revise_lin_geq(rhs : Union[jp.ndarray, iv], coeffs : jp.ndarray, unk : iv):
    """ Apply HC4Revise for the constraint coeffs @ vars >= rhs and return the contracted vars
        This function assumes that coeffs are not zeros
        :param rhs          : The right hand side of the hc4revise
        :param coeffs       : The coefficient of the constraints
        :param vars         : Initial overapproximation of the range of the variables in the constraints
    """
    # Save temporary higher order sums
    _S = unk * coeffs
    _S_lb, _S_ub = jnp.cumsum(_S.lb[::-1]), jnp.cumsum(_S.ub[::-1])
    _S = iv(lb=_S_lb, ub=_S_ub)
    Csum = _S[-1] & iv(lb=rhs.ub if isinstance(rhs, iv) else rhs, ub=jp.inf) # The sum and the rhs are equals

    # Perform contraction along a given variable and propagate the contracted sum
    def op(carry, extra):
        _unk, _c, _csum = extra
        # Trick to avoid division by 0
        zero_notin_c = jp.logical_or(_c > 0, _c < 0)
        _ct = jp.where(zero_notin_c, _c, 1. if not isinstance(coeffs, iv) else iv(1.))
        Cunk = jp.where(zero_notin_c, ((carry - _csum) & (_c * _unk))/_ct, _unk)
        # Use the newly Cunk to provide a contraction of the remaining sum
        carry = (carry - (Cunk * _c)) & _csum
        return carry, Cunk

    _, cunk = jax.scan(op, Csum, (unk, coeffs, itvl.concatenate((_S[::-1], iv(0.)))[1:]) )
    return cunk



# rhs = jnp.array(0.0)
# coeffs = jnp.array([1.0, 1.0, 0.1])
# unk = iv(lb=jnp.array([-0.01, -0.05, -0.1]), ub=jnp.array([1.0, 0.05, 1.]))

# print(jax.jit(hc4revise_lin_eq)(rhs, coeffs, unk))
# print(hc4revise_lin_eq(rhs, coeffs, unk))

# dictUnk = {'f1' : lipinfo_builder(Lip=1.0, vDep=[0,1], weightLip=[1.0,0.1], n=3, bound=(-1e5,1e5), gradBound=None)}
# xTraj = jnp.array([[1.0, 2.0, 3.0]])
# xDotTraj = iv(lb = jnp.array([[1.0, 2.0, 3.0]]), ub=jnp.array([[1.0, 2.0, 3.0]]))
# unkAtXtraj = {'f1' : iv(lb = jnp.array([[1.0, 2.0, 3.0]]), ub=jnp.array([[1.0, 2.0, 3.0]]))}

# x = jax.numpy.array([1,2,3])
# def testf (x, dyn):
#     print(jax.tree_util.tree_flatten(dyn))
#     return x[dyn.data_indx]
# testf = jax.jit(testf)

# mdyn = DynamicsWithSideInfo(dictUnk, xTraj, xDotTraj, unkAtXtraj, data_indx=2)
# testf(x, mdyn)
# print('---')
# testf(x, mdyn)
# print('---')
# t = lambda x : jax.jvp(testf, (x, DynamicsWithSideInfo(dictUnk, xTraj, xDotTraj, unkAtXtraj, data_indx=1))
# print(testf(x, DynamicsWithSideInfo(dictUnk, xTraj, xDotTraj, unkAtXtraj, data_indx=1)))