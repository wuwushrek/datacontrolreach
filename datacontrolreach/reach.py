import jax
import datacontrolreach.jumpy as jp

from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

from datacontrolreach import interval
from datacontrolreach.interval import Interval as iv

import datacontrolreach.dynsideinfo as dynsideinfo
from datacontrolreach.dynsideinfo import DynamicsWithSideInfo

def build_overapprox(dynclass, feas_state_input : Tuple[jp.ndarray, jp.ndarray], lipinfo : dict, 
                        xTraj : Optional[jp.ndarray] = None, xDotTraj : Optional[jp.ndarray] = None,
                        uTraj : Optional[jp.ndarray] = None, max_data_size : Optional[int] = 250):
    """ Build an object of type dynclass (DynamicsWithSideInfo or its children) that provides
        the utilities function to compute the differential inclusion of the underlying dynamics
        :param dynclass         : The class DynamicsWithSideInfo or any subset of the class
        :param feas_state_input : A function that provides a feasible state and input for initialization
        :param lipinfo          : The regularity side information for each unknown variable
        :param xTraj            : The sequence of state trajectories currently available
        :param xDotTraj         : The sequence of state (noisy and might be interval) derivative available
        :param uTraj            : The control signal applied at xTraj with resulting xDotTraj
        :param max_data_size    : The maximum data size of the resulting DynamicsWithSideInfo structure
    """
    # Obtain feasible set of states and control
    xfeas, ufeas = feas_state_input
    n, m = xfeas.shape[0], ufeas.shape[0]

    # Initialize the trajectory of the system
    _xTraj = jp.full((max_data_size, n), fill_value=jp.inf) # jp.zeros((max_data_size, n))
    _xTraj = jp.index_update(_xTraj, 0, xfeas)

    # COmpute the known term to get a structure of what need to be saved for contraction
    kTerms = dynclass.known_terms(xfeas, ufeas)
    _contHistory = {fv : jp.zeros((max_data_size, *v.shape), dtype= iv if isinstance(v, iv) else None) for fv, v in kTerms.items()}

    # We do not know yet information on xdot so fill it with the set Rn
    _xDotTraj = jp.full((max_data_size, n), fill_value=iv(-jp.inf, jp.inf))

    # Use the bound on the unkown terms to provide initial their initial overappproximations
    _unkAtXtraj = { key : jp.full(shape=(max_data_size,), fill_value=v.bound) for key, v in lipinfo.items()}

    # Finally build the instance with side information
    _dyn = dynclass(lipinfo, _xTraj, _xDotTraj, _unkAtXtraj, _contHistory, 0, 1)
    # If initial data are given apply contraction rules to include them in the dataset
    if xTraj is not None and xDotTraj is not None and uTraj is not None:
        rollout_scan = lambda carry, other : (update_overapprox(carry, *other), None)
        _dyn, _ = interval.scan_interval(rollout_scan, _dyn, (xTraj[:uTraj.shape[0]], uTraj, xDotTraj))
    return _dyn
    
def update_overapprox(dyn : DynamicsWithSideInfo, xval : jp.ndarray, uval : jp.ndarray, xdotnoise : jp.ndarray):
    """ Define the one step refinement procedure of the dynamics
        :param dyn          : A system dynamics with side information
        :param xval         : The current state value
        :param uval         : The currently applied control signal
        :param xdotnoise    : The corresponding noisy derivative of the state
        :param args         : Extra arguments required by the function to propagate back the contracted value to the
                                rest of the data set 
    """
    unkTerms = dynsideinfo.overapprox(dyn, xval)
    kTerms = type(dyn).known_terms(xval, uval)
    unkTerms = type(dyn).contraction(unkTerms, kTerms, xdotnoise)
    dyn = update_dynexpr(dyn, xval, xdotnoise, kTerms, unkTerms)

    # TODO [Improve the update rules based on distance between data point in the data set]
    def cond_fun(carry):
        _count, _ =  carry
        return _count < dyn.dataNum-1

    def body_fun(carry):
        _count, _dyn =  carry
        # The number of point in the trajectory has not changed -> so we use dyn.dataNum
        closestp = (-_count + dyn.dataIndx - 1) % dyn.dataNum # Current point index

        # At the first iteration, this point is the point preceding the current point dyn.dataNum
        currp = (-_count + dyn.dataIndx - 2) % dyn.dataNum 
        _xval = dyn.xTraj[currp]

        # The unknown terms are overapproximated using the closest point -> assume to be the current point
        _unkTerms = { key : (_dyn.unkAtXtraj[key][currp] & v) for key, v in dynsideinfo._overapprox(_dyn, _xval, closestp).items()}

        # We extract the known terms that has been previously saved
        _kTerms = {key : v[currp] for key, v in dyn.contHistory.items() }
        _unkTerms = type(dyn).contraction(_unkTerms, _kTerms, dyn.xDotTraj[currp])
        m_unkAtXtraj = { key : jp.index_update(_dyn.unkAtXtraj[key], currp, v) for key, v in _unkTerms.items()}

        # Increment the number of iterations
        _count = _count + 1
        return _count, type(dyn)(dyn.unkLipDict, dyn.xTraj, dyn.xDotTraj, m_unkAtXtraj, dyn.contHistory, dyn.dataIndx, dyn.dataNum)
    # res = interval.while_interval(cond_fun, body_fun, (0, dyn))
    # DO the loop to update the remaining point in the data set
    return interval.while_interval(cond_fun, body_fun, (0, dyn))[1]
    # return res[1]


def update_dynexpr(dyn, xval, xdotnoise, kTerms, unkTerms):
    """ Return a new object DynamicsWithSideInfo where the data set is updated
        with new measurements from the system
        :param dyn       : A system dynamics with side information
        :param xval      : The new state of the system
        :param xdotnoise : The noisy state derivative
        :param kTerms    : The known terms needed to apply the contraction algorithm
        :param unkTerms  : Overapprox of unknown terms at xval
    """
    # # In case only the unkTerm needs to be updated (we are not adding a new point)
    # if extra is None:
    #     return type(dyn)(dyn.unkLipDict, dyn.xTraj, dyn.xDotTraj, m_unkAtXtraj, dyn.contHistory, dyn.dataIndx, dyn.dataNum)

    # In case this corresponds to a new data point --> Perform adequate updates
    n_data_indx = (dyn.dataIndx + 1) % dyn.xTraj.shape[0]
    n_data_num = dyn.dataNum + jp.where(jp.logical_and(dyn.dataNum < dyn.xTraj.shape[0], dyn.dataIndx > 0), 1, 0)
    m_xtraj = jp.index_update(dyn.xTraj, dyn.dataIndx, xval)
    m_xdotnoise = jp.index_update(dyn.xDotTraj, dyn.dataIndx, xdotnoise)
    m_contHistory = { key : jp.index_update(dyn.contHistory[key], dyn.dataIndx, v) for key, v in kTerms.items()}
    m_unkAtXtraj = { key : jp.index_update(dyn.unkAtXtraj[key], dyn.dataIndx, v) for key, v in unkTerms.items()}
    return type(dyn)(dyn.unkLipDict, m_xtraj, m_xdotnoise, m_unkAtXtraj, m_contHistory, n_data_indx, n_data_num)


def apriori_enclosure(dyn, xval, uOver, dt, fixpointWidenCoeff=0.2, 
                        zeroDiameter=1e-5, containTol=1e-4, maxFixpointIter=10):
    """ Compute the a prioir enclosure via either the Theorem in the paper (Gronwall lemma)
        or using a fixpoint iteration scheme
        :param 
        :param
        :param
        :param
    """
    # First compute the vector field at the current pont
    fx, _ = dynsideinfo.dynamics(dyn, xval, uOver)

    # Initial a priori enclosure using the fixpoint formula
    iv_odt = iv(0., dt)
    S = xval + fx * iv_odt

    # Prepare the loop
    def cond_fun(carry):
        isIn, count, _ = carry
        return jp.logical_and(jp.logical_not(isIn) , count <= maxFixpointIter)

    def body_fun(carry):
        _, count, _pastS = carry
        _newS = xval + dynsideinfo.dynamics(dyn, _pastS, uOver)[0] * iv_odt
        # Width increment step
        width_pasS = _pastS.width
        radIncr = jp.where( width_pasS <= zeroDiameter, jp.abs(_pastS.ub), width_pasS) * fixpointWidenCoeff
        # CHeck if the fixpoint condition is satisfied
        isIn = _pastS.contains(_newS, tol=containTol)
        _pastS = jp.where(isIn, _newS, _newS + iv(-radIncr,radIncr))
        return jp.all(isIn), (count+1), _pastS

    _, nbIter, S = interval.while_interval(cond_fun, body_fun, (False, 1, S))
    print('Num Iter ', nbIter)
    fS, _ = dynsideinfo.dynamics(dyn, S, uOver)
    return xval + fS * iv_odt, fx, fS


def DaTaReach(dyn, x0, t0, nPoint, dt, uOver, uDer, 
                fixpointWidenCoeff=0.2, zeroDiameter=1e-5, 
                containTol=1e-4, maxFixpointIter=10):
    """ Compute an over-approximation of the reachable set at time
        t0, t0+dt...,t0 + nPoint*dt.

    Parameters
    ---------- 
    :param dyn : The object representing the dynamics with side information
    :param x0 : Intial state
    :param t0 : Initial time
    :param nPoint : Number of point
    :param dt : Integration time
    :param uOver : interval extension of the control signal u
    :param uDer : Interval extension of the derivative of the control signal u
    :param fixpointWidenCoeff : A coefficient to enlarge the a priori enclosure
    :param containTol : THe tolerance to determine if a fixpoint has been reached
    :param maxFixpointIter : The maximum number of fixpoint iteration

    Returns
    -------
    list
        a list of different time at which the over-approximation is computed
    list
        a list of the over-approximation of the state at the above time
    """

    # Save the integration time
    integTime = jp.array([t0 + i*dt for i in range(nPoint+1)])

    # Save the tube of over-approximation of the reachable set
    curr_x = iv(x0)

    # Constant to not compute every time
    dt_2 = (0.5* dt**2)

    # Define the main body to compute the over-approximation
    def op(xt, tval):
        # Over approximation of the control signal between t and t+dt
        ut = uOver(tval)
        ut_dt = uOver(iv(tval, tval+dt))
        uder_t_dt = uDer(iv(tval, tval+dt))


        # Compute the remainder term given the dynamics evaluation at t
        St, _, fst_utdt = apriori_enclosure(dyn, xt, ut_dt, dt, 
            fixpointWidenCoeff, zeroDiameter, containTol, maxFixpointIter)

        # Compute the known term of the dynamics at t
        fxt_ut, _ = dynsideinfo.dynamics(dyn, xt, ut)

        # Define the dynamics function
        dyn_fun = lambda x : dynsideinfo.dynamics(dyn, x, ut_dt)[0]
        dyn_fun_u = lambda u : dynsideinfo.dynamics(dyn, St, u)[0]

        _, rem = jax.jvp(dyn_fun, (St,), (fst_utdt,))
        _, remu = jax.jvp(dyn_fun_u, (ut_dt,), (uder_t_dt,))
        next_x = xt + dt * fxt_ut + dt_2 * (rem + remu)
        return next_x, next_x

    # Scan over the operations to obtain the reachable set
    _, res = interval.scan_interval(op, curr_x, integTime[:-1])
    return jp.vstack((curr_x,res)), integTime