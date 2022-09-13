import jax
import jax.numpy as jnp

from jax.experimental.host_callback import id_print

from typing import NamedTuple

# Optax for the optimization scheme
import optax

def initialize_optax_solver(init_guess, params, projection_fn=None):
    """ Define an mpc solver based on Optax Library
    """
    # Project the initial guess onto the control constraint set
    if projection_fn is not None:
        init_guess = projection_fn(init_guess)

    # Initialize the optimizer using user-defined parameters
    optim = params['optimizer']
    chain_list = []
    for elem in optim:
        m_fn = getattr(optax, elem['name'])
        m_params = elem.get('params', {})
        if elem.get('scheduler', False):
            m_params = m_fn(**m_params)
            chain_list.append(optax.scale_by_schedule(m_params))
        else:
            chain_list.append(m_fn(**m_params))

    # Construct the optimizer object
    opt = optax.chain(*chain_list)

    # Get the state of the optimizer
    init_opt_state = opt.init(init_guess)

    # Define the update function
    def one_step_optim(carry, cost_fn):
        # Parse the arguments
        curr_opt, _opt_state, past_extra, notdone = carry
        past_cost, _ = past_extra

        # Compute the gradient of the cost
        # extras is a tuple (cost, states)
        grads, extras = jax.jacfwd(cost_fn, has_aux=True)(curr_opt)
        # print('Gradient:')
        # id_print((grads,))
        updates, _opt_state = opt.update(grads, _opt_state, curr_opt)
        new_u = optax.apply_updates(curr_opt, updates)

        # Project the control value
        if projection_fn is not None:
            new_u = projection_fn(new_u)

        # Check if stopping criteria are done
        lb_cost = params['atol'] + params['rtol']*jnp.abs(extras[0])
        notdone = jnp.abs(extras[0]-past_cost) > lb_cost

        # notdone = jnp.array(True)
        new_u = jnp.where(notdone, new_u, curr_opt)
        return new_u, _opt_state, extras, notdone

    def synth_control(opt_state, cost_fn):
        # Obtain the control value
        curr_u = opt_state['u']

        # Now, assuming it is the solution of past iteration
        # we move the element one step backward
        curr_u = curr_u.at[:-1].set(curr_u[1:])

        # Compute the initial cost and state distribution
        init_cost, init_extra = cost_fn(curr_u)

        # Optimizer internal state _> Keep track of number iteration, step size, moments etc
        _opt_state = opt_state['opt_state']

        # Now the state of our descent optimization
        optim_state = (curr_u, _opt_state, (jnp.inf, init_extra[1]), jnp.array(True))

        # The body of each iteration in the loop
        one_step_control = lambda t : one_step_optim(t, cost_fn)

        # Perform the loop iteration
        (opt_u, n_opt_state, extra, notdone) = _while_loop(cond_fun = lambda t: t[-1],
                        body_fun=one_step_control,
                        init_val=optim_state,
                        max_iter=params['max_iter']
                    )

        # Save the opt_state for the next iteratuion
        opt_state = {'u' : opt_u, 'init_cost' : init_cost, 'opt_cost': extra[0],
                        'opt_state' : n_opt_state, 'not_done' : notdone }

        # Return the optimal control and the opt state to propagate
        # Extra contains both optcost and states evolution
        return opt_u, opt_state, extra

    # Define the opt state that will be propagate during the entire control
    init_opt_state = {'u' : init_guess, 'init_cost' : jnp.inf, 'opt_cost' : jnp.inf,
                        'opt_state' : init_opt_state, 'not_done' : jnp.array(True)}

    return init_opt_state, synth_control




######### Not USED YET, Will incude it later ##########
# Implementation inspired from and Armijo line search paper
# Convergence Analysis of Proximal Gradient with Momentum for Nonconvex
# Optimization
# A Function to estimate the suboptimality error
opt_stop_criteria = lambda grad_val: jnp.sum(jnp.square(grad_val))

def init_opt_state(opt_var, optimizer_params):
    """Create an  instance of APGState given

    Args:
        opt_var (TYPE): Description
        optimizer_params (TYPE): Description
    """
    # Extract the initial stepsize
    init_step_size = jnp.array(optimizer_params['linesearch']['max_stepsize']\
                        if 'linesearch' in optimizer_params \
                        else optimizer_params['stepsize'])

    # The momentum, stepsize, and avg_stepsize
    momentum = jnp.array(optimizer_params.get('beta_init', 0.25))

    # Dummy initialization given opt_var
    opt_state_init = APGState(num_steps = jnp.array(1),
                    momentum = momentum,
                    avg_momentum = momentum,
                    # num_linesearch = jnp.array(1),
                    avg_linesearch = jnp.array(1.),
                    stepsize = init_step_size,
                    avg_stepsize = init_step_size, # This can be used to warm-up online next mpc calls
                    opt_cost = jnp.array(0.),
                    init_cost = jnp.array(0.),
                    grad_sqr = jnp.array(0.),
                    not_done = jnp.array(True),
                    xk = opt_var,
                    yk = opt_var,
                    grad_yk = jnp.zeros_like(opt_var))
    return opt_state_init


def init_apg(x0, cost_fn, optimizer_params, proximal_fn=None,
             momentum=None, stepsize=None):
    """Initialize the APG algorithm
    This function will evaluate the cost function at x0 and its gradient
    It is better to use it in a function that is going to be jit-compiled

    Args:
        x0 (TYPE): The initial parameters of the optimization problem
        cost_fn (TYPE): The cost function to minimize
        optimizer_params (TYPE): The parameters for the optimizer
        proximal_fn (None, optional): A function to compute the proximal or projection
                    when the opt variables are subject to contraints. This code
                    assumes only projection on a box constraint for now

    Returns:
        TYPE: an APGState

    """
    # Extract the initial stepsize
    init_step_size = jnp.array(optimizer_params['linesearch']['max_stepsize']\
                        if 'linesearch' in optimizer_params \
                        else optimizer_params['stepsize'])

    # [TODO Franck] Better to project also x0 if proximal function is given
    # This help making sure that we start in the right constrained space
    if proximal_fn is not None:
        x0 = proximal_fn(x0, stepsize=init_step_size)

    # Evaluate the function and its gradient at the current opt variable
    cost_x0, grad_x0 = jax.value_and_grad(cost_fn)(x0)
    grad_x0_nsqr = opt_stop_criteria(grad_x0)

    if momentum is None:
        momentum = jnp.array(optimizer_params.get('beta_init', 0.25))

    if stepsize is None:
        stepsize = init_step_size

    avg_stepsize = stepsize

    opt_state_init = APGState(num_steps = jnp.array(1),
                    momentum = momentum,
                    avg_momentum = momentum,
                    # num_linesearch = jnp.array(1),
                    avg_linesearch = jnp.array(1.),
                    stepsize = stepsize,
                    avg_stepsize = avg_stepsize, # This can be used to warm-up online next mpc calls
                    # init_cost = jnp.abs(cost_x0), # This is used as a scale for terminaison criterion
                    opt_cost = cost_x0,
                    init_cost = cost_x0,
                    grad_sqr = grad_x0_nsqr,
                    not_done = jnp.array(True),
                    xk = x0,
                    yk = x0,
                    grad_yk = grad_x0)

    return opt_state_init

def apg(optim_state, cost_fn, optimizer_params, proximal_fn=None):
    """Compute a one step Accelerated proximal gradient descent given a cost function
        and the optimization parameters such as learning rate, momentum, etc..

    Args:
        cost_fn (TYPE): The cost function to optimize. As
        optim_state (TYPE): The initial state of the optimizer (an instance)
        optimizer_params (TYPE): The parameters of the optimization problems
            - tol: The terminaison criteria based on the norm of gradient
            - max_iter: The maximum of gradient calculations
            - stepsize: The learning rate. if lr <=0 line search is used based on max_stepsize
            - linesearch: A dictionary with the keys:
                - max_stepsize: upper bound on the learning rate
                - coef: ``1-agressiveness`` used for checking the wolfe or improvement condition
                    coef < 0.5 provides a guarantess with Lipschitz constant. low coef -> higher step size
                - decrease_factor: factor to decrease the learning rate.
                - increase_factor: factor to increase the learning rate.
                - maxls: maximum number of steps when doing the backtracking line search
                - reset_option: 'conservative' or increase. The reset strategy at each iteration
                    "conservative": re-use previous stepsize, producing a non increasing
                    sequence of stepsizes. Slow convergence.
                    "increase": attempt to re-use previous stepsize multiplied by
                    increase_factor. Cheap and efficient heuristic.
            - moment_scale: An adaptive coefficient to scale the momentum (it might be unused)
                 if moment_scale <= 0, standard momentum is used. moment_scale in (0,1)
            - beta_init: The initial momentum coeffiicent value -> default is 0.25
        proximal_fn (None, optional): A function to compute the proximal or projection
                    when the opt variables are subject to contraints. This code
                    assumes only projection on a box constraint for now
    """
    assert isinstance(optim_state, APGState), "The input optim_state should be an instance of APG"

    # # Define the loop condition or stopping criteria
    # no_stop = lambda grad_val_crit: jnp.array(grad_val_crit > optimizer_params['tol'])

    # Only iterate at least once -> Reason to enforce the second argument to be true
    #[TODO Franck] Maybe improve it to check a criteria with respect to te current opt_state
    # init_states = optim_state

    def one_step_apg(state_ops):
        """ Define the body function implementing each iteration of APG
        """
        # Extract the variables and current state of the optimizer
        opt_state = state_ops
        xpast, ycurr, cost_ycurr, grad_ycurr = opt_state.xk, opt_state.yk, \
                            opt_state.opt_cost, opt_state.grad_yk

        if 'linesearch' in optimizer_params:
            # Now perform a line search if requested and do one step improvement on ycurr
            _stepsize = opt_state.stepsize
            # # Extract the number of linesearch during the last iteration
            # last_num_search = opt_state.num_linesearch

            # Reset the step size from the previous iteration
            if optimizer_params['linesearch']['reset_option'] == 'increase':
                # Do not increase if the last search required more than one iteration
                # mult_increase = jnp.where(last_num_search > 1, 1., optimizer_params['linesearch']['increase_factor'])
                mult_increase = optimizer_params['linesearch']['increase_factor']
                _stepsize = _stepsize * mult_increase
                _stepsize = jnp.minimum(_stepsize, optimizer_params['linesearch']['max_stepsize'])

            # Compute the new stepsize, current variable and associated cost
            # We augment the decrease factor with a coefficient that depends
            # on the number of linesearch done in the past iterations
            stepsize, xcurr, cost_xcurr, linesearch_numstep = \
                armijo_line_search(cost_fn, ycurr, cost_ycurr, grad_ycurr, opt_state.grad_sqr,
                                        _stepsize, optimizer_params['linesearch'])
        else:
            # If constant learning rate is given
            stepsize = opt_state.stepsize

            # Compute a gradient step given ycurr
            xcurr = ycurr - stepsize * grad_ycurr

            # cost at xcurr is going to be computed if proximal_fn is given
            # so there is no need to re-compute it here
            # We compute it only when proximal_fn is None
            if proximal_fn is None:
                cost_xcurr = cost_fn(xcurr)

            # Only one iteration was enough to find the step size
            linesearch_numstep = 1

        # Apply proximal projection if it is given and compute the resulting cost function
        if proximal_fn is not None:
            xcurr = proximal_fn(xcurr, stepsize=stepsize)
            cost_xcurr = cost_fn(xcurr)

        # Another auxiliary variable to implement to momentum dynamics
        vcurr = xcurr + opt_state.momentum * (xcurr - xpast)
        cost_vcurr = cost_fn(vcurr)

        # Check the non-monotony condition and obtain ynext
        xcurr_less_vcurr = cost_xcurr <= cost_vcurr
        ynext = jnp.where(xcurr_less_vcurr, xcurr, vcurr)

        # Compute ynext grad and cost associated to ynext
        cost_ynext = jnp.where(xcurr_less_vcurr, cost_xcurr, cost_vcurr)
        grad_ynext = jax.grad(cost_fn)(ynext)
        grad_ynext_nsqr = opt_stop_criteria(grad_ynext)

        # Update the states of the optimization problem
        if optimizer_params.get('moment_scale', None) is None: # We don't apply scale
            moment_next = opt_state.num_steps / (opt_state.num_steps + 3.)
        else:
            moment_next = jnp.where(xcurr_less_vcurr,
                            optimizer_params['moment_scale']*opt_state.momentum,
                            jnp.minimum(opt_state.momentum/optimizer_params['moment_scale'], 1.)
                        )
        # Check if we are done
        stop_not_sat = jnp.abs(cost_ynext-cost_ycurr) > optimizer_params['atol'] + optimizer_params['rtol']*jnp.abs(cost_ycurr)

        # Construct the new states of the optimizer
        num_total_step = opt_state.num_steps + 1
        avg_linesearch = (opt_state.num_steps * opt_state.avg_linesearch + linesearch_numstep)/num_total_step
        avg_stepsize = (opt_state.num_steps * opt_state.avg_stepsize + stepsize)/num_total_step
        avg_momentum = (opt_state.num_steps * opt_state.avg_momentum + moment_next)/num_total_step
        opt_state_next = APGState(num_steps = num_total_step,
                                  momentum = moment_next,
                                  avg_momentum = avg_momentum,
                                  avg_linesearch = avg_linesearch,
                                  # num_linesearch = linesearch_numstep,
                                  stepsize = stepsize,
                                  avg_stepsize = avg_stepsize,
                                  opt_cost = cost_ynext,
                                  init_cost = opt_state.init_cost,
                                  # init_cost = opt_state.init_cost,
                                  grad_sqr = grad_ynext_nsqr,
                                  not_done = stop_not_sat,
                                  xk = xcurr,
                                  yk = ynext,
                                  grad_yk = grad_ynext
                                  )
        return opt_state_next # no_stop(grad_ynext_nsqr)

    # _while_loop(cond_fun, body_fun, init_val, max_iter)
    ret = _while_loop(cond_fun = lambda t: t.not_done,  # check boolean violated
                        body_fun=one_step_apg,
                        init_val=optim_state,
                        max_iter=optimizer_params['max_iter']
                    )

    return ret


class APGState(NamedTuple):
    """Named tuple containing state information.
    Attributes:
    num_steps: iteration number
    error: residuals (as gradient) of current estimate
    opt_cost: current value of the cost function
    stepsize: current stepsize
    linesearch_steps: The average number of steps to compute stepsize
    momentum: The momentum coefficient
    """
    num_steps: int
    momentum: float
    avg_momentum: float
    avg_linesearch: float
    # num_linesearch: int
    avg_stepsize: float
    stepsize: float
    opt_cost: float
    init_cost: float
    # init_cost: float
    grad_sqr: float
    not_done: bool
    xk: jnp.ndarray
    yk: jnp.ndarray
    grad_yk: jnp.ndarray


def armijo_line_search(cost_fn, xcurr, f_cur, grad, grad_sqnorm, stepsize, user_params):
    """Perform Armijo (backtracking) Line search from starting opt problem variables,
        learning and gradient at the starting opt variables.

    Args:
        cost_fn: The cost function to optimize
        xcurr: current opt variable to optimize.
        f_cur: value of the cost function at xcurr
        grad: gradient at xcurr.
        stepsize: initial guess for the learning rate.
        user_params: Dictionary
            maxls: maximum number of steps when doing the backtracking line search
            coef: ``1-agressiveness`` used for checking the wolfe condition
            decrease_factor: factor to decrease the learning rate.
            max_stepsize: upper bound on the learning rate

    Returns:
        stepsize: stepsize Armijo line search conditions
        next_params: params after gradient step
        f_next: loss after gradient step
    """
    # Extract the parameters for this function
    coef, decrease_factor, max_stepsize = \
        (user_params[v] for v in ('coef', 'decrease_factor', 'max_stepsize'))

    # compute xnext and the function evaluated at xnext
    xnext = xcurr - stepsize * grad
    f_next = cost_fn(xnext)

    def update_stepsize(_stepsize):
        """Multiply stepsize per factor, return new opt variables and new fun evaluations."""
        _stepsize = jnp.minimum(_stepsize * decrease_factor, max_stepsize)
        _xnext = xcurr - _stepsize * grad
        _f_next = cost_fn(_xnext)
        return _stepsize, _xnext, _f_next

    def body_fun(t):
        """ Body function at each line search iteration
        """
        _stepsize, _xnext, _f_next, num_iter, _ = t

        violated = wolfe_cond_violated(_stepsize, coef, f_cur, _f_next, grad_sqnorm)
        _stepsize, _xnext, _f_next = jax.lax.cond(violated, update_stepsize,
                                                lambda _: (_stepsize, _xnext, _f_next),
                                                operand=_stepsize
                                            )
        num_iter += violated
        return _stepsize, _xnext, _f_next, num_iter, violated

    init_val = stepsize, xnext, f_next, jnp.array(1), jnp.array(True)
    ret = _while_loop(cond_fun=lambda t: t[-1],  # check boolean violated
                        body_fun=body_fun,
                        init_val=init_val, max_iter=user_params['maxls'])
    return ret[:-1] # remove boolean


def wolfe_cond_violated(stepsize, coef, f_cur, f_next, grad_sqnorm):
    """ Check if wolfe condition is violated or not
    """
    eps = jnp.finfo(f_next.dtype).eps
    return stepsize * coef * grad_sqnorm > f_cur - f_next + eps


def _while_loop(cond_fun, body_fun, init_val, max_iter):
  """Scan-based implementation (jit ok, reverse-mode autodiff ok)."""
  def _iter(val):
    next_val = body_fun(val)
    next_cond = cond_fun(next_val)
    return next_val, next_cond

  def _fun(tup, it):
    val, cond = tup
    # When cond is met, we start doing no-ops.
    return jax.lax.cond(cond, _iter, lambda x: (x, False), val), it

  init = (init_val, cond_fun(init_val))
  return jax.lax.scan(_fun, init, None, length=max_iter)[0][0]