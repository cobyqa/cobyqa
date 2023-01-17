import numpy as np

from cobyqa.utils import max_abs_arrays


def bound_constrained_cauchy_step(const, grad, hess_prod, xl, xu, delta, debug):
    r"""
    Maximize approximately the absolute value of a quadratic function subject to
    bound constraints in a trust region.

    This function solves approximately

    .. math::

        \begin{aligned}
            \max_{d \in \R^n}   & \quad \abs[\bigg]{c + g^{\T}d + \frac{1}{2} d^{\T}Hd}\\
            \text{s.t.}         & \quad l \le d \le u,\\
                                & \quad \norm{d} \le \Delta,
        \end{aligned}

    by maximizing the objective function along the constrained Cauchy direction.

    Parameters
    ----------
    const : float
        Constant :math:`c` as shown above.
    grad : numpy.ndarray, shape (n,)
        Gradient :math:`g` as shown above.
    hess_prod : callable
        Product of the Hessian matrix :math:`H` with any vector.

            ``hess_prod(d) -> numpy.ndarray, shape (n,)``

        returns the product :math:`Hd`.
    xl : numpy.ndarray, shape (n,)
        Lower bounds :math:`l` as shown above.
    xu : numpy.ndarray, shape (n,)
        Upper bounds :math:`u` as shown above.
    delta : float
        Trust-region radius :math:`\Delta` as shown above.
    debug : bool
        Whether to make debugging tests during the execution.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Approximate solution :math:`d`.

    Notes
    -----
    This function is described as the first alternative in p. 115 of [1]_. It is
    assumed that the origin is feasible with respect to the bound constraints
    `xl` and `xu`, and that `delta` is finite and positive.

    References
    ----------
    .. [1] T. M. Ragonneau. "Model-Based Derivative-Free Optimization Methods
       and Software." Ph.D. thesis. Hong Kong: Department of Applied
       Mathematics, The Hong Kong Polytechnic University, 2022.
    """
    # Check the feasibility of the subproblem.
    n = grad.size
    tol = 10.0 * np.finfo(float).eps * n * max_abs_arrays(xl, xu)
    if debug:
        assert np.max(xl) <= tol
        assert np.min(xu) >= -tol
        assert np.isfinite(delta) and delta > 0.0
    xl = np.minimum(xl, 0.0)
    xu = np.maximum(xu, 0.0)

    # To maximize the absolute value of a quadratic function, we maximize the
    # function itself or its negative, and we choose the solution that provides
    # the largest function value.
    step1, q_val1 = _bound_constrained_cauchy_step(const, grad, hess_prod, xl, xu, delta, debug)
    step2, q_val2 = _bound_constrained_cauchy_step(-const, -grad, lambda x: -hess_prod(x), xl, xu, delta, debug)
    step = step1 if q_val1 > q_val2 else step2

    if debug:
        assert np.all(xl <= step)
        assert np.all(step <= xu)
        assert np.linalg.norm(step) < 1.1 * delta
    return step


def bound_constrained_xpt_step(const, grad, hess_prod, xpt, xl, xu, delta, debug):
    r"""
    Maximize approximately the absolute value of a quadratic function subject to
    bound constraints in a trust region along specific straight lines.

    This function solves approximately

    .. math::

        \begin{aligned}
            \max_{d \in \R^n}   & \quad \abs[\bigg]{c + g^{\T}d + \frac{1}{2} d^{\T}Hd}\\
            \text{s.t.}         & \quad l \le d \le u,\\
                                & \quad \norm{d} \le \Delta,
        \end{aligned}

    by maximizing the objective function along the straight lines through the
    origin and the rows in `xpt`.

    Parameters
    ----------
    const : float
        Constant :math:`c` as shown above.
    grad : numpy.ndarray, shape (n,)
        Gradient :math:`g` as shown above.
    hess_prod : callable
        Product of the Hessian matrix :math:`H` with any vector.

            ``hess_prod(d) -> numpy.ndarray, shape (n,)``

        returns the product :math:`Hd`.
    xpt : numpy.ndarray, shape (npt, n)
        Points defining the straight lines as shown above.
    xl : numpy.ndarray, shape (n,)
        Lower bounds :math:`l` as shown above.
    xu : numpy.ndarray, shape (n,)
        Upper bounds :math:`u` as shown above.
    delta : float
        Trust-region radius :math:`\Delta` as shown above.
    debug : bool
        Whether to make debugging tests during the execution.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Approximate solution :math:`d`.

    Notes
    -----
    This function is described as the second alternative in p. 115 of [1]_. It
    is assumed that the origin is feasible with respect to the bound constraints
    `xl` and `xu`, and that `delta` is finite and positive.

    References
    ----------
    .. [1] T. M. Ragonneau. "Model-Based Derivative-Free Optimization Methods
       and Software." Ph.D. thesis. Hong Kong: Department of Applied
       Mathematics, The Hong Kong Polytechnic University, 2022.
    """
    # Check the feasibility of the subproblem.
    npt, n = xpt.shape
    if debug:
        tol = 10.0 * np.finfo(float).eps * n * max_abs_arrays(xl, xu)
        assert np.max(xl) <= tol
        assert np.min(xu) >= -tol
        assert np.isfinite(delta) and delta > 0.0
    xl = np.minimum(xl, 0.0)
    xu = np.maximum(xu, 0.0)

    # Iterate through the straight lines.
    step = np.zeros_like(grad)
    q_val = const
    for k in range(npt):
        # Set alpha_tr to the step size for the trust-region constraint.
        s_norm = np.linalg.norm(xpt[k, :])
        if s_norm > np.finfo(float).tiny * delta:
            alpha_tr = max(delta / s_norm, 0.0)
        else:
            continue

        # Set alpha_xl to the step size for the lower-bound constraint and
        # alpha_xu to the step size for the upper-bound constraint.
        i_xl_pos = (xl > -np.inf) & (xpt[k, :] > -np.finfo(float).tiny * xl)
        i_xl_neg = (xl > -np.inf) & (xpt[k, :] < np.finfo(float).tiny * xl)
        i_xu_pos = (xu < np.inf) & (xpt[k, :] > np.finfo(float).tiny * xu)
        i_xu_neg = (xu < np.inf) & (xpt[k, :] < -np.finfo(float).tiny * xu)
        alpha_xl_pos = np.max(xl[i_xl_pos] / xpt[k, i_xl_pos], initial=-np.inf)
        alpha_xl_neg = np.max(xu[i_xu_neg] / xpt[k, i_xu_neg], initial=-np.inf)
        alpha_xu_pos = np.min(xu[i_xu_pos] / xpt[k, i_xu_pos], initial=np.inf)
        alpha_xu_neg = np.min(xl[i_xl_neg] / xpt[k, i_xl_neg], initial=np.inf)
        alpha_xl = max(alpha_xl_pos, alpha_xl_neg)
        alpha_xu = min(alpha_xu_pos, alpha_xu_neg)

        # Set alpha_pos to the step size for the maximization problem without
        # any constraint along the positive direction, and alpha_neg to the step
        # size for the maximization problem along the negative direction.
        grad_step = np.inner(grad, xpt[k, :])
        hess_step = hess_prod(xpt[k, :])
        curv_step = np.inner(xpt[k, :], hess_step)
        if grad_step >= 0.0 and curv_step < -np.finfo(float).tiny * grad_step or grad_step <= 0.0 and curv_step > -np.finfo(float).tiny * grad_step:
            alpha_pos = max(-grad_step / curv_step, 0.0)
        else:
            alpha_pos = np.inf
        if grad_step >= 0.0 and curv_step > np.finfo(float).tiny * grad_step or grad_step <= 0.0 and curv_step < np.finfo(float).tiny * grad_step:
            alpha_neg = min(-grad_step / curv_step, 0.0)
        else:
            alpha_neg = -np.inf

        # Compute the constrained counterparts of alpha_pos and alpha_neg, and
        # accept the one that provides the largest absolute value of the
        # objective function if it improves the current best.
        alpha_pos = min(alpha_pos, alpha_tr, alpha_xu)
        alpha_neg = max(alpha_neg, -alpha_tr, alpha_xl)
        q_val_pos = const + alpha_pos * grad_step + 0.5 * alpha_pos ** 2.0 * curv_step
        q_val_neg = const + alpha_neg * grad_step + 0.5 * alpha_neg ** 2.0 * curv_step
        if abs(q_val_pos) >= abs(q_val_neg) and abs(q_val_pos) > abs(q_val):
            step = np.maximum(xl, np.minimum(alpha_pos * xpt[k, :], xu))
            q_val = q_val_pos
        elif abs(q_val_neg) > abs(q_val_pos) and abs(q_val_neg) > abs(q_val):
            step = np.maximum(xl, np.minimum(alpha_neg * xpt[k, :], xu))
            q_val = q_val_neg

    if debug:
        assert np.all(xl <= step)
        assert np.all(step <= xu)
        assert np.linalg.norm(step) < 1.1 * delta
    return step


def _bound_constrained_cauchy_step(const, grad, hess_prod, xl, xu, delta, debug):
    """
    Same as `bound_constrained_cauchy_step` without the absolute value.
    """
    # Calculate the initial active set.
    fixed_xl = (xl < 0.0) & (grad > 0.0)
    fixed_xu = (xu > 0.0) & (grad < 0.0)

    # Calculate the Cauchy step.
    cauchy_step = np.zeros_like(grad)
    cauchy_step[fixed_xl] = xl[fixed_xl]
    cauchy_step[fixed_xu] = xu[fixed_xu]
    if np.linalg.norm(cauchy_step) > delta:
        working = fixed_xl | fixed_xu
        while True:
            # Calculate the Cauchy step for the directions in the working set.
            g_norm = np.linalg.norm(grad[working])
            delta_reduced = np.sqrt(delta ** 2.0 - np.inner(cauchy_step[~working], cauchy_step[~working]))
            if g_norm > np.finfo(float).tiny * abs(delta_reduced):
                mu = max(delta_reduced / g_norm, 0.0)
            else:
                break
            cauchy_step[working] = mu * grad[working]

            # Update the working set.
            fixed_xl = working & (cauchy_step < xl)
            fixed_xu = working & (cauchy_step > xu)
            if not np.any(fixed_xl) and not np.any(fixed_xu):
                # Stop the calculations as the Cauchy step is now feasible.
                break
            cauchy_step[fixed_xl] = xl[fixed_xl]
            cauchy_step[fixed_xu] = xu[fixed_xu]
            working = working & ~(fixed_xl | fixed_xu)

    # Calculate the step that maximizes the quadratic along the Cauchy step.
    grad_step = np.inner(grad, cauchy_step)
    if grad_step >= 0.0:
        # Set alpha_tr to the step size for the trust-region constraint.
        s_norm = np.linalg.norm(cauchy_step)
        if s_norm > np.finfo(float).tiny * delta:
            alpha_tr = max(delta / s_norm, 0.0)
        else:
            # The Cauchy step is basically zero.
            alpha_tr = 0.0

        # Set alpha_quad to the step size for the maximization problem.
        hess_step = hess_prod(cauchy_step)
        curv_step = np.inner(cauchy_step, hess_step)
        if curv_step < -np.finfo(float).tiny * grad_step:
            alpha_quad = max(-grad_step / curv_step, 0.0)
        else:
            alpha_quad = np.inf

        # Set alpha_bd to the step size for the bound constraints.
        i_xl = (xl > -np.inf) & (cauchy_step < np.finfo(float).tiny * xl)
        i_xu = (xu < np.inf) & (cauchy_step > np.finfo(float).tiny * xu)
        alpha_xl = np.min(xl[i_xl] / cauchy_step[i_xl], initial=np.inf)
        alpha_xu = np.min(xu[i_xu] / cauchy_step[i_xu], initial=np.inf)
        alpha_bd = min(alpha_xl, alpha_xu)

        # Calculate the solution and the corresponding function value.
        alpha = min(alpha_tr, alpha_quad, alpha_bd)
        step = np.maximum(xl, np.minimum(alpha * cauchy_step, xu))
        q_val = const + alpha * grad_step + 0.5 * alpha ** 2.0 * curv_step
    else:
        # This case is never reached in exact arithmetic. It prevents this
        # function to return a step that decreases the objective function.
        step = np.zeros_like(grad)
        q_val = const

    if debug:
        assert np.all(xl <= step)
        assert np.all(step <= xu)
        assert np.linalg.norm(step) < 1.1 * delta
    return step, q_val
