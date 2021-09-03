import numpy as np

from .models import NLCP


class OptimizeResult(dict):
    r"""
    Structure the results of the optimization algorithm.
    """

    def __dir__(self):
        r"""
        Return the list of the names of the attributes in the current scope.
        """
        return list(self.keys())

    def __getattr__(self, name):
        r"""
        Return the value of the attribute ``name``. This method raises an
        `AttributeError` exception if such an attribute does not exist.
        """
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, key, value):
        r"""
        Assign the value ``value`` to the attribute ``key``.
        """
        super().__setitem__(key, value)

    def __delattr__(self, key):
        r"""
        Delete the attribute ``key``.
        """
        super().__delitem__(key)

    def __repr__(self):
        r"""
        Return a string representation of an instance of this class, which looks
        like a valid Python expression that can be used to recreate an object
        with the same value (given an appropriate environment).
        """
        attrs = ', '.join(f'{k}={repr(v)}' for k, v in sorted(self.items()))
        return f'{self.__class__.__name__}({attrs})'

    def __str__(self):
        r"""
        Return an informal string representation of an instance of this class,
        which is designed to be nicely printable.
        """
        if self.keys():
            items = sorted(self.items())
            width = max(map(len, self.keys())) + 1
            return '\n'.join(f'{k.rjust(width)}: {str(v)}' for k, v in items)
        else:
            return f'{self.__class__.__name__}()'


def minimize(fun, x0, args=(), xl=None, xu=None, Aub=None, bub=None, Aeq=None,
             beq=None, options=None, **kwargs):
    r"""
    Minimize a real-valued function subject to linear equality and inequality
    constraints using a derivative-free trust-region SQP method.

    Parameters
    ----------
    fun : callable
        Objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an array with shape (n,) and ``args`` is a tuple of
        parameters to pass to the objective function.
    x0 : array_like, shape (n,)
        Initial guess.
    args : tuple, optional
        Parameters to pass to the objective function.
    xl : array_like, shape (n,), optional
        Lower-bound constraints on the decision variables.
    xu : array_like, shape (n,), optional
        Upper-bound constraints on the decision variables.
    Aub : array_like, shape (mub, n), optional
        Jacobian matrix of the linear inequality constraints.
    bub : array_like, shape (mub,), optional
        Right-hand side vector of the linear inequality constraints

            ``\mathtt{Aub} \times x \le \mathtt{bub}``

        where :math:`x` has the same size than ``x0``.
    Aeq : array_like, shape (meq, n), optional
        Jacobian matrix of the linear equality constraints.
    beq : array_like, shape (meq,), optional
        Right-hand side vector of the linear equality constraints

            ``\mathtt{Aeq} \times x = \mathtt{beq}``

        where :math:`x` has the same size than ``x0``.
    options : dict, optional
        Options to pass to the solver. Accepted options are:

            rhobeg : float
                Initial trust-region radius.
                Default is 1.
            rhoend : float
                Final trust-region radius.
                Default is 1e-6.
            npt : int
                Number of interpolation points.
                Default is :math:`2 \times \mathtt{n} + 1`.
            maxfev : int
                Maximum number of function evaluations.
                Default is :math:`500 \times \mathtt{n}`.
            target : float
                Target value of the objective function.
                Default is ``-numpy.inf``.
            disp : bool
                Whether to print information on the execution.
                Default is False.
            debug : bool
                Whether to make debugging tests during the execution.
                Default is False.

    Returns
    -------
    result : OptimizeResult
        Result of the optimization method.

    Other Parameters
    ----------------
    actf : float, optional
        Factor of proximity to the linear constraints.
        Default is 0.2.
    nsf : float, optional
        Shrinkage factor of the Byrd-Omojokun normal subproblem.
        Default is 0.8.
    bdtol : float, optional
        Tolerance for comparisons on the bound constraints.
        Default is ``10 * eps * n * max(1, max(abs(xl)), max(abs(xu)))``. Note
        that the values of ``xl`` and ``xu`` evolve to include the shift of the
        origin, so that the tolerance may vary from an iteration to another.
    lctol : float, optional
        Tolerance for comparisons on the linear constraints.
        Default is ``10 * eps * n * max(1, max(abs(bub)))``. Note that the value
        of ``bub`` evolves to include the shift of the origin, so that the
        tolerance may vary from an iteration to another.
    lstol : float, optional
        Tolerance on the approximate KKT conditions.
        Default is ``10 * eps * max(n, m) * max(1, max(abs(g)))``, where ``g``
        is the gradient of the current model of the objective function, so that
        the tolerance may vary from an iteration to another.

    Notes
    -----
    TODO

    References
    ----------
    TODO
    """
    # Build the initial models of the optimization problem.
    exit_status = 0
    nlc = NLCP(fun, x0, args, xl, xu, Aub, bub, Aeq, beq, options, **kwargs)

    # Begin the iterative procedure.
    nf = nlc.get_opt('npt')
    nit = 0
    rho = nlc.get_opt('rhobeg')
    delta = rho
    knew = -1
    while True:
        # Update the shift of the origin to manage computer rounding errors.
        nlc.shift_origin(delta)

        # Evaluate the trial step.
        delsav = delta
        fsav = nlc.fopt
        xsav = nlc.xopt
        ksav = knew
        nit += 1
        if knew == -1:
            step = nlc.trust_region_step(delta, **kwargs)
            snorm = np.linalg.norm(step)
            if snorm <= .5 * delta:
                delta = rho if delta <= 1.4 * rho else .5 * delta
                if delsav > rho:
                    knew = nlc.get_furthest_point(delta)
                    continue
        else:
            step = nlc.model_step(knew, max(.1 * delta, rho), **kwargs)
            snorm = np.linalg.norm(step)

        if ksav >= 0. or snorm > .5 * delta:
            # Evaluate the objective function, include the trial point in the
            # interpolation set, and update accordingly the models.
            if nf >= nlc.get_opt('maxfev'):
                exit_status = 3
                break
            nf += 1
            knew, mopt, ratio = nlc.update(step, knew, **kwargs)
            if knew == -1:
                exit_status = 9
                break

            # Update the trust-region radius.
            if ksav == -1:
                if ratio <= .1:
                    delta *= .5
                elif ratio <= .7:
                    delta = max(.5 * delta, snorm)
                else:
                    delbd = np.sqrt(2.) * delta
                    delta = min(delbd, max(.2 * delta, 2. * snorm))

            # If a trust-region step has provided a sufficient decrease or if a
            # model-improvement step has just been computed, then the next
            # iteration is a trust-region step.
            if ksav >= 0 or ratio >= .1:
                knew = -1
                continue

            # If an interpolation point is substantially far from the
            # trust-region center, a model-improvement step is entertained.
            knew = nlc.get_furthest_point(max(delta, 2. * rho))
            if knew >= 0 or delsav > rho:
                continue
            msav = nlc(xsav, fsav)
            if mopt < msav:
                continue

        # Update the lower bound on the trust-region radius.
        if rho > nlc.get_opt('rhoend'):
            delta = .5 * rho
            if rho > 2.5e2 * nlc.get_opt('rhoend'):
                rho *= .1
            elif rho <= 1.6e1:
                rho = nlc.get_opt('rhoend')
            else:
                rho = np.sqrt(rho * nlc.get_opt('rhoend'))
            delta = max(delta, rho)
            knew = -1
            if nlc.get_opt('disp'):
                message = f'New trust-region radius: {rho}.'
                _print(nlc, fun.__name__, nf, message)
            continue
        break

    # Build the result structure and return.
    result = OptimizeResult()
    result.x = nlc.xbase + nlc.xopt
    result.fun = nlc.fopt
    result.jac = nlc.obj_grad()
    result.nfev = nf
    result.nit = nit
    if nlc.mub + nlc.meq > 0:
        result.maxcv = nlc.maxcv
    result.status = exit_status
    result.success = exit_status in [0, 1]
    result.message = {
        0: 'Lower bound for the trust-region radius has been reached.',
        1: 'Target function value has been achieved.',
        3: 'Maximum number of function evaluations has been exceeded.',
        9: 'Denominator of the updating formula is zero.',
    }.get(exit_status, 'Unknown exit status.')
    if nlc.get_opt('disp'):
        _print(nlc, fun.__name__, nf, result.message)
    return result


def _print(nlc, fun, nf, message):
    print()
    print(message)
    print(f'Number of function evaluations: {nf}.')
    print(f'Least value of {fun}: {nlc.fopt}.')
    print(f'Corresponding point: {nlc.xbase + nlc.xopt}.')
    print()
