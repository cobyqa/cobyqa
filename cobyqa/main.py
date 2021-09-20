import numpy as np

from .models import TrustRegion
from .utils import RestartRequiredException


class OptimizeResult(dict):
    """
    Result structure of an optimization algorithm.
    """

    def __dir__(self):
        """
        Get the names of the attributes in the current scope.

        Returns
        -------
        list:
            Names of the attributes in the current scope.
        """
        return list(self.keys())

    def __getattr__(self, name):
        """
        Get the value of an attribute that is not explicitly defined.

        Parameters
        ----------
        name : str
            Name of the attribute to be assessed.

        Returns
        -------
        object
            Value of the attribute.

        Raises
        ------
        AttributeError
            The required attribute does not exist.
        """
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, key, value):
        """
        Assign an existing or a new attribute.

        Parameters
        ----------
        key : str
            Name of the attribute to be assigned.
        value : object
            Value of the attribute to be assigned.
        """
        super().__setitem__(key, value)

    def __delattr__(self, key):
        """
        Delete an attribute.

        Parameters
        ----------
        key : str
            Name of the attribute to be deleted.

        Raises
        ------
        KeyError
            The required attribute does not exist.
        """
        super().__delitem__(key)

    def __repr__(self):
        """
        Get a string representation that looks like valid Python expression,
        which can be used to recreate an object with the same value, given an
        appropriate environment.

        Returns
        -------
        str
            String representation of instances of this class.
        """
        attrs = ', '.join(f'{k}={repr(v)}' for k, v in sorted(self.items()))
        return f'{self.__class__.__name__}({attrs})'

    def __str__(self):
        """
        Get a string representation, designed to be nicely printable.

        Returns
        -------
        str
            String representation of instances of this class.
        """
        if self.keys():
            m = max(map(len, self.keys())) + 1
            return '\n'.join(f'{k:>{m}}: {v}' for k, v in sorted(self.items()))
        else:
            return f'{self.__class__.__name__}()'


def minimize(fun, x0, args=(), xl=None, xu=None, Aub=None, bub=None, Aeq=None,
             beq=None, options=None, **kwargs):
    """
    Minimize a real-valued function subject to linear equality and inequality
    constraints using a derivative-free trust-region SQP method.

    Parameters
    ----------
    fun : callable
        Objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an array with shape (n,) and `args` is a tuple of
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
        ``Aub @ x <= bub``, where ``x`` has the same size than `x0`.
    Aeq : array_like, shape (meq, n), optional
        Jacobian matrix of the linear equality constraints.
    beq : array_like, shape (meq,), optional
        Right-hand side vector of the linear equality constraints
        `Aeq @ x = beq`, where ``x`` has the same size than `x0`.
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
                Default is ``2 * n + 1``.
            maxfev : int
                Maximum number of function evaluations.
                Default is ``500 * n``.
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
        that the values of `xl` and `xu` evolve to include the shift of the
        origin, so that the tolerance may vary from an iteration to another.
    lctol : float, optional
        Tolerance for comparisons on the linear constraints.
        Default is ``10 * eps * n * max(1, max(abs(bub)))``. Note that the value
        of `bub` evolves to include the shift of the origin, so that the
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
    fwk = TrustRegion(fun, x0, args, xl, xu, Aub, bub, Aeq, beq, options,
                      **kwargs)

    # Begin the iterative procedure.
    eps = np.finfo(float).eps
    actf = kwargs.get('actf', .2)
    rho = fwk.rhobeg
    delta = rho
    nf = fwk.npt
    nit = 0
    itest = 0
    while True:
        # Update the shift of the origin to manage computer rounding errors.
        fwk.shift_origin(delta)

        # Evaluate the trial step.
        delsav = delta
        fopt = fwk.fopt
        xopt = fwk.xopt
        nit += 1
        is_trust_region_step = not fwk.is_model_step
        if is_trust_region_step:
            step = fwk.trust_region_step(delta, **kwargs)
            snorm = np.linalg.norm(step)
            stf = actf - np.sqrt(eps)
            if snorm <= stf * delta:
                delta = rho if delta <= 1.4 * rho else .5 * delta
                if delsav > rho:
                    fwk.prepare_model_step(delta)
                    continue
        else:
            step = fwk.model_step(max(.1 * delta, rho), **kwargs)
            snorm = np.linalg.norm(step)

        if not is_trust_region_step or snorm > .5 * delta:
            # Evaluate the objective function, include the trial point in the
            # interpolation set, and update accordingly the models.
            if nf >= fwk.maxfev:
                exit_status = 3
                break
            nf += 1
            try:
                mopt, ratio = fwk.update(step, **kwargs)
            except RestartRequiredException:
                continue
            except ZeroDivisionError:
                exit_status = 9
                break

            # Update the trust-region radius.
            if is_trust_region_step:
                if ratio <= .1:
                    delta *= .5
                elif ratio <= .7:
                    delta = max(.5 * delta, snorm)
                else:
                    delbd = np.sqrt(2.) * delta
                    delta = min(delbd, max(.2 * delta, 2. * snorm))
                if delta <= 1.5 * rho:
                    delta = rho

            # Attempt to replace the models by the alternative ones.
            if is_trust_region_step and delta <= rho:
                if ratio > 1e-2:
                    itest = 0
                else:
                    itest += 1
                    obj_grad_norm = np.linalg.norm(fwk.obj_grad(fwk.xopt))
                    alt_grad_norm = np.linalg.norm(fwk.alt_grad(fwk.xopt))
                    if obj_grad_norm < 10. * alt_grad_norm:
                        itest = 0
                    if itest >= 3:
                        fwk.reset_models()
                        itest = 0

            # If a trust-region step has provided a sufficient decrease or if a
            # model-improvement step has just been computed, then the next
            # iteration is a trust-region step.
            if not is_trust_region_step or ratio >= .1:
                fwk.prepare_trust_region_step()
                continue

            # If an interpolation point is substantially far from the
            # trust-region center, a model-improvement step is entertained.
            fwk.prepare_model_step(max(delta, 2. * rho))
            if fwk.is_model_step or delsav > rho:
                continue
            msav = fwk(xopt, fopt)
            if mopt < msav:
                continue

        # Update the lower bound on the trust-region radius.
        if rho > fwk.rhoend:
            delta = .5 * rho
            if rho > 2.5e2 * fwk.rhoend:
                rho *= .1
            elif rho <= 1.6e1:
                rho = fwk.rhoend
            else:
                rho = np.sqrt(rho * fwk.rhoend)
            delta = max(delta, rho)
            fwk.prepare_trust_region_step()
            if fwk.disp:
                message = f'New trust-region radius: {rho}.'
                _print(fwk, fun.__name__, nf, message)
            continue
        break

    # Build the result structure and return.
    result = OptimizeResult()
    result.x = fwk.xbase + fwk.xopt
    result.fun = fwk.fopt
    result.jac = fwk.obj_grad(fwk.xopt)
    result.nfev = nf
    result.nit = nit
    if fwk.type != 'U':
        result.maxcv = fwk.maxcv
    result.status = exit_status
    result.success = exit_status in [0, 1]
    result.message = {
        0: 'Lower bound for the trust-region radius has been reached.',
        1: 'Target function value has been achieved.',
        3: 'Maximum number of function evaluations has been exceeded.',
        9: 'Denominator of the updating formula is zero.',
    }.get(exit_status, 'Unknown exit status.')
    if fwk.disp:
        _print(fwk, fun.__name__, nf, result.message)
    return result


def _print(problem, fun, nf, message):
    print()
    print(message)
    print(f'Number of function evaluations: {nf}.')
    print(f'Least value of {fun}: {problem.fopt}.')
    print(f'Corresponding point: {problem.xbase + problem.xopt}.')
    print()
