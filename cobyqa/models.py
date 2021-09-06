import warnings

import numpy as np

from .linalg import bvcs, bvlag, bvtcg, cpqp, givens, lctcg, nnls

EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny


class NLCP:
    r"""
    Represent the states of a nonlinear constrained problem.
    """

    def __init__(self, fun, x0, args=(), xl=None, xu=None, Aub=None, bub=None,
                 Aeq=None, beq=None, options=None, **kwargs):
        r"""
        Initialize the states of the nonlinear constrained problem.
        """
        self._fun = fun
        x = np.array(x0, dtype=float)
        n = x.size
        if not isinstance(args, tuple):
            args = (args,)
        self._args = args
        if xl is None:
            xl = np.full_like(x, -np.inf)
        self._xl = np.array(xl, dtype=float)
        if xu is None:
            xu = np.full_like(x, np.inf)
        self._xu = np.array(xu, dtype=float)
        if Aub is None:
            Aub = np.empty((0, n))
        self._Aub = np.array(Aub, dtype=float)
        if bub is None:
            bub = np.empty(0)
        self._bub = np.array(bub, dtype=float)
        if Aeq is None:
            Aeq = np.empty((0, n))
        self._Aeq = np.array(Aeq, dtype=float)
        if beq is None:
            beq = np.empty(0)
        self._beq = np.array(beq, dtype=float)
        if options is None:
            options = {}
        self._opts = dict(options)
        self.set_default_options(n)
        self.check_options(n)

        # Modify the initial guess in order to avoid conflicts between the
        # bounds and the first quadratic models. The initial components of the
        # initial guess should either equal bound components or allow the
        # projection of the initial trust region onto the components to lie
        # entirely inside the bounds.
        rhobeg = self.get_opt('rhobeg')
        rhoend = self.get_opt('rhoend')
        rhobeg = min(.5 * np.min(self._xu - self._xl), rhobeg)
        rhoend = min(rhobeg, rhoend)
        self._opts.update({'rhobeg': rhobeg, 'rhoend': rhoend})
        adj = (x - self._xl <= rhobeg) & (self._xl < x)
        if np.any(adj):
            x[adj] = self._xl[adj] + rhobeg
        adj = (self._xu - x <= rhobeg) & (x < self._xu)
        if np.any(adj):
            x[adj] = self._xu[adj] - rhobeg

        # Set the initial shift of the origin, designed to manage the effects
        # of computer rounding errors in the calculations, and update
        # accordingly the right-hand sides of the constraints at most linear.
        self._xbase = x
        self.shift_constraints(self._xbase)

        # Set the initial models of the problem.
        self._mds = Model(self.fun, self._xbase, self._xl, self._xu, self._opts)
        if self.get_opt('debug'):
            self.check_models()

        # Determine the initial least-squares multipliers of the problem.
        self._gub = 1.
        self._geq = 1.
        self._lmub = np.zeros_like(bub, dtype=float)
        self._lmeq = np.zeros_like(beq, dtype=float)
        self.update_multipliers(**kwargs)

        # Evaluate the merit function at the interpolation points and
        # determine the optimal point so far and update the initial models.
        npt = self.get_opt('npt')
        mval = np.empty(npt, dtype=float)
        for k in range(npt):
            mval[k] = self(self.xpt[k, :], self.fval[k])
        self.kopt = np.argmin(mval)
        if self.get_opt('debug'):
            self.check_models()

    def __call__(self, x, fx, model=False):
        r"""
        Evaluate the merit functions at ``x``. If ``model = True`` is provided,
        the method also returns the value of the merit function corresponding to
        the modeled problem.
        """
        ax = fx
        mx = self.fopt
        if abs(self._gub) > TINY * np.max(np.abs(self._lmub), initial=0.):
            tub = np.dot(self._Aub, x) - self._bub + self._lmub / self._gub
            tub = np.maximum(0., tub)
            alub = .5 * self._gub * np.inner(tub, tub)
            ax += alub
            mx += alub
        if abs(self._geq) > TINY * np.max(np.abs(self._lmeq), initial=0.):
            teq = np.dot(self._Aeq, x) - self._beq + self._lmeq / self._geq
            aleq = .5 * self._geq * np.inner(teq, teq)
            ax += aleq
            mx += aleq
        if model:
            mx += self.obj(x)
            return ax, mx
        return ax

    @property
    def xl(self):
        r"""
        Return the lower-bound constraints on the decision variables.
        """
        return np.copy(self._xl)

    @property
    def xu(self):
        r"""
        Return the upper-bound constraints on the decision variables.
        """
        return np.copy(self._xu)

    @property
    def aub(self):
        r"""
        Return the Jacobian matrix of the linear inequality constraints.
        """
        return np.copy(self._Aub)

    @property
    def bub(self):
        r"""
        Return the right-hand side vector of the linear inequality constraints.
        """
        return np.copy(self._bub)

    @property
    def mub(self):
        r"""
        Return the number of linear inequality constraints.
        """
        return self._bub.size

    @property
    def aeq(self):
        r"""
        Return the Jacobian matrix of the linear equality constraints.
        """
        return np.copy(self._Aeq)

    @property
    def beq(self):
        r"""
        Return the right-hand side vector of the linear equality constraints.
        """
        return np.copy(self._beq)

    @property
    def meq(self):
        r"""
        Return the number of linear equality constraints.
        """
        return self._beq.size

    @property
    def options(self):
        r"""
        Return the option passed to the solver.
        """
        return dict(self._opts)

    @property
    def xbase(self):
        r"""
        Return the shift of the origin in the calculations.
        """
        return np.copy(self._xbase)

    @property
    def gub(self):
        r"""
        Returns the penalty coefficient for the inequality constraints.
        """
        return self._gub

    @property
    def geq(self):
        r"""
        Returns the penalty coefficient for the equality constraints.
        """
        return self._geq

    @property
    def lmub(self):
        r"""
        Returns the Lagrange multipliers for the inequality constraints.
        """
        return np.copy(self._lmub)

    @property
    def lmeq(self):
        r"""
        Returns the Lagrange multipliers for the equality constraints.
        """
        return np.copy(self._lmeq)

    @property
    def xpt(self):
        r"""
        Return the interpolation points.
        """
        return self._mds.xpt

    @property
    def fval(self):
        r"""
        Return the values of the objective function at the interpolation points.
        """
        return self._mds.fval

    @property
    def kopt(self):
        r"""
        Return the index of the best point so far.
        """
        return self._mds.kopt

    @kopt.setter
    def kopt(self, knew):
        r"""
        Set the index of the best point so far.
        """
        self._mds.kopt = knew

    @property
    def xopt(self):
        r"""
        Return the best point so far.
        """
        return self._mds.xopt

    @property
    def fopt(self):
        r"""
        Return the value of the objective function at the best point so far.
        """
        return self._mds.fopt

    @property
    def maxcv(self):
        r"""
        Return the constraint violation at the best point so far.
        """
        cub = np.dot(self._Aub, self.xopt) - self._bub
        ceq = np.dot(self._Aeq, self.xopt) - self._beq
        cb = np.r_[self.xopt - self._xu, self._xl - self.xopt]
        return np.max(np.r_[cub, np.abs(ceq), cb], initial=0.)

    @property
    def type(self):
        # CUTEst classification scheme
        # https://www.cuter.rl.ac.uk/Problems/classification.shtml
        if self.mub + self.meq > 0:
            return 'L'
        elif np.any(self._xl != -np.inf) or np.any(self._xu != np.inf):
            # TODO: All variables may be fixed.
            return 'B'
        else:
            return 'U'

    def fun(self, x):
        r"""
        Evaluate the objective function at ``x``.
        """
        fx = float(self._fun(x, *self._args))
        if self.get_opt('disp'):
            print(f'{self._fun.__name__}({x}) = {fx}.')
        return fx

    def obj(self, x=None):
        r"""
        Evaluate the objective function of the model at ``x``. If ``x`` is None,
        it is evaluated at ``self.xopt``.
        """
        return self._mds.obj(x)

    def obj_grad(self, x=None):
        r"""
        Evaluate the gradient of the objective function of the model at ``x``.
        If ``x`` is None, the gradient is evaluated at ``self.xopt``.
        """
        return self._mds.obj_grad(x)

    def obj_hessp(self, x):
        r"""
        Evaluate the product of the Hessian matrix of the objective function of
        the model and ``x``.
        """
        return self._mds.obj_hessp(x)

    def obj_curv(self, x):
        r"""
        Evaluate the curvature of the objective function of the model at ``x``.
        """
        return self._mds.obj_hessp(x)

    def get_opt(self, option, default=None):
        r"""
        Return the value of an option passed to the solver.
        """
        return self._opts.get(option, default)

    def set_default_options(self, n):
        r"""
        Set the default options of the solvers.
        """
        self._opts.setdefault('rhobeg', max(1., self.get_opt('rhoend', 0.)))
        self._opts.setdefault('rhoend', min(1e-6, self.get_opt('rhobeg')))
        self._opts.setdefault('npt', 2 * n + 1)
        self._opts.setdefault('maxfev', max(500 * n, self.get_opt('npt') + 1))
        self._opts.setdefault('target', -np.inf)
        self._opts.setdefault('disp', False)
        self._opts.setdefault('debug', False)

    def check_options(self, n, stack_level=2):
        r"""
        Set the options passed to the solvers.
        """
        # Ensure that the option 'npt' is in the required interval.
        npt_min = n + 2
        npt_max = (n + 1) * (n + 2) // 2
        npt = self.get_opt('npt')
        if not (npt_min <= npt <= npt_max):
            self._opts['npt'] = min(npt_max, max(npt_min, npt))
            message = "Option 'npt' is not in the required interval and is "
            message += 'increased.' if npt_min > npt else 'decreased.'
            warnings.warn(message, RuntimeWarning, stacklevel=stack_level)

        # Ensure that the option 'maxfev' is large enough.
        maxfev = self.get_opt('maxfev')
        if maxfev <= self.get_opt('npt'):
            self._opts['maxfev'] = self.get_opt('npt') + 1
            if maxfev <= npt:
                message = "Option 'maxfev' is too low and is increased."
            else:
                message = "Option 'maxfev' is correspondingly increased."
            warnings.warn(message, RuntimeWarning, stacklevel=stack_level)

        # Ensure that the options 'rhobeg' and 'rhoend' are consistent.
        if self.get_opt('rhoend') > self.get_opt('rhobeg'):
            self._opts['rhoend'] = self.get_opt('rhobeg')
            message = "Option 'rhoend' is too large and is decreased."
            warnings.warn(message, RuntimeWarning, stacklevel=stack_level)

    def shift_constraints(self, x):
        r"""
        Shift the bound constraints and the right-hand sides of the linear
        inequality and equality constraints.
        """
        # Note: When adapting the algorithm to tackle general nonlinear
        # constraints, the right-hand sides of the new constraints may be
        # undefined when this method is called.
        self._xl -= x
        self._xu -= x
        self._bub -= np.dot(self._Aub, x)
        self._beq -= np.dot(self._Aeq, x)

    def get_furthest_point(self, delta):
        r"""
        Get the index of the further point from ``self.xopt`` if the
        corresponding distance is more than ``delta``, -1 otherwise.
        """
        dsq = np.sum((self.xpt - self.xopt[np.newaxis, :]) ** 2., axis=1)
        dsq[dsq <= delta ** 2.] = -np.inf
        knew = -1
        if np.any(np.isfinite(dsq)):
            knew = np.argmax(dsq)
        return knew

    def shift_origin(self, delta):
        r"""
        Update the shift of the origin if necessary.
        """
        xoptsq = np.inner(self.xopt, self.xopt)

        # Update the shift from the origin only if the displacement from the
        # shift of the best point is substantial in the trust region.
        if xoptsq >= 1e1 * delta ** 2.:
            # Update the models of the problem to include the new shift.
            xold = self.xopt
            self._mds.shift_origin()

            # Update the right hand sides of the constraints and the bounds.
            self.shift_constraints(xold)

            # Complete the shift by updating the shift itself.
            self._xbase += xold
            if self.get_opt('debug'):
                self.check_models()

    def update(self, step, knew, **kwargs):
        r"""
        Update the model to include the trial point in the interpolation set.
        """
        # Evaluate the objective function at the trial point.
        xsav = self.xopt
        xnew = xsav + step
        fx = self.fun(self._xbase + xnew)

        # Update the Lagrange multipliers and the penalty parameters.
        self.update_multipliers(**kwargs)
        mx, mmx, mopt = self.update_penalty_coefficients(xnew, fx, knew)

        # Determine the trust-region ratio.
        if knew == -1 and abs(mopt - mmx) > TINY * abs(mopt - mx):
            ratio = (mopt - mx) / (mopt - mmx)
        else:
            ratio = -1.

        # Update the models of the problem. The step is updated to take into
        # account the fact that the best point so far may have been updated when
        # the penalty coefficients have been updated.
        step += xsav - self.xopt
        knew = self._mds.update(step, knew, fx)
        if knew >= 0:
            if mx < mopt:
                self.kopt = knew
                mopt = mx
            if self.get_opt('debug'):
                self.check_models()
        return knew, mopt, ratio

    def update_multipliers(self, **kwargs):
        r"""
        Update the least-squares Lagrange multipliers.
        """
        if self._bub.size + self._beq.size > 0:
            # Determine the matrix of the least-squares problem. The inequality
            # multipliers corresponding to nonzero constraint values are set to
            # zeros to satisfy the complementary slackness conditions.
            tol = EPS * self._bub.size * np.max(np.abs(self._bub), initial=1.)
            iub = np.less_equal(np.abs(self._bub), tol)
            mub = np.count_nonzero(iub)
            A = np.r_[self._Aub[iub, :], self._Aeq].T

            # Determine the least-squares Lagrange multipliers that have not
            # been fixed by the complementary slackness conditions.
            lm, _ = nnls(A, -self.obj_grad(), mub, **kwargs)
            self._lmub.fill(0.)
            self._lmub[iub] = lm[:mub]
            self._lmeq = lm[mub:]

    def update_penalty_coefficients(self, xnew, fx, knew):
        mx, mmx = self(xnew, fx, True)
        mopt = self(self.xopt, self.fopt)
        if knew == -1 and mmx > mopt:
            npt = self.get_opt('npt')
            mval = np.empty(npt, dtype=float)
            while mmx > mopt:
                self._gub *= 2.
                self._geq *= 2.
                mx, mmx = self(xnew, fx, True)
                for k in range(npt):
                    mval[k] = self(self.xpt[k, :], self.fval[k])
                self.kopt = np.argmin(mval)
                mopt = mval[self.kopt]
        return mx, mmx, mopt

    def trust_region_step(self, delta, **kwargs):
        r"""
        Evaluate a Byrd-Omojokun-like trust-region step.

        Notes
        -----
        The trust-region constraint of the tangential subproblem is not centered
        if the normal step is nonzero. To cope with this difficulty, we use the
        result in Equation (15.4.3) of [1]_.

        References
        ----------
        .. [1] A. R. Conn, N. I. M. Gould, and Ph. L. Toint. Trust-Region
        Methods. MPS-SIAM Ser. Optim. Philadelphia, PA, US: SIAM, 2009.
        """
        # Define the tolerances to compare floating-point numbers with zero.
        tol = 1e1 * EPS * self.xopt.size

        # Evaluate the normal step of the Byrd-Omojokun approach. The normal
        # step attempts to reduce the violations of the linear constraints
        # subject to the bound constraints and a trust-region constraint. The
        # trust-region radius is shrunk to leave some elbow room to the
        # tangential subproblem for the computations whenever the trust-region
        # subproblem is infeasible.
        delta *= np.sqrt(.5)
        nsf = kwargs.get('nsf', .8)
        mc = self._bub.size + self._beq.size
        if mc == 0:
            nstep = np.zeros_like(self.xopt)
            ssq = 0.
        else:
            nstep = cpqp(self.xopt, self._Aub, self._bub, self._Aeq, self._beq,
                         self._xl, self._xu, nsf * delta, **kwargs)
            ssq = np.inner(nstep, nstep)

        # Evaluate the tangential step of the trust-region subproblem, and set
        # the global trust-region step. The tangential subproblem is feasible.
        if ssq <= tol * max(delta, 1.):
            xopt = self.xopt
            delta *= np.sqrt(2.)
        else:
            xopt = self.xopt + nstep
            delta = np.sqrt(delta ** 2. - ssq)
        gq = self.obj_grad(xopt)
        bub = np.maximum(self._bub, np.dot(self._Aub, xopt))
        beq = np.dot(self._Aeq, xopt)
        if mc == 0:
            tstep = bvtcg(xopt, gq, self.obj_hessp, (), self._xl, self._xu,
                          delta, **kwargs)
        else:
            tstep = lctcg(xopt, gq, self.obj_hessp, (), self._Aub, bub,
                          self._Aeq, beq, self._xl, self._xu, delta, **kwargs)
        return nstep + tstep

    def model_step(self, knew, delta, **kwargs):
        r"""
        Evaluate a model-improvement step.
        TODO: Give details.
        """
        return self._mds.model_step(knew, self._xl, self._xu, delta, **kwargs)

    def check_models(self, stack_level=2):
        r"""
        Check whether the models satisfy the interpolation conditions.
        """
        self._mds.check_models(stack_level)


class Model:
    r"""
    Representation of the model of the optimization problem.
    """

    def __init__(self, fun, xbase, xl, xu, options):
        r"""
        Construct the initial model of the optimization problem. The quadratic
        models of the nonlinear functions are obtained by underdetermined
        interpolation, where the freedom bequeathed by the interpolation
        conditions is taken up by minimizing the Hessian matrices of the models
        in Frobenius norm.
        """
        xbase = np.asarray(xbase)
        if xbase.dtype.kind in np.typecodes['AllInteger']:
            xbase = np.asarray(xbase, dtype=float)
        xl = np.asarray(xl)
        if xl.dtype.kind in np.typecodes['AllInteger']:
            xl = np.asarray(xl, dtype=float)
        xu = np.asarray(xu)
        if xu.dtype.kind in np.typecodes['AllInteger']:
            xu = np.asarray(xu, dtype=float)
        n = xbase.size
        npt = options.get('npt')
        rhobeg = options.get('rhobeg')
        self._xpt = np.zeros((npt, n))
        self._fval = np.empty(npt, dtype=float)
        self._bmat = np.zeros((npt + n, n), dtype=float)
        self._zmat = np.zeros((npt, npt - n - 1), dtype=float)
        self._idz = 0
        self._kopt = 0
        stepa = 0.
        stepb = 0.
        for k in range(npt):
            km = k - 1
            kx = km - n

            # Set the displacement from the shift of the origin of the
            # components of the initial interpolation vectors.
            if 1 <= k <= n:
                if abs(xu[km]) <= .5 * rhobeg:
                    stepa = -rhobeg
                else:
                    stepa = rhobeg
                self._xpt[k, km] = stepa
            elif n < k <= 2 * n:
                stepa = self._xpt[kx + 1, kx]
                if abs(xl[kx]) <= .5 * rhobeg:
                    stepb = min(2. * rhobeg, xu[kx])
                elif abs(xu[kx]) <= .5 * rhobeg:
                    stepb = max(-2. * rhobeg, xl[kx])
                else:
                    stepb = -rhobeg
                self._xpt[k, kx] = stepb
            elif k > 2 * n:
                shift = kx // n
                ipt = kx - shift * n
                jpt = (ipt + shift) % n
                self._xpt[k, ipt] = self._xpt[ipt + 1, ipt]
                self._xpt[k, jpt] = self._xpt[jpt + 1, jpt]

            # Evaluate the objective function at the interpolation points. The
            # values are interpreted as floating-point numbers by NumPy
            # automatically when stored.
            self._fval[k] = fun(xbase + self._xpt[k, :])

            # Set the initial elements of the KKT matrix of interpolation.
            if k <= 2 * n:
                if 1 <= k <= n and npt <= k + n:
                    self._bmat[0, km] = -1 / stepa
                    self._bmat[k, km] = 1 / stepa
                    self._bmat[npt + km, km] = -.5 * rhobeg ** 2.
                elif k > n:
                    self._bmat[0, kx] = -(stepa + stepb) / (stepa * stepb)
                    self._bmat[k, kx] = -.5 / self._xpt[kx + 1, kx]
                    self._bmat[kx + 1, kx] = -self._bmat[0, kx]
                    self._bmat[kx + 1, kx] -= self._bmat[k, kx]
                    self._zmat[0, kx] = np.sqrt(2.) / (stepa * stepb)
                    self._zmat[k, kx] = np.sqrt(.5) / rhobeg ** 2.
                    self._zmat[kx + 1, kx] = -self._zmat[0, kx]
                    self._zmat[kx + 1, kx] -= self._zmat[k, kx]
            else:
                shift = kx // n
                ipt = kx - shift * n
                jpt = (ipt + shift) % n
                self._zmat[0, kx] = 1. / rhobeg ** 2.
                self._zmat[k, kx] = 1. / rhobeg ** 2.
                self._zmat[ipt + 1, kx] = -1. / rhobeg ** 2.
                self._zmat[jpt + 1, kx] = -1. / rhobeg ** 2.

        self._obj = Quadratic(self._bmat, self._zmat, self._idz, self._fval)

    @property
    def xpt(self):
        r"""
        Return the interpolation points.
        """
        return np.copy(self._xpt)

    @property
    def fval(self):
        r"""
        Return the values of the objective function at the interpolation points.
        """
        return np.copy(self._fval)

    @property
    def bmat(self):
        r"""
        Return the last columns of the KKT matrix of interpolation.
        """
        return np.copy(self._bmat)

    @property
    def zmat(self):
        r"""
        Return the factorization matrix of the leading submatrix of the KKT
        matrix of interpolation.
        """
        return np.copy(self._zmat)

    @property
    def idz(self):
        r"""
        Return the index of inversion of the factorization matrix.
        """
        return self._idz

    @property
    def kopt(self):
        r"""
        Return the index of the best point so far.
        """
        return self._kopt

    @kopt.setter
    def kopt(self, knew):
        r"""
        Set the index of the best point so far.
        """
        if self._kopt != knew:
            xold = self.xopt
            self._kopt = knew
            self._obj.shift(self.xopt - xold, self._xpt)

    @property
    def xopt(self):
        r"""
        Return the best point so far.
        """
        return np.copy(self._xpt[self._kopt, :])

    @property
    def fopt(self):
        r"""
        Return the value of the objective function at the best point so far.
        """
        return self._fval[self._kopt]

    @staticmethod
    def omega_prod(zmat, idz, x):
        r"""
        Perform the product of the leading submatrix of the KKT matrix of
        interpolation and ``x``. If ``x`` is an integer, it is understood as
        the ``x``-th standard unit vector.
        """
        if isinstance(x, (int, np.integer)):
            temp = np.r_[-zmat[x, :idz], zmat[x, idz:]]
        else:
            temp = np.dot(np.c_[-zmat[:, :idz], zmat[:, idz:]].T, x)
        return np.dot(zmat, temp)

    def obj(self, x=None):
        r"""
        Evaluate the objective function of the model at ``x``. If ``x`` is None,
        it is evaluated at ``self.xopt``.
        """
        return 0. if x is None else self._obj(x, self._xpt, self._kopt)

    def obj_grad(self, x=None):
        r"""
        Evaluate the gradient of the objective function of the model at ``x``.
        If ``x`` is None, the gradient is evaluated at ``self.xopt``.
        """
        return self._obj.grad(x, self._xpt, self._kopt)

    def obj_hessp(self, x):
        r"""
        Evaluate the product of the Hessian matrix of the objective function of
        the model and ``x``.
        """
        return self._obj.hessp(x, self._xpt)

    def obj_curv(self, x):
        r"""
        Evaluate the curvature of the objective function of the model at ``x``.
        """
        return self._obj.curv(x, self._xpt)

    def shift_origin(self):
        r"""
        Update the model when the shift of the origin is modified.
        """
        xopt = self.xopt
        npt, n = self._xpt.shape
        xoptsq = np.inner(xopt, xopt)

        # Make the changes to the columns of the KKT matrix of interpolation
        # that do not depend on the factorization of the leading submatrix.
        qoptsq = .25 * xoptsq
        updt = np.dot(self._xpt, self.xopt) - .5 * xoptsq
        self._xpt -= .5 * xopt[np.newaxis, :]
        for k in range(npt):
            step = updt[k] * self._xpt[k, :] + qoptsq * self.xopt
            temp = np.outer(self._bmat[k, :], step)
            self._bmat[npt:, :] += temp + temp.T

        # Calculate the remaining revisions of the matrix.
        temp = qoptsq * np.outer(xopt, np.sum(self._zmat, axis=0))
        temp += np.matmul(self._xpt.T, self._zmat * updt[:, np.newaxis])
        for k in range(self._idz):
            self._bmat[:npt, :] -= np.outer(self._zmat[:, k], temp[:, k])
            self._bmat[npt:, :] -= np.outer(temp[:, k], temp[:, k])
        for k in range(self._idz, npt - n - 1):
            self._bmat[:npt, :] += np.outer(self._zmat[:, k], temp[:, k])
            self._bmat[npt:, :] += np.outer(temp[:, k], temp[:, k])

        # Complete the shift by updating the models.
        self._obj.shift_origin(self._xpt, xopt)
        self._xpt -= .5 * xopt[np.newaxis, :]

    def update(self, step, knew, fx):
        r"""
        Update KKT matrix of interpolation and the model to replace the
        ``knew``-th interpolation point by ``self.xopt + step``. If
        ``knew = -1``, the index is selected by maximizing the absolute value of
        the weighted denominator in Equation (2.12) of Powell (2004).
        """
        npt, n = self._xpt.shape

        # Evaluate the Lagrange polynomials of the interpolation points and the
        # real parameter given in Equation (2.13) of Powell (2004).
        vlag = self._lagrange(step)
        vlag[self._kopt] += 1
        beta, updt = self._beta(step)

        # Select the index of the interpolation point to be deleted if it has
        # not been provided. It is picked by maximizing the absolute value of
        # the weighted denominator in Equation (2.12) of Powell (2004).
        if knew == -1:
            zsq = self._zmat ** 2.
            zsq = np.c_[-zsq[:, :self._idz], zsq[:, self._idz:]]
            alpha = np.sum(zsq, axis=1)
            sigma = vlag ** 2. + beta * alpha
            dsq = np.sum((self._xpt - self.xopt[np.newaxis, :]) ** 2., axis=1)
            knew = np.argmax(np.abs(sigma) * np.square(dsq))

        # Apply a sequence of Givens rotations to the factorization of the
        # leading submatrix of the KKT matrix of interpolation.
        jdz = 0
        for j in range(1, npt - n - 1):
            if j == self._idz:
                jdz = self._idz
            elif abs(self._zmat[knew, j]) > 0.:
                cval = self._zmat[knew, jdz]
                sval = self._zmat[knew, j]
                givens(self._zmat, cval, sval, j, jdz, 1)
                self._zmat[knew, j] = 0.

        # Evaluate the remaining parameters of Equation (2.12) of Powell (2004).
        scala = self._zmat[knew, 0] if self._idz == 0 else -self._zmat[knew, 0]
        scalb = 0. if jdz == 0 else self._zmat[knew, jdz]
        omega = scala * self._zmat[:, 0] + scalb * self._zmat[:, jdz]
        alpha = omega[knew]
        tau = vlag[knew]
        sigma = alpha * beta + tau ** 2.
        vlag[knew] -= 1.
        bmax = np.max(np.abs(self._bmat), initial=1.)
        zmax = np.max(np.abs(self._zmat), initial=1.)
        if abs(sigma) < TINY * max(bmax, zmax):
            # The denominator of the updating formula is too small to safely
            # divide the coefficients of the KKT matrix of interpolation.
            # Theoretically, the value of abs(sigma) is always positive, and
            # becomes small only for ill-conditioned problems.
            return -1

        # Complete the updating of the factorization matrix of the leading
        # submatrix of the KKT matrix of interpolation.
        reduce = False
        hval = np.sqrt(abs(sigma))
        if jdz == 0:
            scala = tau / hval
            scalb = self._zmat[knew, 0] / hval
            self._zmat[:, 0] = scala * self._zmat[:, 0] - scalb * vlag
            if sigma < 0.:
                if self._idz == 0:
                    self._idz = 1
                else:
                    reduce = True
        else:
            kdz = jdz if beta >= 0. else 0
            jdz -= kdz
            tempa = self._zmat[knew, jdz] * beta / sigma
            tempb = self._zmat[knew, jdz] * tau / sigma
            temp = self._zmat[knew, kdz]
            scala = 1. / np.sqrt(abs(beta) * temp ** 2. + tau ** 2.)
            scalb = scala * hval
            self._zmat[:, kdz] = tau * self._zmat[:, kdz] - temp * vlag
            self._zmat[:, kdz] *= scala
            self._zmat[:, jdz] -= tempa * omega + tempb * vlag
            self._zmat[:, jdz] *= scalb
            if sigma <= 0.:
                if beta < 0.:
                    self._idz += 1
                else:
                    reduce = True
        if reduce:
            self._idz -= 1
            self._zmat[:, [0, self._idz]] = self._zmat[:, [self._idz, 0]]

        # Update accordingly the remaining components of the KKT matrix of
        # interpolation. The copy below is crucial, as the slicing would
        # otherwise return a view of it only.
        bsav = np.copy(self._bmat[knew, :])
        for j in range(n):
            cosv = (alpha * updt[j] - tau * bsav[j]) / sigma
            sinv = (tau * updt[j] + beta * bsav[j]) / sigma
            self._bmat[:npt, j] += cosv * vlag - sinv * omega
            self._bmat[npt:npt + j + 1, j] += cosv * updt[:j + 1]
            self._bmat[npt:npt + j + 1, j] -= sinv * bsav[:j + 1]
            self._bmat[npt + j, :j + 1] = self._bmat[npt:npt + j + 1, j]

        # Update the models of the problem.
        xnew = self.xopt + step
        xold = np.copy(self._xpt[knew, :])
        diff = fx - self.fopt - self.obj(xnew)
        self._fval[knew] = fx
        self._xpt[knew, :] = xnew
        self._obj.update(self._xpt, self._kopt, xold, self._bmat, self._zmat,
                         self._idz, knew, diff)
        return knew

    def reset_models(self):
        r"""
        Reset the quadratic models of the nonlinear functions to the ones whose
        Hessian matrices are least in Frobenius norm.
        """
        self._obj = Quadratic(self._bmat, self._zmat, self._idz, self._fval)
        self._obj.shift(self.xopt, self._xpt)

    def model_step(self, knew, xl, xu, delta, **kwargs):
        r"""
        Evaluate a model-improvement step.
        TODO: Give details.
        """
        # Define the tolerances to compare floating-point numbers with zero.
        npt = self._xpt.shape[0]
        tol = EPS * npt

        # Define the knew-th Lagrange polynomial at the best point.
        lag = Quadratic(self._bmat, self._zmat, self._idz, knew)
        lag.shift(self.xopt, self._xpt)

        # Determine a point on a line between the optimal point and the other
        # interpolation points, chosen to maximize the absolute value of the
        # knew-th Lagrange polynomial, defined above.
        omega = self.omega_prod(self._zmat, self._idz, knew)
        alpha = omega[knew]
        step = bvlag(self._xpt, self._kopt, knew, lag.grad(), xl, xu, delta,
                     alpha, **kwargs)

        # Evaluate the constrained Cauchy step from the optimal point of the
        # absolute value of the knew-th Lagrange polynomial.
        salt, cauchy = bvcs(self._xpt, self._kopt, lag.grad(), lag.curv,
                            (self._xpt,), xl, xu, delta, **kwargs)

        # Among the two computed alternative points, we choose the one leading
        # to the greatest value of sigma in Equation (2.13) or Powell (2004).
        vlag = lag(self.xopt + step, self._xpt, self._kopt)
        beta, _ = self._beta(step)
        sigma = vlag ** 2. + alpha * beta
        if sigma < cauchy and cauchy > tol * max(1, abs(sigma)):
            step = salt
        return step

    def _lagrange(self, step):
        r"""
        Evaluate the Lagrange polynomials at ``self.xopt + step``.
        """
        # Evaluate the inner products required by the computations below.
        npt = self._xpt.shape[0]
        xstep = np.dot(self._xpt, step)
        xxopt = np.dot(self._xpt, self.xopt)

        # Evaluate the Lagrange polynomials at the interpolation points.
        check = np.multiply(xstep, .5 * xstep + xxopt)
        vlag = np.dot(self._bmat[:npt, :], step)
        vlag += self.omega_prod(self._zmat, self._idz, check)

        return vlag

    def _beta(self, step):
        r"""
        Evaluate the real parameter sigma in Equation (2.13) or Powell (2004).
        """
        npt = self._xpt.shape[0]
        stepsq = np.inner(step, step)
        xoptsq = np.inner(self.xopt, self.xopt)
        stx = np.inner(step, self.xopt)
        xstep = np.dot(self._xpt, step)
        xxopt = np.dot(self._xpt, self.xopt)
        check = xstep * (.5 * xstep + xxopt)
        zalt = np.c_[-self._zmat[:, :self._idz], self._zmat[:, self._idz:]]
        temp = np.dot(zalt.T, check)
        beta = np.inner(temp[:self._idz], temp[:self._idz])
        beta -= np.inner(temp[self._idz:], temp[self._idz:])
        updt = np.dot(self._bmat[:npt, :].T, check)
        bsp = np.inner(updt, step)
        updt += np.dot(self._bmat[npt:, :], step)
        bsp += np.inner(updt, step)
        beta += stx ** 2. + stepsq * (xoptsq + 2. * stx + .5 * stepsq) - bsp
        return beta, updt

    def check_models(self, stack_level=2):
        r"""
        Check whether the models satisfy the interpolation conditions.
        """
        stack_level += 1
        self._obj.check_model(self._xpt, self._fval, self._kopt, stack_level)


class Quadratic:
    r"""
    Representation of a quadratic function.
    """

    def __init__(self, bmat, zmat, idz, fval):
        r"""
        Construct the quadratic function. If ``fval`` is an integer, it builds
        the ``fval``-th Lagrange polynomial.
        """
        bmat = np.asarray(bmat)
        if bmat.dtype.kind in np.typecodes['AllInteger']:
            bmat = np.asarray(bmat, dtype=float)
        zmat = np.asarray(zmat)
        if zmat.dtype.kind in np.typecodes['AllInteger']:
            zmat = np.asarray(zmat, dtype=float)
        npt = zmat.shape[0]
        if isinstance(fval, (int, np.integer)):
            # Build the fval-th Lagrange quadratic model.
            self._gq = np.copy(bmat[fval, :])
        else:
            # Build a generic quadratic function.
            fval = np.asarray(fval)
            if fval.dtype.kind in np.typecodes['AllInteger']:
                fval = np.asarray(fval, dtype=float)
            self._gq = np.dot(bmat[:npt, :].T, fval)
        self._pq = Model.omega_prod(zmat, idz, fval)
        self._hq = None

    def __call__(self, x, xpt, kopt):
        r"""
        Evaluate the quadratic function at ``x``.
        """
        x = x - xpt[kopt, :]
        qx = np.inner(self._gq, x)
        qx += .5 * np.inner(self._pq, np.dot(xpt, x) ** 2.)
        if self._hq is not None:
            qx += .5 * np.inner(x, np.dot(self._hq, x))
        return qx

    def grad(self, x=None, xpt=None, kopt=None):
        r"""
        Evaluate the gradient of the quadratic function at ``x``. If ``x`` is
        None, the gradient is evaluated at ``xpt[kopt, :]``.
        """
        gx = np.copy(self._gq)
        if x is not None:
            x = x - xpt[kopt, :]
            gx += self.hessp(x, xpt)
        return gx

    def hessp(self, x, xpt):
        r"""
        Evaluate the product of the Hessian of the quadratic function and ``x``.
        """
        hx = np.dot(xpt.T, self._pq * np.dot(xpt, x))
        if self._hq is not None:
            hx += np.dot(self._hq, x)
        return hx

    def curv(self, x, xpt):
        r"""
        Evaluate the curvature of the quadratic function at ``x``.
        """
        cx = np.inner(self._pq, np.dot(xpt, x) ** 2.)
        if self._hq is not None:
            cx += np.inner(x, np.dot(self._hq, x))
        return cx

    def shift(self, step, xpt):
        r"""
        Shift the evaluation points of the model from ``step``.
        """
        self._gq += self.hessp(step, xpt)

    def shift_origin(self, hxpt, xopt):
        r"""
        Update the model when the shift of the origin is modified.
        """
        temp = np.outer(np.dot(hxpt.T, self._pq), xopt)
        if self._hq is None:
            self._hq = np.zeros((xopt.size, xopt.size), dtype=float)
        self._hq += temp + temp.T

    def update(self, xpt, kopt, xold, bmat, zmat, idz, knew, diff):
        r"""
        Update the model when the KKT matrix of interpolation is modified.
        """
        # Update the explicit and implicit Hessian matrices of the model.
        omega = Model.omega_prod(zmat, idz, knew)
        if self._hq is None:
            self._hq = np.zeros((xold.size, xold.size), dtype=float)
        self._hq += self._pq[knew] * np.outer(xold, xold)
        self._pq[knew] = 0.
        self._pq += diff * omega

        # Update the gradient of the model.
        temp = omega * np.dot(xpt, xpt[kopt, :])
        self._gq += diff * (bmat[knew, :] + np.dot(xpt.T, temp))

    def check_model(self, xpt, fval, kopt, stack_level=2):
        r"""
        Check whether the model satisfies the interpolation conditions.
        """
        tol = 1e1 * np.sqrt(EPS) * fval.size * np.max(np.abs(fval), initial=1.)
        diff = 0.
        for k in range(fval.size):
            qx = self(xpt[k, :], xpt, kopt)
            diff = max(diff, abs(qx + fval[kopt] - fval[k]))
        if diff > tol:
            stack_level += 1
            message = f'error in interpolation conditions is {diff:e}.'
            warnings.warn(message, RuntimeWarning, stacklevel=stack_level)
