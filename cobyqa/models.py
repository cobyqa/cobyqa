import warnings

import numpy as np

from .linalg import bvcs, bvlag, bvtcg, cpqp, givens, lctcg, nnls
from .utils import RestartRequiredException, omega_product

EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny


class TrustRegion:
    r"""
    Represent the states of a nonlinear constrained problem.
    """

    def __init__(self, fun, x0, args=(), xl=None, xu=None, Aub=None, bub=None,
                 Aeq=None, beq=None, cub=None, ceq=None, options=None,
                 **kwargs):
        r"""
        Initialize the states of the nonlinear constrained problem.
        """
        self._fun = fun
        x0 = np.array(x0, dtype=float)
        n = x0.size
        if not isinstance(args, tuple):
            args = (args,)
        self._args = args
        if xl is None:
            xl = np.full_like(x0, -np.inf)
        xl = np.array(xl, dtype=float)
        if xu is None:
            xu = np.full_like(x0, np.inf)
        xu = np.array(xu, dtype=float)
        if Aub is None:
            Aub = np.empty((0, n))
        Aub = np.array(Aub, dtype=float)
        if bub is None:
            bub = np.empty(0)
        bub = np.array(bub, dtype=float)
        if Aeq is None:
            Aeq = np.empty((0, n))
        Aeq = np.array(Aeq, dtype=float)
        if beq is None:
            beq = np.empty(0)
        beq = np.array(beq, dtype=float)
        if options is None:
            options = {}
        self._cub = cub
        self._ceq = ceq
        self._options = dict(options)
        self.set_default_options(n)
        self.check_options(n)

        # Project the initial guess onto the bound constraints.
        x0 = np.minimum(xu, np.maximum(xl, x0))

        # Modify the initial guess in order to avoid conflicts between the
        # bounds and the first quadratic models. The initial components of the
        # initial guess should either equal bound components or allow the
        # projection of the initial trust region onto the components to lie
        # entirely inside the bounds.
        rhobeg = self.rhobeg
        rhoend = self.rhoend
        rhobeg = min(.5 * np.min(xu - xl), rhobeg)
        rhoend = min(rhobeg, rhoend)
        self._options.update({'rhobeg': rhobeg, 'rhoend': rhoend})
        adj = (x0 - xl <= rhobeg) & (xl < x0)
        if np.any(adj):
            x0[adj] = xl[adj] + rhobeg
        adj = (xu - x0 <= rhobeg) & (x0 < xu)
        if np.any(adj):
            x0[adj] = xu[adj] - rhobeg

        # Set the initial shift of the origin, designed to manage the effects
        # of computer rounding errors in the calculations, and update
        # accordingly the right-hand sides of the constraints at most linear.
        self._xbase = x0

        # Set the initial models of the problem.
        self._models = Models(self.fun, self._xbase, xl, xu, Aub, bub, Aeq, beq,
                              self.cub, self.ceq, self._options)
        if self.debug:
            self.check_models()

        # Determine the initial least-squares multipliers of the problem.
        self._penub = 0.
        self._peneq = 0.
        self._lmlub = np.zeros_like(bub)
        self._lmleq = np.zeros_like(beq)
        self._lmnlub = np.zeros(self.mnlub, dtype=float)
        self._lmnleq = np.zeros(self.mnleq, dtype=float)
        self.update_multipliers(**kwargs)

        # Evaluate the merit function at the interpolation points and
        # determine the optimal point so far and update the initial models.
        # self.kopt = self.get_best_point()
        npt = self.npt
        mval = np.empty(npt, dtype=float)
        for k in range(npt):
            mval[k] = self(self.xpt[k, :], self.fval[k], self.cvalub[k, :],
                           self.cvaleq[k, :])
        self.kopt = np.argmin(mval)
        if self.debug:
            self.check_models()

        # The initial step is a trust-region step.
        self._knew = None

    def __call__(self, x, fx, cubx, ceqx, model=False):
        r"""
        Evaluate the merit functions at ``x``. If ``model = True`` is provided,
        the method also returns the value of the merit function corresponding to
        the modeled problem.
        """
        ax = fx
        mx = 0.
        if abs(self._penub) > TINY * np.max(np.abs(self._lmlub), initial=0.):
            tub = np.dot(self.aub, x) - self.bub + self._lmlub / self._penub
            tub = np.maximum(0., tub)
            alub = .5 * self._penub * np.inner(tub, tub)
            ax += alub
            mx += alub
        lmnlub_max = np.max(np.abs(self._lmnlub), initial=0.)
        if abs(self._penub) > TINY * lmnlub_max:
            tub = cubx + self._lmnlub / self._penub
            tub = np.maximum(0., tub)
            ax += .5 * self._penub * np.inner(tub, tub)
        if abs(self._peneq) > TINY * np.max(np.abs(self._lmleq), initial=0.):
            teq = np.dot(self.aeq, x) - self.beq + self._lmleq / self._peneq
            aleq = .5 * self._peneq * np.inner(teq, teq)
            ax += aleq
            mx += aleq
        lmnleq_max = np.max(np.abs(self._lmnleq), initial=0.)
        if abs(self._peneq) > TINY * lmnleq_max:
            teq = ceqx + self._lmnleq / self._peneq
            ax += .5 * self._peneq * np.inner(teq, teq)
        if model:
            mx += self.model_obj(x)
            if abs(self._penub) > TINY * lmnlub_max:
                tub = self._lmnlub / self._penub
                for i in range(self.mnlub):
                    tub[i] += self.model_cub(x, i)
                tub = np.maximum(0., tub)
                mx += .5 * self._penub * np.inner(tub, tub)
            if abs(self._peneq) > TINY * lmnleq_max:
                teq = self._lmnleq / self._peneq
                for i in range(self.mnleq):
                    teq[i] += self.model_ceq(x, i)
                mx += .5 * self._peneq * np.inner(teq, teq)
            return ax, mx
        return ax

    def __getattr__(self, item):
        try:
            return self._options[item]
        except KeyError as e:
            raise AttributeError(item) from e

    @property
    def xbase(self):
        r"""
        Return the shift of the origin in the calculations.
        """
        return self._xbase

    @property
    def options(self):
        r"""
        Return the option passed to the solver.
        """
        return self._options

    @property
    def penub(self):
        r"""
        Returns the penalty coefficient for the inequality constraints.
        """
        return self._penub

    @property
    def peneq(self):
        r"""
        Returns the penalty coefficient for the equality constraints.
        """
        return self._peneq

    @property
    def lmlub(self):
        return self._lmlub

    @property
    def lmleq(self):
        return self._lmleq

    @property
    def lmnlub(self):
        return self._lmnlub

    @property
    def lmnleq(self):
        return self._lmnleq

    @property
    def knew(self):
        return self._knew

    @property
    def xl(self):
        return self._models.xl

    @property
    def xu(self):
        return self._models.xu

    @property
    def aub(self):
        return self._models.aub

    @property
    def bub(self):
        return self._models.bub

    @property
    def mlub(self):
        return self._models.mlub

    @property
    def aeq(self):
        return self._models.aeq

    @property
    def beq(self):
        return self._models.beq

    @property
    def mleq(self):
        return self._models.mleq

    @property
    def xpt(self):
        r"""
        Return the interpolation points.
        """
        return self._models.xpt

    @property
    def fval(self):
        r"""
        Return the values of the objective function at the interpolation points.
        """
        return self._models.fval

    @property
    def rval(self):
        return self._models.rval

    @property
    def cvalub(self):
        return self._models.cvalub

    @property
    def mnlub(self):
        return self._models.mnlub

    @property
    def cvaleq(self):
        return self._models.cvaleq

    @property
    def mnleq(self):
        return self._models.mnleq

    @property
    def kopt(self):
        r"""
        Return the index of the best point so far.
        """
        return self._models.kopt

    @kopt.setter
    def kopt(self, knew):
        r"""
        Set the index of the best point so far.
        """
        self._models.kopt = knew

    @property
    def xopt(self):
        r"""
        Return the best point so far.
        """
        return self._models.xopt

    @property
    def fopt(self):
        r"""
        Return the value of the objective function at the best point so far.
        """
        return self._models.fopt

    @property
    def maxcv(self):
        r"""
        Return the constraint violation at the best point so far.
        """
        return self._models.ropt

    @property
    def coptub(self):
        return self._models.coptub

    @property
    def copteq(self):
        return self._models.copteq

    @property
    def type(self):
        return self._models.type

    @property
    def is_model_step(self):
        return self._knew is not None

    def fun(self, x):
        fx = float(self._fun(x, *self._args))
        if self.disp:
            print(f'{self._fun.__name__}({x}) = {fx}.')
        return fx

    def cub(self, x):
        return self._eval_con(self._cub, x)

    def ceq(self, x):
        return self._eval_con(self._ceq, x)

    def model_obj(self, x):
        return self._models.obj(x)

    def model_obj_grad(self, x):
        return self._models.obj_grad(x)

    def model_obj_hess(self):
        return self._models.obj_hess()

    def model_obj_hessp(self, x):
        return self._models.obj_hessp(x)

    def model_obj_curv(self, x):
        return self._models.obj_curv(x)

    def model_obj_alt(self, x):
        return self._models.obj_alt(x)

    def model_obj_alt_grad(self, x):
        return self._models.obj_alt_grad(x)

    def model_obj_alt_hess(self):
        return self._models.obj_alt_hess()

    def model_obj_alt_hessp(self, x):
        return self._models.obj_alt_hessp(x)

    def model_obj_alt_curv(self, x):
        return self._models.obj_alt_curv(x)

    def model_cub(self, x, i):
        return self._models.cub(x, i)

    def model_cub_grad(self, x, i):
        return self._models.cub_grad(x, i)

    def model_cub_hess(self, i):
        return self._models.cub_hess(i)

    def model_cub_hessp(self, x, i):
        return self._models.cub_hessp(x, i)

    def model_cub_curv(self, x, i):
        return self._models.cub_curv(x, i)

    def model_cub_alt(self, x, i):
        return self._models.cub_alt(x, i)

    def model_cub_alt_grad(self, x, i):
        return self._models.cub_alt_grad(x, i)

    def model_cub_alt_hess(self, i):
        return self._models.cub_alt_hess(i)

    def model_cub_alt_hessp(self, x, i):
        return self._models.cub_alt_hessp(x, i)

    def model_cub_alt_curv(self, x, i):
        return self._models.cub_alt_curv(x, i)

    def model_ceq(self, x, i):
        return self._models.ceq(x, i)

    def model_ceq_grad(self, x, i):
        return self._models.ceq_grad(x, i)

    def model_ceq_hess(self, i):
        return self._models.ceq_hess(i)

    def model_ceq_hessp(self, x, i):
        return self._models.ceq_hessp(x, i)

    def model_ceq_curv(self, x, i):
        return self._models.ceq_curv(x, i)

    def model_ceq_alt(self, x, i):
        return self._models.ceq_alt(x, i)

    def model_ceq_alt_grad(self, x, i):
        return self._models.ceq_alt_grad(x, i)

    def model_ceq_alt_hess(self, i):
        return self._models.ceq_alt_hess(i)

    def model_ceq_alt_hessp(self, x, i):
        return self._models.ceq_alt_hessp(x, i)

    def model_ceq_alt_curv(self, x, i):
        return self._models.ceq_alt_curv(x, i)

    def model_lag(self, x):
        return self._models.lag(x, self._lmlub, self._lmleq, self._lmnlub,
                                self._lmnleq)

    def model_lag_grad(self, x):
        return self._models.lag_grad(x, self._lmlub, self._lmleq, self._lmnlub,
                                     self._lmnleq)

    def model_lag_hess(self):
        return self._models.lag_hess(self._lmnlub, self._lmnleq)

    def model_lag_hessp(self, x):
        r"""
        Evaluate the product of the Hessian matrix of the Lagrangian function of
        the model and ``x``.
        """
        return self._models.lag_hessp(x, self._lmnlub, self._lmnleq)

    def set_default_options(self, n):
        r"""
        Set the default options of the solvers.
        """
        rhoend = getattr(self, 'rhoend', 1e-6)
        self._options.setdefault('rhobeg', max(1., rhoend))
        self._options.setdefault('rhoend', min(rhoend, self.rhobeg))
        self._options.setdefault('npt', 2 * n + 1)
        self._options.setdefault('maxfev', max(500 * n, self.npt + 1))
        self._options.setdefault('target', -np.inf)
        self._options.setdefault('disp', False)
        self._options.setdefault('debug', False)

    def check_options(self, n, stack_level=2):
        r"""
        Set the options passed to the solvers.
        """
        # Ensure that the option 'npt' is in the required interval.
        npt_min = n + 2
        npt_max = (n + 1) * (n + 2) // 2
        npt = self.npt
        if not (npt_min <= npt <= npt_max):
            self._options['npt'] = min(npt_max, max(npt_min, npt))
            message = "Option 'npt' is not in the required interval and is "
            message += 'increased.' if npt_min > npt else 'decreased.'
            warnings.warn(message, RuntimeWarning, stacklevel=stack_level)

        # Ensure that the option 'maxfev' is large enough.
        maxfev = self.maxfev
        if maxfev <= self.npt:
            self._options['maxfev'] = self.npt + 1
            if maxfev <= npt:
                message = "Option 'maxfev' is too low and is increased."
            else:
                message = "Option 'maxfev' is correspondingly increased."
            warnings.warn(message, RuntimeWarning, stacklevel=stack_level)

        # Ensure that the options 'rhobeg' and 'rhoend' are consistent.
        if self.rhoend > self.rhobeg:
            self._options['rhoend'] = self.rhobeg
            message = "Option 'rhoend' is too large and is decreased."
            warnings.warn(message, RuntimeWarning, stacklevel=stack_level)

    def get_best_point(self):
        kopt = self.kopt
        mopt = self(self.xopt, self.fopt, self.coptub, self.copteq)
        for k in range(self.npt):
            if k != kopt:
                mval = self(self.xpt[k, :], self.fval[k], self.cvalub[k, :],
                            self.cvaleq[k, :])
                if self.less_merit(mval, self.rval[k], mopt, self.rval[kopt]):
                    kopt = k
                    mopt = mval
        return kopt

    def prepare_trust_region_step(self):
        self._knew = None

    def prepare_model_step(self, delta):
        r"""
        Get the index of the further point from ``self.xopt`` if the
        corresponding distance is more than ``delta``, -1 otherwise.
        """
        dsq = np.sum((self.xpt - self.xopt[np.newaxis, :]) ** 2., axis=1)
        dsq[dsq <= delta ** 2.] = -np.inf
        if np.any(np.isfinite(dsq)):
            self._knew = np.argmax(dsq)
        else:
            self._knew = None

    def less_merit(self, mval1, rval1, mval2, rval2):
        tol = 10. * EPS * self.npt * max(1., abs(mval2))
        if mval1 < mval2:
            return True
        elif max(self._penub, self._peneq) < tol:
            if abs(mval1 - mval2) <= tol and rval1 < rval2:
                return True
        return False

    def shift_origin(self, delta):
        r"""
        Update the shift of the origin if necessary.
        """
        xoptsq = np.inner(self.xopt, self.xopt)

        # Update the shift from the origin only if the displacement from the
        # shift of the best point is substantial in the trust region.
        if xoptsq >= 10. * delta ** 2.:
            # Update the models of the problem to include the new shift.
            self._xbase += self.xopt
            self._models.shift_origin()
            if self.debug:
                self.check_models()

    def update(self, step, **kwargs):
        r"""
        Update the model to include the trial point in the interpolation set.
        """
        # Evaluate the objective function at the trial point.
        xsav = np.copy(self.xopt)
        xnew = xsav + step
        fx = self.fun(self._xbase + xnew)
        cubx = self.cub(self._xbase + xnew)
        ceqx = self.ceq(self._xbase + xnew)

        # Update the Lagrange multipliers and the penalty parameters.
        self.update_multipliers(**kwargs)
        ksav = self.kopt
        mx, mmx, mopt = self.update_penalty_coefficients(xnew, fx, cubx, ceqx)
        if ksav != self.kopt:
            self.prepare_trust_region_step()
            raise RestartRequiredException

        # Determine the trust-region ratio.
        if not self.is_model_step and abs(mopt - mmx) > TINY * abs(mopt - mx):
            ratio = (mopt - mx) / (mopt - mmx)
        else:
            ratio = -1.

        # Update the models of the problem. The step is updated to take into
        # account the fact that the best point so far may have been updated when
        # the penalty coefficients have been updated.
        step += xsav - self.xopt
        rx = self._models.resid(xnew, cubx, ceqx)
        self._knew = self._models.update(step, fx, cubx, ceqx, self._knew)
        if self.less_merit(mx, rx, mopt, self.maxcv):
            self.kopt = self._knew
            mopt = mx
        if self.debug:
            self.check_models()
        return mopt, ratio

    def update_multipliers(self, **kwargs):
        r"""
        Update the least-squares Lagrange multipliers.
        """
        n = self.xopt.size
        if self.mlub + self.mnlub + self.mleq + self.mnleq > 0:
            # Determine the matrix of the least-squares problem. The inequality
            # multipliers corresponding to nonzero constraint values are set to
            # zeros to satisfy the complementary slackness conditions.
            tol = 10. * EPS * self.mlub * np.max(np.abs(self.bub), initial=1.)
            rub = np.dot(self.aub, self.xopt) - self.bub
            ilub = np.less_equal(np.abs(rub), tol)
            mlub = np.count_nonzero(ilub)
            rub = self.coptub
            tol = 10. * EPS * self.mlub * np.max(np.abs(rub), initial=1.)
            cub_jac = np.empty((self.mnlub, n), dtype=float)
            for i in range(self.mnlub):
                cub_jac[i, :] = self.model_cub_grad(self.xopt, i)
                cub_jac[i, :] -= self.model_cub_hessp(self.xopt, i)
            inlub = np.less_equal(np.abs(rub), tol)
            mnlub = np.count_nonzero(inlub)
            ceq_jac = np.empty((self.mnleq, n), dtype=float)
            for i in range(self.mnleq):
                ceq_jac[i, :] = self.model_ceq_grad(self.xopt, i)
                ceq_jac[i, :] -= self.model_ceq_hessp(self.xopt, i)
            A = np.r_[self.aub[ilub, :], cub_jac[inlub, :], self.aeq, ceq_jac].T

            # Determine the least-squares Lagrange multipliers that have not
            # been fixed by the complementary slackness conditions.
            gopt = self.model_obj_grad(self.xopt)
            lm, _ = nnls(A, -gopt, mlub + mnlub, **kwargs)
            self._lmlub.fill(0.)
            self._lmnlub.fill(0.)
            self._lmlub[ilub] = lm[:mlub]
            self._lmnlub[inlub] = lm[mlub:mlub + mnlub]
            self._lmleq = lm[mlub + mnlub:mlub + mnlub + self.mleq]
            self._lmnleq = lm[mlub + mnlub + self.mleq:]

    def update_penalty_coefficients(self, xnew, fx, cubx, ceqx):
        mx, mmx = self(xnew, fx, cubx, ceqx, True)
        mopt = -np.inf
        if not self.is_model_step and mmx > mopt:
            ksav = self.kopt
            while ksav == self.kopt and mmx > mopt:
                if self._penub > 0.:
                    self._penub *= 2.
                elif self.mlub + self.mnlub > 0:
                    self._penub = 1.
                if self._peneq > 0.:
                    self._peneq *= 2.
                elif self.mleq + self.mnleq > 0:
                    self._peneq = 1.
                mx, mmx = self(xnew, fx, cubx, ceqx, True)
                self.kopt = self.get_best_point()
                mopt = self(self.xopt, self.fopt, self.coptub, self.copteq)
        else:
            mopt = self(self.xopt, self.fopt, self.coptub, self.copteq)
        return mx, mmx, mopt

    def reduce_penalty_coefficients(self):
        fmin = np.min(self.fval)
        fmax = np.max(self.fval)
        if self._penub > 0.:
            resid = np.dot(self.xpt, self.aub.T) - self.bub[np.newaxis, :]
            resid = np.c_[resid, self.cvalub]
            cmin = np.min(resid, axis=1, initial=0.)
            cmax = np.max(resid, axis=1, initial=0.)
            iub = np.less(cmin, 2. * cmax)
            cmin[iub] = np.minimum(0., cmin[iub])
            if np.any(iub):
                denom = np.min(cmax[iub] - cmin[iub])
                self._penub = (fmax - fmin) / denom
            else:
                self._penub = 0.
        if self._peneq > 0.:
            resid = np.dot(self.xpt, self.aeq.T) - self.beq[np.newaxis, :]
            resid = np.c_[resid, self.cvaleq]
            cmin = np.min(resid, axis=1, initial=0.)
            cmax = np.max(resid, axis=1, initial=0.)
            ieq = (cmin < 2. * cmax) | (cmin < .5 * cmax)
            cmax[ieq] = np.maximum(0., cmax[ieq])
            cmin[ieq] = np.minimum(0., cmin[ieq])
            if np.any(ieq):
                denom = np.min(cmax[ieq] - cmin[ieq])
                self._peneq = (fmax - fmin) / denom
            else:
                self._peneq = 0.

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
        mc = self.mlub + self.mnlub + self.mleq + self.mnleq
        aub = np.copy(self.aub)
        bub = np.copy(self.bub)
        for i in range(self.mnlub):
            lhs = self.model_cub_grad(self.xopt, i)
            rhs = np.inner(self.xopt, lhs) - self.coptub[i]
            lhs -= self.model_cub_hessp(self.xopt, i)
            rhs -= .5 * self.model_cub_curv(self.xopt, i)
            aub = np.vstack([aub, lhs])
            bub = np.r_[bub, rhs]
        aeq = np.copy(self.aeq)
        beq = np.copy(self.beq)
        for i in range(self.mnleq):
            lhs = self.model_ceq_grad(self.xopt, i)
            rhs = np.inner(self.xopt, lhs) - self.copteq[i]
            lhs -= self.model_ceq_hessp(self.xopt, i)
            rhs -= .5 * self.model_ceq_curv(self.xopt, i)
            aeq = np.vstack([aeq, lhs])
            beq = np.r_[beq, rhs]
        if mc == 0:
            nstep = np.zeros_like(self.xopt)
            ssq = 0.
        else:
            nstep = cpqp(self.xopt, aub, bub, aeq, beq, self.xl, self.xu,
                         nsf * delta, **kwargs)
            ssq = np.inner(nstep, nstep)

        # Evaluate the tangential step of the trust-region subproblem, and set
        # the global trust-region step. The tangential subproblem is feasible.
        if np.sqrt(ssq) <= tol * max(delta, 1.):
            nstep = np.zeros_like(self.xopt)
            delta *= np.sqrt(2.)
        else:
            delta = np.sqrt(delta ** 2. - ssq)
        xopt = self.xopt + nstep
        gopt = self.model_obj_grad(xopt)
        bub = np.maximum(bub, np.dot(aub, xopt))
        beq = np.dot(aeq, xopt)
        if mc == 0:
            tstep = bvtcg(xopt, gopt, self.model_lag_hessp, (), self.xl,
                          self.xu, delta, **kwargs)
        else:
            tstep = lctcg(xopt, gopt, self.model_lag_hessp, (), aub, bub, aeq,
                          beq, self.xl, self.xu, delta, **kwargs)
        return nstep + tstep

    def model_step(self, delta, **kwargs):
        r"""
        Evaluate a model-improvement step.
        TODO: Give details.
        """
        return self._models.improve_geometry(self._knew, delta, **kwargs)

    def reset_models(self):
        self._models.reset_models()

    def check_models(self, stack_level=2):
        r"""
        Check whether the models satisfy the interpolation conditions.
        """
        self._models.check_models(stack_level)

    def _eval_con(self, con, x):
        if con is not None:
            cx = np.atleast_1d(con(x, *self._args))
            if cx.dtype.kind in np.typecodes['AllInteger']:
                cx = np.asarray(cx, dtype=float)
            if self.disp:
                print(f'{con.__name__}({x}) = {cx}.')
        else:
            cx = np.asarray([], dtype=float)
        return cx


class Models:
    """
    Representation of a model of an optimization problem for which the objective
    and nonlinear constraint functions are modeled by quadratic functions
    obtained by underdetermined interpolation.

    Given an interpolation set, the freedom bequeathed by the interpolation
    conditions are taken up by minimizing the updates of the Hessian matrices in
    Frobenius norm [1]_. The interpolation points may be infeasible, but they
    always satisfies the bound constraints.

    References
    ----------
    .. [1] M. J. D. Powell. "Least Frobenius norm updating of quadratic models
       that satisfy interpolation conditions." In: Math. Program. 100 (2004),
       pp. 183--215.
    """

    def __init__(self, fun, xbase, xl, xu, Aub, bub, Aeq, beq, cub, ceq,
                 options):
        """
        Construct the initial models of the optimization problem.
        """
        self._xl = xl
        self._xu = xu
        self._Aub = Aub
        self._bub = bub
        self._Aeq = Aeq
        self._beq = beq
        self.shift_constraints(xbase)
        n = xbase.size
        npt = options.get('npt')
        rhobeg = options.get('rhobeg')
        cub_xbase = cub(xbase)
        mnlub = cub_xbase.size
        ceq_xbase = ceq(xbase)
        mnleq = ceq_xbase.size
        self._xpt = np.zeros((npt, n), dtype=float)
        self._fval = np.empty(npt, dtype=float)
        self._rval = np.empty(npt, dtype=float)
        self._cvalub = np.empty((npt, mnlub), dtype=float)
        self._cvaleq = np.empty((npt, mnleq), dtype=float)
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
                if abs(self._xu[km]) <= .5 * rhobeg:
                    stepa = -rhobeg
                else:
                    stepa = rhobeg
                self._xpt[k, km] = stepa
            elif n < k <= 2 * n:
                stepa = self._xpt[kx + 1, kx]
                if abs(self._xl[kx]) <= .5 * rhobeg:
                    stepb = min(2. * rhobeg, self._xu[kx])
                elif abs(self._xu[kx]) <= .5 * rhobeg:
                    stepb = max(-2. * rhobeg, self._xl[kx])
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
            if k == 0:
                # The constraints functions have already been evaluated at xbase
                # to get the number of constraints.
                self._cvalub[0, :] = cub_xbase
                self._cvaleq[0, :] = ceq_xbase
            else:
                self._cvalub[k, :] = cub(xbase + self._xpt[k, :])
                self._cvaleq[k, :] = ceq(xbase + self._xpt[k, :])
            self._rval[k] = self.resid(k)

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
        self._obj_alt = self.alt_model(self._fval)
        self._cub = np.empty(mnlub, dtype=Quadratic)
        self._cub_alt = np.empty(mnlub, dtype=Quadratic)
        for i in range(mnlub):
            self._cub[i] = Quadratic(self._bmat, self._zmat, self._idz,
                                     self._cvalub[:, i])
            self._cub_alt[i] = self.alt_model(self._cvalub[:, i])
        self._ceq = np.empty(mnleq, dtype=Quadratic)
        self._ceq_alt = np.empty(mnleq, dtype=Quadratic)
        for i in range(mnleq):
            self._ceq[i] = Quadratic(self._bmat, self._zmat, self._idz,
                                     self._cvaleq[:, i])
            self._ceq_alt[i] = self.alt_model(self._cvaleq[:, i])

    @property
    def xl(self):
        r"""
        Return the lower-bound constraints on the decision variables.
        """
        return self._xl

    @property
    def xu(self):
        r"""
        Return the upper-bound constraints on the decision variables.
        """
        return self._xu

    @property
    def aub(self):
        r"""
        Return the Jacobian matrix of the linear inequality constraints.
        """
        return self._Aub

    @property
    def bub(self):
        r"""
        Return the right-hand side vector of the linear inequality constraints.
        """
        return self._bub

    @property
    def mlub(self):
        return self.bub.size

    @property
    def aeq(self):
        r"""
        Return the Jacobian matrix of the linear equality constraints.
        """
        return self._Aeq

    @property
    def beq(self):
        r"""
        Return the right-hand side vector of the linear equality constraints.
        """
        return self._beq

    @property
    def mleq(self):
        return self.beq.size

    @property
    def xpt(self):
        r"""
        Return the interpolation points.
        """
        return self._xpt

    @property
    def fval(self):
        r"""
        Return the values of the objective function at the interpolation points.
        """
        return self._fval

    @property
    def rval(self):
        return self._rval

    @property
    def cvalub(self):
        return self._cvalub

    @property
    def mnlub(self):
        return self._cvalub.shape[1]

    @property
    def cvaleq(self):
        return self._cvaleq

    @property
    def mnleq(self):
        return self._cvaleq.shape[1]

    @property
    def bmat(self):
        r"""
        Return the last columns of the KKT matrix of interpolation.
        """
        return self._bmat

    @property
    def zmat(self):
        r"""
        Return the factorization matrix of the leading submatrix of the KKT
        matrix of interpolation.
        """
        return self._zmat

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
            step = self.xpt[knew, :] - self.xopt
            self._obj.shift_expansion_point(step, self._xpt)
            self._obj_alt.shift_expansion_point(step, self._xpt)
            for i in range(self.mnlub):
                self._cub[i].shift_expansion_point(step, self._xpt)
                self._cub_alt[i].shift_expansion_point(step, self._xpt)
            for i in range(self.mnleq):
                self._ceq[i].shift_expansion_point(step, self._xpt)
                self._ceq_alt[i].shift_expansion_point(step, self._xpt)
            self._kopt = knew

    @property
    def xopt(self):
        r"""
        Return the best point so far.
        """
        return self._xpt[self._kopt, :]

    @property
    def fopt(self):
        r"""
        Return the value of the objective function at the best point so far.
        """
        return self._fval[self._kopt]

    @property
    def ropt(self):
        """
        Return the constraint violation at the best point so far.
        """
        return self._rval[self._kopt]

    @property
    def coptub(self):
        return self._cvalub[self._kopt, :]

    @property
    def copteq(self):
        return self._cvaleq[self._kopt, :]

    @property
    def type(self):
        # CUTEst classification scheme
        # https://www.cuter.rl.ac.uk/Problems/classification.shtml
        n = self._xpt.shape[1]
        if self.mnlub + self.mnleq > 0:
            return 'O'
        elif self.mlub + self.mleq > 0:
            return 'L'
        elif np.all(self._xl == -np.inf) and np.all(self._xu == np.inf):
            return 'U'
        elif np.all(self._xu - self._xl <= 10. * EPS * n * np.abs(self._xu)):
            return 'X'
        else:
            return 'B'

    def obj(self, x):
        r"""
        Evaluate the objective function of the model at ``x``.
        """
        return self.fopt + self._obj(x, self._xpt, self._kopt)

    def obj_grad(self, x):
        r"""
        Evaluate the gradient of the objective function of the model at ``x``.
        """
        return self._obj.grad(x, self._xpt, self._kopt)

    def obj_hess(self):
        return self._obj.hess(self._xpt)

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

    def obj_alt(self, x):
        return self.fopt + self._obj_alt(x, self._xpt, self._kopt)

    def obj_alt_grad(self, x):
        return self._obj_alt.grad(x, self._xpt, self._kopt)

    def obj_alt_hess(self):
        return self._obj_alt.hess(self._xpt)

    def obj_alt_hessp(self, x):
        return self._obj_alt.hessp(x, self._xpt)

    def obj_alt_curv(self, x):
        return self._obj_alt.curv(x, self._xpt)

    def cub(self, x, i):
        return self.coptub + self._cub[i](x, self._xpt, self._kopt)

    def cub_grad(self, x, i):
        return self._cub[i].grad(x, self._xpt, self._kopt)

    def cub_hess(self, i):
        return self._cub[i].hess(self._xpt)

    def cub_hessp(self, x, i):
        return self._cub[i].hessp(x, self._xpt)

    def cub_curv(self, x, i):
        return self._cub[i].curv(x, self._xpt)

    def cub_alt(self, x, i):
        return self.coptub + self._cub_alt[i](x, self._xpt, self._kopt)

    def cub_alt_grad(self, x, i):
        return self._cub_alt[i].grad(x, self._xpt, self._kopt)

    def cub_alt_hess(self, i):
        return self._cub_alt[i].hess(self._xpt)

    def cub_alt_hessp(self, x, i):
        return self._cub_alt[i].hessp(x, self._xpt)

    def cub_alt_curv(self, x, i):
        return self._cub_alt[i].curv(x, self._xpt)

    def ceq(self, x, i):
        return self.copteq + self._ceq[i](x, self._xpt, self._kopt)

    def ceq_grad(self, x, i):
        return self._ceq[i].grad(x, self._xpt, self._kopt)

    def ceq_hess(self, i):
        return self._ceq[i].hess(self._xpt)

    def ceq_hessp(self, x, i):
        return self._ceq[i].hessp(x, self._xpt)

    def ceq_curv(self, x, i):
        return self._ceq[i].curv(x, self._xpt)

    def ceq_alt(self, x, i):
        return self.copteq + self._ceq_alt[i](x, self._xpt, self._kopt)

    def ceq_alt_grad(self, x, i):
        return self._ceq_alt[i].grad(x, self._xpt, self._kopt)

    def ceq_alt_hess(self, i):
        return self._ceq_alt[i].hess(self._xpt)

    def ceq_alt_hessp(self, x, i):
        return self._ceq_alt[i].hessp(x, self._xpt)

    def ceq_alt_curv(self, x, i):
        return self._ceq_alt[i].curv(x, self._xpt)

    def lag(self, x, lmlub, lmleq, lmnlub, lmnleq):
        lx = self.obj(x)
        lx += np.inner(lmlub, np.dot(self._Aub, x) - self._bub)
        lx += np.inner(lmleq, np.dot(self._Aeq, x) - self._beq)
        for i in range(self.mnlub):
            lx += lmnlub[i] * self.cub(x, i)
        for i in range(self.mnleq):
            lx += lmnleq[i] * self.ceq(x, i)
        return lx

    def lag_grad(self, x, lmlub, lmleq, lmnlub, lmnleq):
        gx = self.obj_grad(x)
        gx += np.dot(self._Aub.T, lmlub)
        gx += np.dot(self._Aeq.T, lmleq)
        for i in range(self.mnlub):
            gx += lmnlub[i] * self.cub_grad(x, i)
        for i in range(self.mnleq):
            gx += lmnleq[i] * self.ceq_grad(x, i)
        return gx

    def lag_hess(self, lmnlub, lmnleq):
        hx = self.obj_hess()
        for i in range(self.mnlub):
            hx += lmnlub[i] * self.cub_hess(i)
        for i in range(self.mnleq):
            hx += lmnleq[i] * self.ceq_hess(i)
        return hx

    def lag_hessp(self, x, lmnlub, lmnleq):
        r"""
        Evaluate the product of the Hessian matrix of the Lagrangian function of
        the model and ``x``.
        """
        hx = self.obj_hessp(x)
        for i in range(self.mnlub):
            hx += lmnlub[i] * self.cub_hessp(x, i)
        for i in range(self.mnleq):
            hx += lmnleq[i] * self.ceq_hessp(x, i)
        return hx

    def lag_curv(self, x, lmnlub, lmnleq):
        cx = self.obj_curv(x)
        for i in range(self.mnlub):
            cx += lmnlub[i] * self.cub_curv(x, i)
        for i in range(self.mnleq):
            cx += lmnleq[i] * self.ceq_curv(x, i)
        return cx

    def shift_constraints(self, x):
        r"""
        Shift the bound constraints and the right-hand sides of the linear
        inequality and equality constraints.
        """
        self._xl -= x
        self._xu -= x
        self._bub -= np.dot(self._Aub, x)
        self._beq -= np.dot(self._Aeq, x)

    def shift_origin(self):
        r"""
        Update the model when the shift of the origin is modified.
        """
        xopt = np.copy(self.xopt)
        npt, n = self._xpt.shape
        xoptsq = np.inner(xopt, xopt)

        # Make the changes to the columns of the KKT matrix of interpolation
        # that do not depend on the factorization of the leading submatrix.
        qoptsq = .25 * xoptsq
        updt = np.dot(self._xpt, xopt) - .5 * xoptsq
        hxpt = self._xpt - .5 * xopt[np.newaxis, :]
        for k in range(npt):
            step = updt[k] * hxpt[k, :] + qoptsq * xopt
            temp = np.outer(self._bmat[k, :], step)
            self._bmat[npt:, :] += temp + temp.T

        # Calculate the remaining revisions of the matrix.
        temp = qoptsq * np.outer(xopt, np.sum(self._zmat, axis=0))
        temp += np.matmul(hxpt.T, self._zmat * updt[:, np.newaxis])
        for k in range(self._idz):
            self._bmat[:npt, :] -= np.outer(self._zmat[:, k], temp[:, k])
            self._bmat[npt:, :] -= np.outer(temp[:, k], temp[:, k])
        for k in range(self._idz, npt - n - 1):
            self._bmat[:npt, :] += np.outer(self._zmat[:, k], temp[:, k])
            self._bmat[npt:, :] += np.outer(temp[:, k], temp[:, k])

        # Complete the shift by updating the models.
        self._obj.shift_origin(self._xpt, self._kopt)
        self._obj_alt.shift_origin(self._xpt, self._kopt)
        for i in range(self.mnlub):
            self._cub[i].shift_origin(self._xpt, self._kopt)
            self._cub_alt[i].shift_origin(self._xpt, self._kopt)
        for i in range(self.mnleq):
            self._ceq[i].shift_origin(self._xpt, self._kopt)
            self._ceq_alt[i].shift_origin(self._xpt, self._kopt)
        self.shift_constraints(xopt)
        self._xpt -= xopt[np.newaxis, :]

    def update(self, step, fx, cubx, ceqx, knew=None):
        r"""
        Update KKT matrix of interpolation and the model to replace the
        ``knew``-th interpolation point by ``self.xopt + step``. If
        ``knew = -1``, the index is selected by maximizing the absolute value of
        the weighted denominator in Equation (2.12) of Powell (2004).
        """
        npt, n = self._xpt.shape

        # Evaluate the Lagrange polynomials of the interpolation points and the
        # real parameter given in Equation (2.13) of Powell (2004).
        beta, vlag = self._beta(step)

        # Select the index of the interpolation point to be deleted if it has
        # not been provided. It is picked by maximizing the absolute value of
        # the weighted denominator in Equation (2.12) of Powell (2004).
        if knew is None:
            knew = self._get_point_to_remove(beta, vlag)

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
            raise ZeroDivisionError

        # Complete the updating of the factorization matrix of the leading
        # submatrix of the KKT matrix of interpolation.
        reduce = False
        hval = np.sqrt(abs(sigma))
        if jdz == 0:
            scala = tau / hval
            scalb = self._zmat[knew, 0] / hval
            self._zmat[:, 0] = scala * self._zmat[:, 0] - scalb * vlag[:npt]
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
            self._zmat[:, kdz] = tau * self._zmat[:, kdz] - temp * vlag[:npt]
            self._zmat[:, kdz] *= scala
            self._zmat[:, jdz] -= tempa * omega + tempb * vlag[:npt]
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
            cosv = (alpha * vlag[npt + j] - tau * bsav[j]) / sigma
            sinv = (tau * vlag[npt + j] + beta * bsav[j]) / sigma
            self._bmat[:npt, j] += cosv * vlag[:npt] - sinv * omega
            self._bmat[npt:npt + j + 1, j] += cosv * vlag[npt:npt + j + 1]
            self._bmat[npt:npt + j + 1, j] -= sinv * bsav[:j + 1]
            self._bmat[npt + j, :j + 1] = self._bmat[npt:npt + j + 1, j]

        # Update the models of the problem.
        xnew = self.xopt + step
        xold = np.copy(self._xpt[knew, :])
        dfx = fx - self.obj(xnew)
        dcubx = np.empty(self.mnlub, dtype=float)
        for i in range(self.mnlub):
            dcubx[i] = cubx[i] - self.cub(xnew, i)
        dceqx = np.empty(self.mnleq, dtype=float)
        for i in range(self.mnleq):
            dceqx[i] = ceqx[i] - self.ceq(xnew, i)
        self._fval[knew] = fx
        self._cvalub[knew, :] = cubx
        self._cvaleq[knew, :] = ceqx
        self._xpt[knew, :] = xnew
        self._rval[knew] = self.resid(knew)
        self._obj.update(self._xpt, self._kopt, xold, self._bmat, self._zmat,
                         self._idz, knew, dfx)
        self._obj_alt = self.alt_model(self._fval)
        for i in range(self.mnlub):
            self._cub[i].update(self._xpt, self._kopt, xold, self._bmat,
                                self._zmat, self._idz, knew, dcubx[i])
            self._cub_alt[i] = self.alt_model(self._cvalub[:, i])
        for i in range(self.mnleq):
            self._ceq[i].update(self._xpt, self._kopt, xold, self._bmat,
                                self._zmat, self._idz, knew, dceqx[i])
            self._ceq_alt[i] = self.alt_model(self._cvaleq[:, i])
        return knew

    def alt_model(self, fval):
        model = Quadratic(self._bmat, self._zmat, self._idz, fval)
        model.shift_expansion_point(self.xopt, self._xpt)
        return model

    def reset_models(self):
        r"""
        Reset the quadratic models of the nonlinear functions to the ones whose
        Hessian matrices are least in Frobenius norm.
        """
        self._obj = self._obj_alt
        self._cub = self._cub_alt
        self._ceq = self._ceq_alt

    def improve_geometry(self, klag, delta, **kwargs):
        r"""
        Evaluate a model-improvement step.
        TODO: Give details.
        """
        # Define the tolerances to compare floating-point numbers with zero.
        npt = self._xpt.shape[0]
        tol = 10. * EPS * npt

        # Define the klag-th Lagrange polynomial at the best point.
        lag = Quadratic(self._bmat, self._zmat, self._idz, klag)
        lag.shift_expansion_point(self.xopt, self._xpt)

        # Determine a point on a line between the optimal point and the other
        # interpolation points, chosen to maximize the absolute value of the
        # klag-th Lagrange polynomial, defined above.
        omega = omega_product(self._zmat, self._idz, klag)
        alpha = omega[klag]
        glag = lag.grad(self.xopt, self._xpt, self._kopt)
        step = bvlag(self._xpt, self._kopt, klag, glag, self._xl, self._xu,
                     delta, alpha, **kwargs)

        # Evaluate the constrained Cauchy step from the optimal point of the
        # absolute value of the klag-th Lagrange polynomial.
        salt, cauchy = bvcs(self._xpt, self._kopt, glag, lag.curv, (self._xpt,),
                            self._xl, self._xu, delta, **kwargs)

        # Among the two computed alternative points, we choose the one leading
        # to the greatest value of sigma in Equation (2.13) or Powell (2004).
        beta, vlag = self._beta(step)
        sigma = vlag[klag] ** 2. + alpha * beta
        if sigma < cauchy and cauchy > tol * max(1, abs(sigma)):
            step = salt
        return step

    def resid(self, x, cubx=None, ceqx=None):
        if isinstance(x, (int, np.integer)):
            cubx = self._cvalub[x, :]
            ceqx = self._cvaleq[x, :]
            x = self.xpt[x, :]
        cub = np.r_[np.dot(self._Aub, x) - self._bub, cubx]
        ceq = np.r_[np.dot(self._Aeq, x) - self._beq, ceqx]
        cbd = np.r_[x - self._xu, self._xl - x]
        return np.max(np.r_[cub, np.abs(ceq), cbd], initial=0.)

    def check_models(self, stack_level=2):
        r"""
        Check whether the models satisfy the interpolation conditions.
        """
        stack_level += 1
        self._obj.check_model(self._xpt, self._fval, self._kopt, stack_level)
        for i in range(self.mnlub):
            self._cub[i].check_model(self._xpt, self._cvalub[:, i], self._kopt,
                                     stack_level)
        for i in range(self.mnleq):
            self._ceq[i].check_model(self._xpt, self._cvaleq[:, i], self._kopt,
                                     stack_level)

    def _get_point_to_remove(self, beta, vlag):
        npt = self._xpt.shape[0]
        zsq = self._zmat ** 2.
        zsq = np.c_[-zsq[:, :self._idz], zsq[:, self._idz:]]
        alpha = np.sum(zsq, axis=1)
        sigma = vlag[:npt] ** 2. + beta * alpha
        dsq = np.sum((self._xpt - self.xopt[np.newaxis, :]) ** 2., axis=1)
        return np.argmax(np.abs(sigma) * np.square(dsq))

    def _beta(self, step):
        r"""
        Evaluate the real parameter sigma in Equation (2.13) or Powell (2004).
        """
        npt, n = self._xpt.shape
        vlag = np.empty(npt + n, dtype=float)
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
        vlag[:npt] = np.dot(self._bmat[:npt, :], step)
        vlag[:npt] += np.dot(self._zmat, temp)
        vlag[self._kopt] += 1.
        vlag[npt:] = np.dot(self._bmat[:npt, :].T, check)
        bsp = np.inner(vlag[npt:], step)
        vlag[npt:] += np.dot(self._bmat[npt:, :], step)
        bsp += np.inner(vlag[npt:], step)
        beta += stx ** 2. + stepsq * (xoptsq + 2. * stx + .5 * stepsq) - bsp
        return beta, vlag


class Quadratic:
    """
    Representation of a quadratic multivariate function.

    To improve the computational efficiency of the updates of the models, the
    Hessian matrix of a model is stored as an explicit and an implicit part,
    which rely on the coordinates of the interpolation points [1]_.

    References
    ----------
    .. [1] M. J. D. Powell. "The NEWUOA software for unconstrained optimization
       without derivatives." In: Large-Scale Nonlinear Optimization. Ed. by G.
       Di Pillo and M. Roma. New York, NY, US: Springer, 2006, pp. 255--297.
    """

    def __init__(self, bmat, zmat, idz, fval):
        """
        Construct a quadratic function by underdetermined interpolation.

        The freedom bequeathed by the interpolation conditions is taken up by
        minimizing the Hessian matrix of the model in Frobenius norm [1]_.

        Parameters
        ----------
        bmat : numpy.ndarray, shape (npt + n, n)
            Last ``n`` columns of the inverse KKT matrix of interpolation.
        zmat : numpy.ndarray, shape (npt, npt - n - 1)
            Rank factorization matrix of the leading ``npt`` submatrix of the
            inverse KKT matrix of interpolation.
        idz : int
            Number of nonpositive eigenvalues of the leading ``npt`` submatrix
            of the inverse KKT matrix of interpolation. Although its theoretical
            value is always 0, it is designed to tackle numerical difficulties
            caused by ill-conditioned problems.
        fval : int or numpy.ndarray, shape (npt,)
            Evaluations associated with the interpolation points. An integer
            value represents the ``npt``-dimensional vector whose components are
            all zero, except the `fval`-th one whose value is one. Hence,
            passing an integer value construct the `fval`-th Lagrange polynomial
            associated with the interpolation points.

        References
        ----------
        .. [1] M. J. D. Powell. "Least Frobenius norm updating of quadratic
           models that satisfy interpolation conditions." In: Math. Program. 100
           (2004), pp. 183--215.
        """
        npt = zmat.shape[0]
        if isinstance(fval, (int, np.integer)):
            # The gradient of the fval-th Lagrange quadratic model is the
            # product of the first npt rows of bmat with the npt-dimensional
            # vector whose components are zero, except the fval-th one whose
            # value is one. To improve the computational efficiency of the code,
            # the product is made implicitly.
            self._gq = np.copy(bmat[fval, :])
        else:
            self._gq = np.dot(bmat[:npt, :].T, fval)
        self._pq = omega_product(zmat, idz, fval)

        # Initially, the explicit part of the Hessian matrix of the model is the
        # zero matrix. To improve the computational efficiency of the code, it
        # is stored only if it becomes a nonzero matrix.
        self._hq = None

    def __call__(self, x, xpt, kopt):
        """
        Evaluate the quadratic function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function.
        kopt : int
            Index of the interpolation point around which the quadratic function
            is defined. Since the constant term of the quadratic function is not
            maintained, ``self.__call__(xpt[kopt, :], xpt, kopt)`` is zero.

        Returns
        -------
        float
            The value of the quadratic function at `x`.
        """
        x = x - xpt[kopt, :]
        qx = np.inner(self._gq, x)
        qx += .5 * np.inner(self._pq, np.dot(xpt, x) ** 2.)
        if self._hq is not None:
            # To improve the computational efficiency of the code, the explicit
            # part of the Hessian matrix of the quadratic function may be
            # undefined, in which case it is understood as the zero matrix.
            qx += .5 * np.inner(x, np.dot(self._hq, x))
        return qx

    @property
    def gq(self):
        """
        Get the stored gradient of the model.

        Returns
        -------
        numpy.ndarray, shape (n,)
            The stored gradient of the model.
        """
        return self._gq

    @property
    def pq(self):
        """
        Get the stored implicit Hessian matrix of the model.

        Returns
        -------
        numpy.ndarray, shape (npt,)
            The stored implicit Hessian matrix of the model.
        """
        return self._pq

    @property
    def hq(self):
        """
        Get the stored explicit Hessian matrix of the model.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            The stored explicit Hessian matrix of the model.
        """
        if self._hq is None:
            return np.zeros((self._gq.size, self._gq.size), dtype=float)
        return self._hq

    def grad(self, x, xpt, kopt):
        """
        Evaluate the gradient of the quadratic function.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the quadratic function is to be evaluated.
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function.
        kopt : int
            Index of the interpolation point around which the quadratic function
            is defined. Since the constant term of the quadratic function is not
            maintained, ``self.__call__(xpt[kopt, :], xpt, kopt)`` is zero.

        Returns
        -------
        numpy.ndarray, shape (n,)
            The value of the gradient of the quadratic function at `x`.
        """
        return self._gq + self.hessp(x - xpt[kopt, :], xpt)

    def hess(self, xpt):
        """
        Evaluate the Hessian matrix of the quadratic function.

        Parameters
        ----------
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function.

        Returns
        -------
        numpy.ndarray, shape (n, n)
            The Hessian matrix of the quadratic function.
        """
        return self.hq + np.matmul(xpt.T, self._pq[:, np.newaxis] * xpt)

    def hessp(self, x, xpt):
        """
        Evaluate the product of the Hessian matrix of the quadratic function
        with any vector.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Left-hand side of the product to be evaluated.
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function.

        Returns
        -------
        numpy.ndarray, shape (n,)
            The value of the product of the Hessian matrix of the quadratic
            function with the vector `x`.
        """
        hx = np.dot(xpt.T, self._pq * np.dot(xpt, x))
        if self._hq is not None:
            hx += np.dot(self._hq, x)
        return hx

    def curv(self, x, xpt):
        """
        Evaluate the curvature of the quadratic function.

        Although it is defined as ``numpy.dot(x, self.hessp(x, xpt))``, the
        evaluation of this method improves the computational efficiency.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the curvature is to be evaluated.
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function.

        Returns
        -------
        hx : numpy.ndarray, shape (n,)
            The value of the product of the Hessian matrix of the quadratic
            function with the vector `x`.
        """
        cx = np.inner(self._pq, np.dot(xpt, x) ** 2.)
        if self._hq is not None:
            cx += np.inner(x, np.dot(self._hq, x))
        return cx

    def shift_expansion_point(self, step, xpt):
        """
        Shift the point around which the quadratic function is defined.

        This method must be called when the index around which the quadratic
        function is defined is modified, or when the point in `xpt` around
        which the quadratic function is defined is modified.

        Parameters
        ----------
        step : numpy.ndarray, shape (n,)
            Displacement from the current point ``xopt`` around which the
            quadratic function is defined. After calling this method, the value
            of the quadratic function at ``xopt + step`` is zero, since the
            constant term of the function is not maintained.
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function.
        """
        self._gq += self.hessp(step, xpt)

    def shift_origin(self, xpt, kopt):
        """
        Update the model when the shift of the origin is modified.

        Parameters
        ----------
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function.
        kopt : int
            Index of the interpolation point around which the quadratic function
            is defined. Since the constant term of the quadratic function is not
            maintained, ``self.__call__(xpt[kopt, :], xpt, kopt)`` is zero.
        """
        n = xpt.shape[1]
        hxpt = xpt - .5 * xpt[np.newaxis, kopt, :]
        temp = np.outer(np.dot(hxpt.T, self._pq), xpt[kopt, :])
        if self._hq is None:
            self._hq = np.zeros((n, n), dtype=float)
        self._hq += temp + temp.T

    def update(self, xpt, kopt, xold, bmat, zmat, idz, knew, diff):
        """
        Update the model when the KKT matrix of interpolation is modified.

        Parameters
        ----------
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function.
        kopt : int
            Index of the interpolation point around which the quadratic function
            is defined. Since the constant term of the quadratic function is not
            maintained, ``self.__call__(xpt[kopt, :], xpt, kopt)`` is zero.
        xold : numpy.ndarray, shape (n,)
            Previous point around which the quadratic function was defined.
        bmat : numpy.ndarray, shape (npt + n, n)
            Last ``n`` columns of the inverse KKT matrix of interpolation.
        zmat : numpy.ndarray, shape (npt, npt - n - 1)
            Rank factorization matrix of the leading ``npt`` submatrix of the
            inverse KKT matrix of interpolation.
        idz : int
            Number of nonpositive eigenvalues of the leading ``npt`` submatrix
            of the inverse KKT matrix of interpolation. Although its theoretical
            value is always 0, it is designed to tackle numerical difficulties
            caused by ill-conditioned problems.
        knew : int
            Index of the interpolation point that has been modified.
        diff : float
            Difference between the evaluation of the previous model and the
            expected value at ``xpt[kopt, :]``.
        """
        # Update the explicit and implicit parts of the Hessian matrix of the
        # quadratic function. The knew-th component of the implicit part of the
        # Hessian matrix is decoded and added to the explicit Hessian matrix.
        # Then, the implicit part of the Hessian matrix is modified.
        n = xold.size
        omega = omega_product(zmat, idz, knew)
        if self._hq is None:
            self._hq = np.zeros((n, n), dtype=float)
        self._hq += self._pq[knew] * np.outer(xold, xold)
        self._pq[knew] = 0.
        self._pq += diff * omega

        # Update the gradient of the model. The constant term is not maintained,
        # to improve the computational efficiency.
        temp = omega * np.dot(xpt, xpt[kopt, :])
        self._gq += diff * (bmat[knew, :] + np.dot(xpt.T, temp))

    def check_model(self, xpt, fval, kopt, stack_level=2):
        """
        Check whether the evaluations of the quadratic function at the
        interpolation points match the expected values.

        Parameters
        ----------
        xpt : numpy.ndarray, shape (npt, n)
            Interpolation points that define the quadratic function.
        fval : numpy.ndarray, shape (npt,)
            Evaluations associated with the interpolation points.
        kopt : int
            Index of the interpolation point around which the quadratic function
            is defined. Since the constant term of the quadratic function is not
            maintained, ``self.__call__(xpt[kopt, :], xpt, kopt)`` is zero.
        stack_level : int, optional
            Stack level of the warning.
            Default is 2.
        """
        npt = fval.size
        tol = 10. * np.sqrt(EPS) * npt * np.max(np.abs(fval), initial=1.)
        diff = 0.
        for k in range(npt):
            qx = self(xpt[k, :], xpt, kopt)
            diff = max(diff, abs(qx + fval[kopt] - fval[k]))
        if diff > tol:
            stack_level += 1
            message = f'error in interpolation conditions is {diff:e}.'
            warnings.warn(message, RuntimeWarning, stacklevel=stack_level)
