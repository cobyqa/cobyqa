import numpy as np


class Optimizer:

    def __init__(self, problem, solver, max_eval, options, callback, *args, **kwargs):
        self.problem = problem
        self.solver = solver
        self.max_eval = max_eval
        self.options = options
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        options = dict(self.options)
        fun_values = []
        resid_values = []
        if self.solver.lower() == 'cobyqa':
            from cobyqa import minimize
            options['max_eval'] = self.max_eval
            res = minimize(lambda x: self.fun(x, fun_values, resid_values), self.problem.x0, xl=self.xl, xu=self.xu, aub=self.aub, bub=self.bub, aeq=self.aeq, beq=self.beq, cub=self.cub, ceq=self.ceq, options=options)
            success = res.success
        elif self.solver.lower() in ['pdfo', 'uobyqa', 'newuoa', 'bobyqa', 'lincoa', 'cobyla']:
            from pdfo import Bounds, LinearConstraint, NonlinearConstraint, pdfo
            bounds = Bounds(self.xl, self.xu)
            constraints = []
            if self.m_linear_ub > 0:
                constraints.append(
                    LinearConstraint(self.aub, -np.inf, self.bub))
            if self.m_linear_eq > 0:
                constraints.append(
                    LinearConstraint(self.aeq, self.beq, self.beq))
            if self.m_nonlinear_ub > 0:
                constraints.append(NonlinearConstraint(self.cub, -np.inf, np.zeros(self.m_nonlinear_ub)))
            if self.m_nonlinear_eq > 0:
                constraints.append(NonlinearConstraint(self.ceq, np.zeros(self.m_nonlinear_eq), np.zeros(self.m_nonlinear_eq)))
            options['maxfev'] = self.max_eval
            options['eliminate_lin_eq'] = False
            method = None if self.solver.lower() == 'pdfo' else self.solver
            res = pdfo(self.fun, self.problem.x0, (fun_values, resid_values), method, bounds, constraints, options)
            success = res.success
        else:
            from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
            bounds = Bounds(self.xl, self.xu)
            constraints = []
            if self.m_linear_ub > 0:
                constraints.append(LinearConstraint(self.aub, -np.inf, self.bub))
            if self.m_linear_eq > 0:
                constraints.append(LinearConstraint(self.aeq, self.beq, self.beq))
            if self.m_nonlinear_ub > 0:
                constraints.append(NonlinearConstraint(self.cub, -np.inf, np.zeros(self.m_nonlinear_ub)))
            if self.m_nonlinear_eq > 0:
                constraints.append(NonlinearConstraint(self.ceq, np.zeros(self.m_nonlinear_eq), np.zeros(self.m_nonlinear_eq)))
            if self.solver.lower() in ['cg', 'bfgs', 'newton-cg', 'cobyla', 'slsqp', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']:
                options['maxiter'] = self.max_eval
            elif self.solver.lower() in ['l-bfgs-b', 'tnc']:
                options['maxfun'] = self.max_eval
            else:
                options['maxfev'] = self.max_eval
            res = minimize(self.fun, self.problem.x0, (fun_values, resid_values), method=self.solver, bounds=bounds, constraints=constraints, options=options)
            success = res.success
        return success, np.array(fun_values), np.array(resid_values)

    @property
    def m_linear_ub(self):
        if self.problem.m == 0:
            return 0
        else:
            return np.count_nonzero(self.problem.is_linear_cons & np.logical_not(self.problem.is_eq_cons))

    @property
    def m_linear_eq(self):
        if self.problem.m == 0:
            return 0
        else:
            return np.count_nonzero(self.problem.is_linear_cons & self.problem.is_eq_cons)

    @property
    def m_nonlinear_ub(self):
        if self.problem.m == 0:
            return 0
        else:
            return np.count_nonzero(np.logical_not(self.problem.is_linear_cons | self.problem.is_eq_cons))

    @property
    def m_nonlinear_eq(self):
        if self.problem.m == 0:
            return 0
        else:
            return np.count_nonzero(np.logical_not(self.problem.is_linear_cons) & self.problem.is_eq_cons)

    @property
    def xl(self):
        xl = np.array(self.problem.bl)
        xl[xl <= -1e20] = -np.inf
        return xl

    @property
    def xu(self):
        xu = np.array(self.problem.bu)
        xu[xu >= 1e20] = np.inf
        return xu

    @property
    def cl(self):
        cl = np.array(self.problem.cl)
        cl[cl <= -1e20] = -np.inf
        return cl

    @property
    def cu(self):
        cu = np.array(self.problem.cu)
        cu[cu >= 1e20] = np.inf
        return cu

    @property
    def aub(self):
        if self.problem.m == 0:
            return np.empty((0, self.problem.n))
        iub = self.problem.is_linear_cons & np.logical_not(self.problem.is_eq_cons)
        iub_cl = self.cl[iub] > -np.inf
        iub_cu = self.cu[iub] < np.inf
        aub = []
        for i, index in enumerate(np.flatnonzero(iub)):
            _, g_val = self.problem.cons(np.zeros(self.problem.n), index, True)
            if iub_cl[i]:
                aub.append(-g_val)
            if iub_cu[i]:
                aub.append(g_val)
        return np.reshape(aub, (-1, self.problem.n))

    @property
    def bub(self):
        if self.problem.m == 0:
            return np.empty(0)
        iub = self.problem.is_linear_cons & np.logical_not(self.problem.is_eq_cons)
        iub_cl = self.cl[iub] > -np.inf
        iub_cu = self.cu[iub] < np.inf
        bub = []
        for i, index in enumerate(np.flatnonzero(iub)):
            c_val = self.problem.cons(np.zeros(self.problem.n), index)
            if iub_cl[i]:
                bub.append(c_val - self.cl[index])
            if iub_cu[i]:
                bub.append(self.cu[index] - c_val)
        return np.array(bub)

    @property
    def aeq(self):
        if self.problem.m == 0:
            return np.empty((0, self.problem.n))
        ieq = self.problem.is_linear_cons & self.problem.is_eq_cons
        aeq = []
        for index in np.flatnonzero(ieq):
            _, g_val = self.problem.cons(np.zeros(self.problem.n), index, True)
            aeq.append(g_val)
        return np.reshape(aeq, (-1, self.problem.n))

    @property
    def beq(self):
        if self.problem.m == 0:
            return np.empty(0)
        ieq = self.problem.is_linear_cons & self.problem.is_eq_cons
        beq = []
        for index in np.flatnonzero(ieq):
            c_val = self.problem.cons(np.zeros(self.problem.n), index)
            beq.append(c_val - 0.5 * (self.cl[index] + self.cu[index]))
        return np.array(beq)

    def fun(self, x, fun_values, resid_values):
        x = np.asarray(x, dtype=float)
        f = self.problem.obj(x)
        fun_values.append(f)
        resid_values.append(self.resid(x))
        if self.callback is not None:
            # Add noise to the function value.
            f = self.callback(x, f, *self.args, **self.kwargs)
        return f

    def cub(self, x):
        if self.problem.m == 0:
            return np.empty(0)
        x = np.asarray(x, dtype=float)
        iub = np.logical_not(self.problem.is_linear_cons | self.problem.is_eq_cons)
        iub_cl = self.cl[iub] > -np.inf
        iub_cu = self.cu[iub] < np.inf
        c = []
        for i, index in enumerate(np.flatnonzero(iub)):
            c_val = self.problem.cons(x, index)
            if iub_cl[i]:
                c.append(self.cl[index] - c_val)
            if iub_cu[i]:
                c.append(c_val - self.cu[index])
        return np.array(c, dtype=float)

    def ceq(self, x):
        if self.problem.m == 0:
            return np.empty(0)
        x = np.asarray(x, dtype=float)
        ieq = np.logical_not(self.problem.is_linear_cons) & self.problem.is_eq_cons
        c = []
        for index in np.flatnonzero(ieq):
            c_val = self.problem.cons(x, index)
            c.append(c_val - 0.5 * (self.cl[index] + self.cu[index]))
        return np.array(c, dtype=float)

    def resid(self, x):
        maxcv = np.max(self.xl - x, initial=0.0)
        maxcv = np.max(x - self.xu, initial=maxcv)
        maxcv = np.max(self.aub @ x - self.bub, initial=maxcv)
        maxcv = np.max(np.abs(self.aeq @ x - self.beq), initial=maxcv)
        maxcv = np.max(self.cub(x), initial=maxcv)
        maxcv = np.max(np.abs(self.ceq(x)), initial=maxcv)
        return maxcv
