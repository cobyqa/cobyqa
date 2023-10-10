import numpy as np
import pycutest


class Optimizer:
    """
    A wrapper class for the optimization solvers.
    """

    def __init__(self, problem, solver_name, max_eval_factor, options, callback, *args):
        """
        Initialize the optimization procedure.

        Parameters
        ----------
        problem : pycutest.CUTEstProblem
            Problem to be solved.
        solver_name : str
            Solver to be used.
        max_eval_factor : float
            Factor for the maximum number of function evaluations. The maximum
            number of function evaluations is ``max_eval_factor * problem.n``.
        options : dict
            Options for the solver.
        callback : callable
            Perturbation function.

                ``callback(x, f, *args) -> float``

            where ``x`` is the point at which the objective function is
            evaluated, ``f`` is the objective function value at ``x``, and
            ``args`` is a tuple of additional arguments.
        *args : tuple
            Additional arguments for the perturbation function.
        """
        self.problem = problem
        self.solver_name = solver_name
        self.max_eval = max_eval_factor * problem.n
        self.options = options
        self.callback = callback
        self.args = args

    def __call__(self):
        """
        Run the optimization procedure.

        Returns
        -------
        bool
            Whether the optimization is successful.
        numpy.ndarray, shape (n_eval,)
            History of the objective function values.
        numpy.ndarray, shape (n_eval,)
            History of the residual values.
        """
        options = dict(self.options)
        fun_values = []
        resid_values = []
        if self.solver_name.lower() == 'cobyqa':
            from cobyqa import minimize

            options['max_eval'] = self.max_eval
            res = minimize(lambda x: self.fun(x, fun_values, resid_values), self.x0, xl=self.xl, xu=self.xu, aub=self.aub, bub=self.bub, aeq=self.aeq, beq=self.beq, cub=self.cub, ceq=self.ceq, options=options)
            success = res.success
        elif self.solver_name.lower() == 'cobyqa-latest':
            from cobyqa_latest import minimize

            options['max_eval'] = self.max_eval
            res = minimize(lambda x: self.fun(x, fun_values, resid_values), self.x0, xl=self.xl, xu=self.xu, aub=self.aub, bub=self.bub, aeq=self.aeq, beq=self.beq, cub=self.cub, ceq=self.ceq, options=options)
            success = res.success
        elif self.solver_name.lower() in ['pdfo', 'uobyqa', 'newuoa', 'bobyqa', 'lincoa', 'cobyla']:
            from pdfo import Bounds, LinearConstraint, NonlinearConstraint, pdfo

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
            options['maxfev'] = self.max_eval
            options['eliminate_lin_eq'] = False
            method = None if self.solver_name.lower() == 'pdfo' else self.solver_name
            res = pdfo(self.fun, self.x0, (fun_values, resid_values), method, bounds, constraints, options)
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
            if self.solver_name.lower() in ['cg', 'bfgs', 'newton-cg', 'cobyla', 'slsqp', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']:
                options['maxiter'] = self.max_eval
            elif self.solver_name.lower() in ['l-bfgs-b', 'tnc']:
                options['maxfun'] = self.max_eval
            else:
                options['maxfev'] = self.max_eval
            res = minimize(self.fun, self.x0, (fun_values, resid_values), method=self.solver_name, bounds=bounds, constraints=constraints, options=options)
            success = res.success
        return success, np.array(fun_values), np.array(resid_values)

    @property
    def m_linear_ub(self):
        """
        Number of linear inequality constraints.

        Returns
        -------
        int
            Number of linear inequality constraints.
        """
        if self.problem.m == 0:
            return 0
        else:
            iub = self.problem.is_linear_cons & ~self.problem.is_eq_cons
            iub_cl = self.cl[iub] > -np.inf
            iub_cu = self.cu[iub] < np.inf
            return np.count_nonzero(iub_cl) + np.count_nonzero(iub_cu)

    @property
    def m_linear_eq(self):
        """
        Number of linear equality constraints.

        Returns
        -------
        int
            Number of linear equality constraints.
        """
        if self.problem.m == 0:
            return 0
        else:
            return np.count_nonzero(self.problem.is_linear_cons & self.problem.is_eq_cons)

    @property
    def m_nonlinear_ub(self):
        """
        Number of nonlinear inequality constraints.

        Returns
        -------
        int
            Number of nonlinear inequality constraints.
        """
        if self.problem.m == 0:
            return 0
        else:
            iub = ~(self.problem.is_linear_cons | self.problem.is_eq_cons)
            iub_cl = self.cl[iub] > -np.inf
            iub_cu = self.cu[iub] < np.inf
            return np.count_nonzero(iub_cl) + np.count_nonzero(iub_cu)

    @property
    def m_nonlinear_eq(self):
        """
        Number of nonlinear equality constraints.

        Returns
        -------
        int
            Number of nonlinear equality constraints.
        """
        if self.problem.m == 0:
            return 0
        else:
            return np.count_nonzero(~self.problem.is_linear_cons & self.problem.is_eq_cons)

    @property
    def x0(self):
        """
        Initial guess.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Initial guess.
        """
        from scipy.optimize import Bounds, LinearConstraint, minimize

        x0 = np.array(self.problem.x0)

        type_problem = pycutest.problem_properties(self.problem.name)['constraints']
        if type_problem in ['unconstrained', 'fixed']:
            return x0
        elif type_problem == 'bound':
            return np.clip(x0, self.xl, self.xu)
        else:
            def dsq(x):
                g = x - x0
                return 0.5 * np.inner(g, g), g

            bounds = Bounds(self.xl, self.xu)
            constraints = []
            if self.m_linear_ub > 0:
                constraints.append(LinearConstraint(self.aub, -np.inf, self.bub))
            if self.m_linear_eq > 0:
                constraints.append(LinearConstraint(self.aeq, self.beq, self.beq))
            res = minimize(dsq, x0, jac=True, bounds=bounds, constraints=constraints)
            return res.x

    @property
    def xl(self):
        """
        Lower bounds of the variables.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Lower bounds of the variables.
        """
        xl = np.array(self.problem.bl)
        xl[xl <= -1e20] = -np.inf
        return xl

    @property
    def xu(self):
        """
        Upper bounds of the variables.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Upper bounds of the variables.
        """
        xu = np.array(self.problem.bu)
        xu[xu >= 1e20] = np.inf
        return xu

    @property
    def cl(self):
        """
        Lower bounds of the constraints.

        Returns
        -------
        numpy.ndarray, shape (m,)
            Lower bounds of the constraints.
        """
        cl = np.array(self.problem.cl)
        cl[cl <= -1e20] = -np.inf
        return cl

    @property
    def cu(self):
        """
        Upper bounds of the constraints.

        Returns
        -------
        numpy.ndarray, shape (m,)
            Upper bounds of the constraints.
        """
        cu = np.array(self.problem.cu)
        cu[cu >= 1e20] = np.inf
        return cu

    @property
    def aub(self):
        """
        Left-hand side of the linear inequality constraints ``aub @ x <= bub``.

        Returns
        -------
        numpy.ndarray, shape (m_linear_ub, n)
            Left-hand side of the linear inequality constraints.
        """
        if self.problem.m == 0:
            return np.empty((0, self.problem.n))
        iub = self.problem.is_linear_cons & ~self.problem.is_eq_cons
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
        """
        Right-hand side of the linear inequality constraints ``aub @ x <= bub``.

        Returns
        -------
        numpy.ndarray, shape (m_linear_ub,)
            Right-hand side of the linear inequality constraints.
        """
        if self.problem.m == 0:
            return np.empty(0)
        iub = self.problem.is_linear_cons & ~self.problem.is_eq_cons
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
        """
        Left-hand side of the linear equality constraints ``aeq @ x = beq``.

        Returns
        -------
        numpy.ndarray, shape (m_linear_eq, n)
            Left-hand side of the linear equality constraints.
        """
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
        """
        Right-hand side of the linear equality constraints ``aeq @ x = beq``.

        Returns
        -------
        numpy.ndarray, shape (m_linear_eq, n)
            Right-hand side of the linear equality constraints.
        """
        if self.problem.m == 0:
            return np.empty(0)
        ieq = self.problem.is_linear_cons & self.problem.is_eq_cons
        beq = []
        for index in np.flatnonzero(ieq):
            c_val = self.problem.cons(np.zeros(self.problem.n), index)
            beq.append(c_val - 0.5 * (self.cl[index] + self.cu[index]))
        return np.array(beq)

    def fun(self, x, fun_values, resid_values):
        """
        Evaluate the objective function at ``x``.

        This method also applies a perturbation function if provided.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the objective function is evaluated.
        fun_values : list
            History of the objective function values.
        resid_values : list
            History of the residual values.

        Returns
        -------
        float
            Objective function value at ``x``.
        """
        x = np.asarray(x, dtype=float)
        f = self.problem.obj(x)
        fun_values.append(f)
        resid_values.append(self.resid(x))
        if self.callback is not None:
            # Add perturbation to the function value.
            f = self.callback(x, f, *self.args)
        return f

    def cub(self, x):
        """
        Evaluate the nonlinear inequality constraints at ``x``.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the nonlinear inequality constraints are evaluated.

        Returns
        -------
        numpy.ndarray, shape (m_nonlinear_ub,)
            Values of the nonlinear inequality constraints at ``x``.
        """
        if self.problem.m == 0:
            return np.empty(0)
        x = np.asarray(x, dtype=float)
        iub = ~(self.problem.is_linear_cons | self.problem.is_eq_cons)
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
        """
        Evaluate the nonlinear equality constraints at ``x``.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the nonlinear equality constraints are evaluated.

        Returns
        -------
        numpy.ndarray, shape (m_nonlinear_eq,)
            Values of the nonlinear equality constraints at ``x``.
        """
        if self.problem.m == 0:
            return np.empty(0)
        x = np.asarray(x, dtype=float)
        ieq = ~self.problem.is_linear_cons & self.problem.is_eq_cons
        c = []
        for index in np.flatnonzero(ieq):
            c_val = self.problem.cons(x, index)
            c.append(c_val - 0.5 * (self.cl[index] + self.cu[index]))
        return np.array(c, dtype=float)

    def resid(self, x):
        """
        Evaluate the residuals of the constraints at ``x``.

        Parameters
        ----------
        x : numpy.ndarray, shape (n,)
            Point at which the residuals are evaluated.

        Returns
        -------
        float
            Maximum residual value at ``x``.
        """
        maxcv = np.max(self.xl - x, initial=0.0)
        maxcv = np.max(x - self.xu, initial=maxcv)
        maxcv = np.max(self.aub @ x - self.bub, initial=maxcv)
        maxcv = np.max(np.abs(self.aeq @ x - self.beq), initial=maxcv)
        maxcv = np.max(self.cub(x), initial=maxcv)
        maxcv = np.max(np.abs(self.ceq(x)), initial=maxcv)
        return maxcv
