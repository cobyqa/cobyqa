import numpy as np
import pdfo
import pybobyqa
import scipy

import cobyqa


class Minimizer:
    def __init__(self, problem, solver, max_eval, options, callback, *args, **kwargs):
        self.problem = problem
        self.solver = solver
        self.max_eval = max_eval
        self.options = dict(options)
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        self.fun_history = None
        self.constraint_violation_history = None
        if not self.validate():
            raise NotImplementedError

    def __call__(self):
        self.fun_history = []
        self.constraint_violation_history = []

        x0 = self.problem.x0
        xl = self.problem.xl
        xu = self.problem.xu
        a_inequality = self.problem.a_inequality
        b_inequality = self.problem.b_inequality
        a_equality = self.problem.a_equality
        b_equality = self.problem.b_equality
        options = dict(self.options)
        if self.solver.lower() == 'cobyqa':
            options['maxfev'] = self.max_eval
            res = cobyqa.minimize(self.eval, x0, xl=xl, xu=xu, Aub=a_inequality, bub=b_inequality, Aeq=a_equality, beq=b_equality, cub=self.problem.c_inequality, ceq=self.problem.c_equality, options=options)
            success = res.success
        elif self.solver.lower() in pdfo.__all__:
            method = self.solver if self.solver.lower() != 'pdfo' else None
            bounds = pdfo.Bounds(xl, xu)
            constraints = []
            if self.problem.m_linear_inequality > 0:
                constraints.append(pdfo.LinearConstraint(a_inequality, -np.inf, b_inequality))
            if self.problem.m_linear_equality > 0:
                constraints.append(pdfo.LinearConstraint(a_equality, b_equality, b_equality))
            if self.problem.m_nonlinear_inequality > 0:
                constraints.append(pdfo.NonlinearConstraint(self.problem.c_inequality, -np.inf, np.zeros(self.problem.m_nonlinear_inequality)))
            if self.problem.m_nonlinear_equality > 0:
                constraints.append(pdfo.NonlinearConstraint(self.problem.c_equality, np.zeros(self.problem.m_nonlinear_equality), np.zeros(self.problem.m_nonlinear_equality)))
            options['maxfev'] = self.max_eval
            options['eliminate_lin_eq'] = False
            res = pdfo.pdfo(self.eval, x0, method=method, bounds=bounds, constraints=constraints, options=options)
            success = res.success
        elif self.solver.lower() == 'py-bobyqa':
            xl = self.problem.xl
            xl[xl == -np.inf] = -1e20
            xu = self.problem.xu
            xu[xu == np.inf] = 1e20
            rhobeg = 0.1 * max(np.max(np.abs(x0)), 1.0)
            rhobeg = min(rhobeg, 0.4999 * np.min(xu - xl))
            rhoend = min(rhobeg, 1e-8)
            res = pybobyqa.solve(self.eval, x0, bounds=(xl, xu), rhobeg=rhobeg, rhoend=rhoend, maxfun=self.max_eval, do_logging=False)
            success = res.flag == res.EXIT_SUCCESS
        else:
            bounds = scipy.optimize.Bounds(xl, xu)
            constraints = []
            if self.problem.m_linear_inequality > 0:
                constraints.append(scipy.optimize.LinearConstraint(a_inequality, -np.inf, b_inequality))
            if self.problem.m_linear_equality > 0:
                constraints.append(scipy.optimize.LinearConstraint(a_equality, b_equality, b_equality))
            if self.problem.m_nonlinear_inequality > 0:
                constraints.append(scipy.optimize.NonlinearConstraint(self.problem.c_inequality, -np.inf, np.zeros(self.problem.m_nonlinear_inequality)))
            if self.problem.m_nonlinear_equality > 0:
                constraints.append(scipy.optimize.NonlinearConstraint(self.problem.c_equality, np.zeros(self.problem.m_nonlinear_equality), np.zeros(self.problem.m_nonlinear_equality)))
            if self.solver.lower() in ['bfgs', 'cg', 'slsqp']:
                options['maxiter'] = self.max_eval
            elif self.solver.lower() in ['l-bfgs-b', 'tnc']:
                options['maxfun'] = self.max_eval
            else:
                options['maxfev'] = self.max_eval
            res = scipy.optimize.minimize(self.eval, x0, method=self.solver, bounds=bounds, constraints=constraints, options=options)
            success = res.success
        return success, np.array(self.fun_history, copy=True), np.array(self.constraint_violation_history, copy=True)

    def validate(self):
        valid_solvers = {'cobyla', 'cobyqa', 'pdfo', 'slsqp'}
        if self.problem.type not in 'quadratic other':
            valid_solvers.update({'lincoa'})
            if self.problem.type not in 'adjacency linear':
                valid_solvers.update({'bobyqa', 'l-bfgs-b', 'nelder-mead', 'py-bobyqa', 'tnc'})
                if self.problem.type not in 'equality bound':
                    valid_solvers.update({'bfgs', 'cg', 'newuoa', 'uobyqa'})
        return self.solver.lower() in valid_solvers

    def eval(self, x):
        f = self.problem.fun(x, self.callback, *self.args, **self.kwargs)
        if self.callback is not None:
            self.fun_history.append(f[0])
            f = f[1]
        else:
            self.fun_history.append(f)
        self.constraint_violation_history.append(self.problem.constraint_violation(x))
        return f
