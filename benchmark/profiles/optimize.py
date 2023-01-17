import sys

import numpy as np
import pdfo
import pybobyqa
import scipy

import cobyqa
from .utils import NullIO, get_logger


class Minimizer:
    def __init__(self, problem, solver, max_eval, options, callback, *args, **kwargs):
        self.problem = problem
        self.solver = solver
        self.max_eval = max_eval
        self.options = dict(options)
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        if not self.validate():
            raise NotImplementedError

        # The following attributes store the objective function and maximum
        # constraint violation values obtained during a run.
        self.fun_history = None
        self.maxcv_history = None

    def __call__(self):
        self.fun_history = []
        self.maxcv_history = []

        x0 = self.problem.x0
        xl = self.problem.xl
        xu = self.problem.xu
        a_ineq = self.problem.a_ineq
        b_ineq = self.problem.b_ineq
        a_eq = self.problem.a_eq
        b_eq = self.problem.b_eq
        options = dict(self.options)
        logger = get_logger(__name__)
        if self.solver.lower() == "cobyqa":
            options["maxfev"] = self.max_eval
            res = cobyqa.minimize(self.eval, x0, xl=xl, xu=xu, aub=a_ineq, bub=b_ineq, aeq=a_eq, beq=b_eq, cub=self.problem.c_ineq, ceq=self.problem.c_eq, options=options)
            success = res.success
        elif self.solver.lower() == "cobyqa-old":
            # Import cobyqa_old locally, as it is not necessary to generate
            # performance and data profiles if no solver is "cobyqa-old".
            import cobyqa_old

            options["maxfev"] = self.max_eval
            res = cobyqa_old.minimize(self.eval, x0, xl=xl, xu=xu, Aub=a_ineq, bub=b_ineq, Aeq=a_eq, beq=b_eq, cub=self.problem.c_ineq, ceq=self.problem.c_eq, options=options)
            success = res.success
        elif self.solver.lower() in pdfo.__all__:
            method = self.solver if self.solver.lower() != "pdfo" else None
            bounds = pdfo.Bounds(xl, xu)
            constraints = []
            if self.problem.m_lin_ineq > 0:
                constraints.append(pdfo.LinearConstraint(a_ineq, -np.inf, b_ineq))
            if self.problem.m_lin_eq > 0:
                constraints.append(pdfo.LinearConstraint(a_eq, b_eq, b_eq))
            if self.problem.m_nonlin_ineq > 0:
                constraints.append(pdfo.NonlinearConstraint(self.problem.c_ineq, -np.inf, np.zeros(self.problem.m_nonlin_ineq)))
            if self.problem.m_nonlin_eq > 0:
                constraints.append(pdfo.NonlinearConstraint(self.problem.c_eq, np.zeros(self.problem.m_nonlin_eq), np.zeros(self.problem.m_nonlin_eq)))
            options["maxfev"] = self.max_eval
            options["eliminate_lin_eq"] = False
            res = pdfo.pdfo(self.eval, x0, method=method, bounds=bounds, constraints=constraints, options=options)
            success = res.success
        elif self.solver.lower() == "py-bobyqa":
            # Py-BOBYQA does not accept infinite bounds.
            xl = self.problem.xl
            xl[xl == -np.inf] = -1e20
            xu = self.problem.xu
            xu[xu == np.inf] = 1e20

            # Py-BOBYQA requires that np.min(xu - xl) >= 2.0 * rhobeg. The
            # values of rhobeg and rhoend are the default values otherwise.
            rhobeg = 0.1 * max(np.max(np.abs(x0)), 1.0)
            rhobeg = min(rhobeg, 0.4999 * np.min(xu - xl))
            rhoend = min(rhobeg, 1e-8)
            res = pybobyqa.solve(self.eval, x0, bounds=(xl, xu), rhobeg=rhobeg, rhoend=rhoend, maxfun=self.max_eval, do_logging=False)
            success = res.flag == res.EXIT_SUCCESS
        elif self.solver.lower() == "sqpdfo":
            # Import sqpdfo locally, as it is not necessary to generate
            # performance and data profiles if no solver is "sqpdfo".
            import sqpdfo

            max_eval = self.max_eval

            class Options:

                def __init__(self):
                    self.hess_approx = "model"
                    self.bfgs_restart = 0
                    self.whichmodel = "subbasis"
                    self.final_degree = "quadratic"
                    self.tol_grad = 1e-4
                    self.tol_feas = 1e-4
                    self.tol_bnds = 1e-4
                    self.miter = max_eval
                    self.msimul = max_eval
                    self.verbose = 0

            def constraints(x):
                c = np.dot(self.problem.a_eq, x) - self.problem.b_eq
                c = np.r_[c, self.problem.c_eq(x)]
                c = np.r_[c, self.problem.b_ineq - np.dot(self.problem.a_ineq, x)]
                c = np.r_[c, -self.problem.c_ineq(x)]
                return c

            # SQPDFO does not accept infinite bounds.
            x0 = np.atleast_2d(x0)
            xl = np.atleast_2d(self.problem.xl).T
            xl[xl == -np.inf] = -1e20
            xu = np.atleast_2d(self.problem.xu).T
            xu[xu == np.inf] = 1e20
            m_ineq = self.problem.m_lin_ineq + self.problem.m_nonlin_ineq
            m_eq = self.problem.m_lin_eq + self.problem.m_nonlin_eq
            options = Options()

            # Even when verbose = 0, standard outputs are produced. The standard
            # output if therefore redirected to null.
            sys.stdout = NullIO()
            if m_eq + m_ineq > 0:
                try:
                    _, _, info = sqpdfo.optimize(options, lambda x: self.eval(x), x0, xl, xu, m_eq, m_ineq, constraints)
                    success = info.flag == 0
                except (IndexError, ValueError) as err:
                    # SQPDFO seems to contain bugs, as these errors occur only
                    # on some linearly and nonlinearly constrained problems.
                    logger.warning(f"{self.solver}({self.problem.name}): {err}")
                    success = False
            else:
                _, _, info = sqpdfo.optimize(options, lambda x: self.eval(x), x0, xl, xu)
                success = info.flag == 0
            sys.stdout = sys.__stdout__
        else:
            bounds = scipy.optimize.Bounds(xl, xu)
            constraints = []
            if self.problem.m_lin_ineq > 0:
                constraints.append(scipy.optimize.LinearConstraint(a_ineq, -np.inf, b_ineq))
            if self.problem.m_lin_eq > 0:
                constraints.append(scipy.optimize.LinearConstraint(a_eq, b_eq, b_eq))
            if self.problem.m_nonlin_ineq > 0:
                constraints.append(scipy.optimize.NonlinearConstraint(self.problem.c_ineq, -np.inf, np.zeros(self.problem.m_nonlin_ineq)))
            if self.problem.m_nonlin_eq > 0:
                constraints.append(scipy.optimize.NonlinearConstraint(self.problem.c_eq, np.zeros(self.problem.m_nonlin_eq), np.zeros(self.problem.m_nonlin_eq)))
            if self.solver.lower() in ["bfgs", "cg", "slsqp"]:
                # These solvers do not have any option to control the maximum
                # number of function evaluations.
                options["maxiter"] = self.max_eval
            elif self.solver.lower() in ["l-bfgs-b", "tnc"]:
                options["maxfun"] = self.max_eval
            else:
                options["maxfev"] = self.max_eval
            res = scipy.optimize.minimize(self.eval, x0, method=self.solver, bounds=bounds, constraints=constraints, options=options)
            success = res.success
        return success, np.array(self.fun_history, copy=True), np.array(self.maxcv_history, copy=True)

    def validate(self):
        valid_solvers = {"cobyla", "cobyqa", "cobyqa-old", "pdfo", "slsqp", "sqpdfo"}
        if self.problem.type not in "quadratic other":
            valid_solvers.update({"lincoa"})
            if self.problem.type not in "adjacency linear":
                valid_solvers.update({"bobyqa", "l-bfgs-b", "nelder-mead", "py-bobyqa", "tnc"})
                if self.problem.type not in "equality bound":
                    valid_solvers.update({"bfgs", "cg", "newuoa", "uobyqa"})
        return self.solver.lower() in valid_solvers

    def eval(self, x):
        f = self.problem.fun(x, self.callback, *self.args, **self.kwargs)
        if self.callback is not None:
            # If a noise function is supplied, the objective function returns
            # both the plain and the noisy function evaluations. We return the
            # noisy function evaluation, but we store the plain function
            # evaluation (used to build the performance and data profiles).
            self.fun_history.append(f[0])
            f = f[1]
        else:
            self.fun_history.append(f)
        self.maxcv_history.append(self.problem.maxcv(x))
        return f
