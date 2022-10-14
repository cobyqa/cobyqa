import re
import subprocess
import warnings

import numpy as np
import pycutest
from joblib import delayed, Parallel
from numpy import ma
from scipy.linalg import lstsq
from scipy.optimize import Bounds, LinearConstraint, minimize

from .utils import get_logger


class Problems(list):

    EXCLUDED = {}

    def __init__(self, n_min, n_max, m_min, m_max, constraints, callback=None):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max
        names = pycutest.find_problems(objective='constant linear quadratic sum of squares other', constraints=constraints, regular=True, origin='academic modelling real-world', n=[self.n_min, self.n_max], m=[self.m_min, self.m_max], userM=False)

        # The problem instances are not appended yet, because it cannot be
        # Get all problem instances. If a failure occurred when loading a
        # problem, the corresponding instance is replaced with None. They are
        # not appended to the object as this cannot be done in a multiprocessing
        # paradigm. They are appended below in a sequential paradigm.
        attempts = Parallel(n_jobs=-1)(self.load(sorted(names), i) for i in range(len(names)))
        for problem in attempts:
            if problem is not None:
                self.append(problem, callback)

    def append(self, problem, callback=None):
        if self.validate(problem, callback):
            super().append(problem)

    @delayed
    def load(self, names, i):
        logger = get_logger(__name__)
        try:
            if names[i] not in self.EXCLUDED:
                logger.info(f'Loading {names[i]} ({i + 1}/{len(names)})')

                # If the problem's dimension is not fixed, we select the largest
                # available dimension that matches the requirements.
                if pycutest.problem_properties(names[i])['n'] == 'variable':
                    sif_n = self.get_sif_n(names[i])

                    # Since PyCUTEst removes the fixed variables, the reduced
                    # problem's dimension may not satisfy the requirements.
                    sif_n_masked = ma.masked_array(sif_n, mask=(sif_n < self.n_min) | (sif_n > self.n_max))
                    if sif_n_masked.size > 0:
                        sif_n_max = sif_n_masked.max()
                        if sif_n_max is not ma.masked:
                            return Problem(names[i], sifParams={'N': sif_n_max})
                else:
                    return Problem(names[i])
        except (AttributeError, FileNotFoundError, ModuleNotFoundError, RuntimeError) as err:
            logger.warning(f'{names[i]}: {err}')

    def validate(self, problem, callback=None):
        valid = np.all(problem.vartype == 0)
        valid = valid and self.n_min <= problem.n <= self.n_max
        valid = valid and self.m_min <= problem.m <= self.m_max
        if callback is not None:
            valid = valid and callback(problem)
        return valid

    @staticmethod
    def get_sif_n(name):
        # Get all the available SIF parameters for all variables.
        cmd = [pycutest.get_sifdecoder_path(), '-show', name]
        sp = subprocess.Popen(cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        sif_stdout = sp.stdout.read()
        sp.wait()

        # Extract all the available SIF parameters for the problem's dimension.
        regex = re.compile(r'^N=(?P<param>\d+)')
        sif_n = []
        for stdout in sif_stdout.split('\n'):
            sif_match = regex.match(stdout)
            if sif_match:
                sif_n.append(int(sif_match.group('param')))
        return np.sort(sif_n)


class Problem:

    def __init__(self, *args, **kwargs):
        problem = pycutest.import_problem(*args, **kwargs)

        self.name = problem.name
        self.n = problem.n
        self.m = problem.m
        self.x0 = np.array(problem.x0, copy=True)
        self.sifParams = problem.sifParams
        self.vartype = np.array(problem.vartype, copy=True)
        self.xl = np.array(problem.bl, copy=True)
        self.xu = np.array(problem.bu, copy=True)
        self.xl[self.xl <= -1e20] = -np.inf
        self.xu[self.xu >= 1e20] = np.inf
        self.cl = problem.cl
        self.cu = problem.cu
        if self.m > 0:
            self.cl[self.cl <= -1e20] = -np.inf
            self.cu[self.cu >= 1e20] = np.inf
        self.is_eq_cons = problem.is_eq_cons
        self.is_linear_cons = problem.is_linear_cons

        self.obj = problem.obj
        self.cons = problem.cons

        # The following attributes can be built from other attributes. However,
        # they may be time-consuming to build. Therefore, we construct them only
        # when they are accessed for the first time.
        self._a_ineq = None
        self._b_ineq = None
        self._a_eq = None
        self._b_eq = None
        self._m_lin_ineq = None
        self._m_lin_eq = None
        self._m_nonlin_ineq = None
        self._m_nonlin_eq = None

        # Project the initial guess only the feasible polyhedron (including the
        # bound and the linear constraints).
        self.project_x0()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def a_ineq(self):
        if self._a_ineq is None:
            self._a_ineq, self._b_ineq = self.build_lin_ineq_cons()
        return self._a_ineq

    @property
    def b_ineq(self):
        if self._b_ineq is None:
            self._a_ineq, self._b_ineq = self.build_lin_ineq_cons()
        return self._b_ineq

    @property
    def a_eq(self):
        if self._a_eq is None:
            self._a_eq, self._b_eq = self.build_lin_eq_cons()
        return self._a_eq

    @property
    def b_eq(self):
        if self._b_eq is None:
            self._a_eq, self._b_eq = self.build_lin_eq_cons()
        return self._b_eq

    @property
    def m_lin_ineq(self):
        if self._m_lin_ineq is None:
            if self.m == 0:
                self._m_lin_ineq = 0
            else:
                iub = self.is_linear_cons & np.logical_not(self.is_eq_cons)
                iub_cl = self.cl[iub] >= -np.inf
                iub_cu = self.cu[iub] < np.inf
                self._m_lin_ineq = np.count_nonzero(iub_cl) + np.count_nonzero(iub_cu)
        return self._m_lin_ineq

    @property
    def m_lin_eq(self):
        if self._m_lin_eq is None:
            if self.m == 0:
                self._m_lin_eq = 0
            else:
                ieq = self.is_linear_cons & self.is_eq_cons
                self._m_lin_eq = np.count_nonzero(ieq)
        return self._m_lin_eq

    @property
    def m_nonlin_ineq(self):
        if self._m_nonlin_ineq is None:
            if self.m == 0:
                self._m_nonlin_ineq = 0
            else:
                iub = np.logical_not(self.is_linear_cons | self.is_eq_cons)
                iub_cl = self.cl[iub] > -np.inf
                iub_cu = self.cu[iub] < np.inf
                self._m_nonlin_ineq = np.count_nonzero(iub_cl) + np.count_nonzero(iub_cu)
        return self._m_nonlin_ineq

    @property
    def m_nonlin_eq(self):
        if self._m_nonlin_eq is None:
            if self.m == 0:
                self._m_nonlin_eq = 0
            else:
                ieq = np.logical_not(self.is_linear_cons) & self.is_eq_cons
                self._m_nonlin_eq = np.count_nonzero(ieq)
        return self._m_nonlin_eq

    @property
    def type(self):
        properties = pycutest.problem_properties(self.name)
        return properties.get('constraints')

    def fun(self, x, callback=None, *args, **kwargs):
        x = np.asarray(x, dtype=float)
        f = self.obj(x)

        # If a noise function is supplied, return both the plain and the noisy
        # function evaluations, the former being required for building the
        # performance and data profiles.
        if callback is not None:
            return f, callback(x, f, *args, **kwargs)
        return f

    def c_ineq(self, x):
        if self.m == 0:
            return np.empty(0)
        x = np.asarray(x, dtype=float)
        iub = np.logical_not(self.is_linear_cons | self.is_eq_cons)
        iub_cl = self.cl[iub] > -np.inf
        iub_cu = self.cu[iub] < np.inf
        c = []
        for i, index in enumerate(np.flatnonzero(iub)):
            c_index = self.cons(x, index)
            if iub_cl[i]:
                c.append(self.cl[index] - c_index)
            if iub_cu[i]:
                c.append(c_index - self.cu[index])
        return np.array(c, dtype=float)

    def c_eq(self, x):
        if self.m == 0:
            return np.empty(0)
        x = np.asarray(x, dtype=float)
        ieq = np.logical_not(self.is_linear_cons) & self.is_eq_cons
        c = []
        for index in np.flatnonzero(ieq):
            c_index = self.cons(x, index)
            c.append(c_index - 0.5 * (self.cl[index] + self.cu[index]))
        return np.array(c, dtype=float)

    def maxcv(self, x):
        maxcv = np.max(self.xl - x, initial=0.0)
        maxcv = np.max(x - self.xu, initial=maxcv)
        maxcv = np.max(np.dot(self.a_ineq, x) - self.b_ineq, initial=maxcv)
        maxcv = np.max(np.abs(np.dot(self.a_eq, x) - self.b_eq), initial=maxcv)
        maxcv = np.max(self.c_ineq(x), initial=maxcv)
        maxcv = np.max(np.abs(self.c_eq(x)), initial=maxcv)
        return maxcv

    def build_lin_ineq_cons(self):
        if self.m == 0:
            return np.empty((0, self.n)), np.empty(0)
        iub = self.is_linear_cons & np.logical_not(self.is_eq_cons)
        iub_cl = self.cl[iub] > -np.inf
        iub_cu = self.cu[iub] < np.inf
        aub = []
        bub = []
        for i, index in enumerate(np.flatnonzero(iub)):
            c_index, g_index = self.cons(np.zeros(self.n), index, True)
            if iub_cl[i]:
                aub.append(-g_index)
                bub.append(c_index - self.cl[index])
            if iub_cu[i]:
                aub.append(g_index)
                bub.append(self.cu[index] - c_index)
        return np.reshape(aub, (-1, self.n)), np.array(bub)

    def build_lin_eq_cons(self):
        if self.m == 0:
            return np.empty((0, self.n)), np.empty(0)
        ieq = self.is_linear_cons & self.is_eq_cons
        aeq = []
        beq = []
        for index in np.flatnonzero(ieq):
            c_index, g_index = self.cons(np.zeros(self.n), index, True)
            aeq.append(g_index)
            beq.append(c_index - 0.5 * (self.cl[index] + self.cu[index]))
        return np.reshape(aeq, (-1, self.n)), np.array(beq)

    def project_x0(self):
        if self.m == 0:
            self.x0 = np.minimum(self.xu, np.maximum(self.xl, self.x0))
        elif self.m_lin_ineq == 0 and self.m_lin_eq > 0 and np.all(self.xl == -np.inf) and np.all(self.xu == np.inf):
            self.x0 += lstsq(self.a_eq, self.b_eq - np.dot(self.a_eq, self.x0))[0]
        else:
            bounds = Bounds(self.xl, self.xu, True)
            constraints = []
            if self.m_lin_ineq > 0:
                constraints.append(LinearConstraint(self.a_ineq, -np.inf, self.b_ineq))
            if self.m_lin_eq > 0:
                constraints.append(LinearConstraint(self.a_eq, self.b_eq, self.b_eq))

            def distance_square(x):
                g = x - self.x0
                f = 0.5 * np.inner(x - self.x0, x - self.x0)
                return f, g

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = minimize(distance_square, self.x0, jac=True, bounds=bounds, constraints=constraints)
            self.x0 = np.array(res.x)
