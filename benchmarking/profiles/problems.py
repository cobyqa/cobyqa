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
                logger.info('Loading %s (%d/%d)', names[i], i + 1, len(names))
                if pycutest.problem_properties(names[i])['n'] == 'variable':
                    sif_n = self.get_sif_n(names[i])
                    sif_n_masked = ma.masked_array(sif_n, mask=(sif_n < self.n_min) | (sif_n > self.n_max))
                    if sif_n_masked.size > 0:
                        sif_n_max = sif_n_masked.max()
                        if sif_n_max is not ma.masked:
                            return Problem(names[i], sifParams={'N': sif_n_max})
                else:
                    return Problem(names[i])
        except (AttributeError, ModuleNotFoundError, RuntimeError) as err:
            logger.warning(err)

    def validate(self, problem, callback=None):
        valid = np.all(problem.vartype == 0)
        valid = valid and self.n_min <= problem.n <= self.n_max
        valid = valid and self.m_min <= problem.m <= self.m_max
        if callback is not None:
            valid = valid and callback(problem)
        return valid

    @staticmethod
    def get_sif_n(name):
        cmd = [pycutest.get_sifdecoder_path(), '-show', name]
        sp = subprocess.Popen(cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        sif_stdout = sp.stdout.read()
        sp.wait()

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
        self.xl[self.xl <= -1e20] = -np.inf
        self.xu = np.array(problem.bu, copy=True)
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

        self._a_inequality = None
        self._b_inequality = None
        self._a_equality = None
        self._b_equality = None
        self._m_linear_inequality = None
        self._m_linear_equality = None
        self._m_nonlinear_inequality = None
        self._m_nonlinear_equality = None

        self.project_x0()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def a_inequality(self):
        if self._a_inequality is None:
            self._a_inequality, self._b_inequality = self.build_linear_inequality_constraints()
        return self._a_inequality

    @property
    def b_inequality(self):
        if self._b_inequality is None:
            self._a_inequality, self._b_inequality = self.build_linear_inequality_constraints()
        return self._b_inequality

    @property
    def a_equality(self):
        if self._a_equality is None:
            self._a_equality, self._b_equality = self.build_linear_equality_constraints()
        return self._a_equality

    @property
    def b_equality(self):
        if self._b_equality is None:
            self._a_equality, self._b_equality = self.build_linear_equality_constraints()
        return self._b_equality

    @property
    def m_linear_inequality(self):
        if self._m_linear_inequality is None:
            if self.m == 0:
                self._m_linear_inequality = 0
            else:
                iub = self.is_linear_cons & np.logical_not(self.is_eq_cons)
                iub_cl = self.cl[iub] >= -np.inf
                iub_cu = self.cu[iub] < np.inf
                self._m_linear_inequality = np.count_nonzero(iub_cl) + np.count_nonzero(iub_cu)
        return self._m_linear_inequality

    @property
    def m_linear_equality(self):
        if self._m_linear_equality is None:
            if self.m == 0:
                self._m_linear_equality = 0
            else:
                ieq = self.is_linear_cons & self.is_eq_cons
                self._m_linear_equality = np.count_nonzero(ieq)
        return self._m_linear_equality

    @property
    def m_nonlinear_inequality(self):
        if self._m_nonlinear_inequality is None:
            if self.m == 0:
                self._m_nonlinear_inequality = 0
            else:
                iub = np.logical_not(self.is_linear_cons | self.is_eq_cons)
                iub_cl = self.cl[iub] > -np.inf
                iub_cu = self.cu[iub] < np.inf
                self._m_nonlinear_inequality = np.count_nonzero(iub_cl) + np.count_nonzero(iub_cu)
        return self._m_nonlinear_inequality

    @property
    def m_nonlinear_equality(self):
        if self._m_nonlinear_equality is None:
            if self.m == 0:
                self._m_nonlinear_equality = 0
            else:
                ieq = np.logical_not(self.is_linear_cons) & self.is_eq_cons
                self._m_nonlinear_equality = np.count_nonzero(ieq)
        return self._m_nonlinear_equality

    @property
    def type(self):
        properties = pycutest.problem_properties(self.name)
        return properties.get('constraints')

    def fun(self, x, callback=None, *args, **kwargs):
        x = np.asarray(x, dtype=float)
        f = self.obj(x)
        if callback is not None:
            return f, callback(x, f, *args, **kwargs)
        return f

    def c_inequality(self, x):
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

    def c_equality(self, x):
        if self.m == 0:
            return np.empty(0)
        x = np.asarray(x, dtype=float)
        ieq = np.logical_not(self.is_linear_cons) & self.is_eq_cons
        c = []
        for index in np.flatnonzero(ieq):
            c_index = self.cons(x, index)
            c.append(c_index - 0.5 * (self.cl[index] + self.cu[index]))
        return np.array(c, dtype=float)

    def constraint_violation(self, x):
        maximum_violation = np.max(self.xl - x, initial=0.0)
        maximum_violation = np.max(x - self.xu, initial=maximum_violation)
        maximum_violation = np.max(np.dot(self.a_inequality, x) - self.b_inequality, initial=maximum_violation)
        maximum_violation = np.max(np.abs(np.dot(self.a_equality, x) - self.b_equality), initial=maximum_violation)
        maximum_violation = np.max(self.c_inequality(x), initial=maximum_violation)
        maximum_violation = np.max(np.abs(self.c_equality(x)), initial=maximum_violation)
        return maximum_violation

    def build_linear_inequality_constraints(self):
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

    def build_linear_equality_constraints(self):
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
        elif self.m_linear_inequality == 0 and self.m_linear_equality > 0 and np.all(self.xl == -np.inf) and np.all(self.xu == np.inf):
            self.x0 += lstsq(self.a_equality, self.b_equality - np.dot(self.a_equality, self.x0))[0]
        else:
            bounds = Bounds(self.xl, self.xu, True)
            constraints = []
            if self.m_linear_inequality > 0:
                constraints.append(LinearConstraint(self.a_inequality, -np.inf, self.b_inequality))
            if self.m_linear_equality > 0:
                constraints.append(LinearConstraint(self.a_equality, self.b_equality, self.b_equality))

            def distance_square(x):
                g = x - self.x0
                f = 0.5 * np.inner(x - self.x0, x - self.x0)
                return f, g

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = minimize(distance_square, self.x0, jac=True, bounds=bounds, constraints=constraints)
            self.x0 = np.array(res.x)
