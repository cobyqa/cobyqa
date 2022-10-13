import re
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from cycler import cycler
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from matplotlib.backends import backend_pdf
from matplotlib.ticker import MultipleLocator

from .optimize import Minimizer
from .problems import Problems
from .utils import get_logger


class Profiles:

    BASE_DIR = Path(__file__).resolve(True).parent.parent
    ARCH_DIR = Path(BASE_DIR, 'archives')

    def __init__(self, n_min, n_max, constraints, m_min=0, m_max=np.inf, feature='plain', callback=None, **kwargs):
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max
        self.max_eval = 500 * self.n_max
        self.constraints = constraints
        self.feature = feature
        self.callback = callback

        # Extract from the keyword arguments the feature options.
        self.feature_options = self.get_feature_options(**kwargs)

        # Determinate the paths of storage.
        n_range = f'{self.n_min}-{self.n_max}'
        self.perf_dir = Path(self.ARCH_DIR, 'performance', self.feature, n_range)
        self.data_dir = Path(self.ARCH_DIR, 'data', self.feature, n_range)
        self.storage_dir = Path(self.ARCH_DIR, 'storage', self.feature)
        if self.feature != 'plain':
            options_suffix = dict(self.feature_options)
            if self.feature != 'noisy':
                del options_suffix['rerun']
            if self.feature in ['Lq', 'Lh', 'L1']:
                del options_suffix['p']
            options_details = '_'.join(f'{k}-{v}' for k, v in options_suffix.items())
            self.perf_dir = Path(self.perf_dir, options_details)
            self.data_dir = Path(self.data_dir, options_details)
            self.storage_dir = Path(self.storage_dir, options_details)

        # Load the CUTEst problems.
        logger = get_logger(__name__)
        self.problems = Problems(self.n_min, self.n_max, self.m_min, self.m_max, self.constraints, self.callback)
        logger.info('Problem(s) successfully loaded: %d', len(self.problems))

        # Set up matplotlib for plotting the profiles.
        standard_cycle = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
        standard_cycle += cycler(linestyle=['-', '--', ':', '-.'])
        plt.rc('axes', prop_cycle=standard_cycle)
        plt.rc('lines', linewidth=1)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    def __call__(self, solvers, options=None, load=True, **kwargs):
        solvers = list(map(str.lower, solvers))
        if options is None:
            options = [{} for _ in range(len(solvers))]

        self.perf_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.generate_profiles(solvers, options, load, **kwargs)

    def get_feature_options(self, **kwargs):
        significant = re.match(r'digits(\d+)', self.feature)
        options = {'rerun': 1}
        if self.feature in ['Lq', 'Lh', 'L1']:
            options['p'] = {'Lq': 0.25, 'Lh': 0.5, 'L1': 1.0}.get(self.feature)
            options['level'] = kwargs.get('regularization', 1.0)
        elif self.feature == 'noisy':
            options['type'] = kwargs.get('noise_type', 'relative')
            options['level'] = kwargs.get('noise_level', 1e-3)
            options['rerun'] = int(kwargs.get('rerun', 10))
        elif significant:
            self.feature = 'digits'
            options['digits'] = int(significant.group(1))
        elif self.feature != 'plain':
            raise NotImplementedError
        return options

    def get_profiles_path(self, solvers):
        if not isinstance(solvers, str):
            solvers = '_'.join(sorted(solvers))
        pdf_perf_path = Path(self.perf_dir, f'perf-{solvers}-{self.constraints}.pdf')
        pdf_data_path = Path(self.data_dir, f'data-{solvers}-{self.constraints}.pdf')
        return pdf_perf_path, pdf_data_path

    def get_storage_path(self, problem, solver, k):
        if problem.sifParams is None:
            cache = Path(self.storage_dir, problem.name)
        else:
            sif = '_'.join(f'{k}{v}' for k, v in problem.sifParams.items())
            cache = Path(self.storage_dir, f'{problem.name}_{sif}')
        cache.mkdir(exist_ok=True)
        if self.feature_options['rerun'] == 1:
            fun_path = Path(cache, f'fun-hist-{solver.lower()}.npy')
            constraint_violation_path = Path(cache, f'maxcv-hist-{solver.lower()}.npy')
            var_path = Path(cache, f'var-hist-{solver.lower()}.npy')
        else:
            fun_path = Path(cache, f'fun-hist-{solver.lower()}-{k}.npy')
            constraint_violation_path = Path(cache, f'maxcv-hist-{solver.lower()}-{k}.npy')
            var_path = Path(cache, f'var-hist-{solver.lower()}-{k}.npy')
        return fun_path, constraint_violation_path, var_path

    def generate_profiles(self, solvers, options, load, **kwargs):
        logger = get_logger(__name__)
        logger.info('Starting the computations (feature is %s)', self.feature)
        merit_values = self.run_all(solvers, options, load, **kwargs)

        log_precision_min = 1
        log_precision_max = 9
        penalty = 2
        ratio_max = 1e-6

        f0 = np.empty((len(self.problems), self.feature_options['rerun']), dtype=float)
        for i, p in enumerate(self.problems):
            for k in range(self.feature_options['rerun']):
                fun = p.fun(p.x0, self.noise, k)
                if hasattr(fun, '__len__') and len(fun) == 2:
                    fun = fun[0]
                constraint_violation = p.constraint_violation(p.x0)
                f0[i, k] = self.merit(fun, constraint_violation, barrier=False, **kwargs)

        f_min = np.min(merit_values, (1, 2, 3))
        if self.feature in ['digits', 'noisy']:
            logger.info('Starting the computations on the plain problems')
            rerun_sav = self.feature_options['rerun']
            feature_sav = self.feature
            self.feature_options['rerun'] = 1
            self.feature = 'plain'
            merit_values_plain = self.run_all(solvers, options, load, **kwargs)
            f_min_plain = np.min(merit_values_plain, (1, 2, 3))
            f_min = np.minimum(f_min, f_min_plain)
            self.feature_options['rerun'] = rerun_sav
            self.feature = feature_sav

        pdf_perf_path, pdf_data_path = self.get_profiles_path(solvers)
        pdf_perf = backend_pdf.PdfPages(pdf_perf_path)
        pdf_data = backend_pdf.PdfPages(pdf_data_path)
        for log_precision in range(log_precision_min, log_precision_max + 1):
            logger.info('Creating performance and data profiles with tau = 1e-%d', log_precision)
            tau = 10 ** (-log_precision)

            work = np.full((len(self.problems), len(solvers), self.feature_options['rerun']), np.nan)
            for i in range(len(self.problems)):
                for j in range(len(solvers)):
                    for k in range(self.feature_options['rerun']):
                        if np.isfinite(f_min[i]):
                            threshold = max(tau * f0[i, k] + (1.0 - tau) * f_min[i], f_min[i])
                        else:
                            threshold = -np.inf
                        if np.min(merit_values[i, j, k, :]) <= threshold:
                            index = np.argmax(merit_values[i, j, k, :] <= threshold)
                            work[i, j, k] = index + 1
            work = np.mean(work, -1)

            perf = np.full((len(self.problems), len(solvers)), np.nan)
            for i in range(len(self.problems)):
                if not np.all(np.isnan(work[i, :])):
                    perf[i, :] = work[i, :] / np.nanmin(work[i, :])
            perf = np.maximum(np.log2(perf), 0.0)
            perf_ratio_max = np.nanmax(perf, initial=ratio_max)
            perf[np.isnan(perf)] = penalty * perf_ratio_max
            perf = np.sort(perf, 0)

            data = np.full((len(self.problems), len(solvers)), np.nan)
            for i, p in enumerate(self.problems):
                if not np.all(np.isnan(work[i, :])):
                    data[i, :] = work[i, :] / (p.n + 1)
            data_ratio_max = np.nanmax(data, initial=ratio_max)
            data[np.isnan(data)] = penalty * data_ratio_max
            data = np.sort(data, 0)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params(direction='in', which='both')
            y = np.linspace(1 / len(self.problems), 1.0, len(self.problems))
            y = np.repeat(y, 2)[:-1]
            y = np.r_[0, 0, y, y[-1]]
            for j, solver in enumerate(solvers):
                x = np.repeat(perf[:, j], 2)[1:]
                x = np.r_[0, x[0], x, penalty * perf_ratio_max]
                plt.plot(x, y, label=solvers[j])
            plt.xlim(0, 1.1 * perf_ratio_max)
            plt.ylim(0, 1)
            plt.xlabel(r'$\log_2(\mathrm{NF}/\mathrm{NF}_{\min})$')
            plt.ylabel(fr'Performance profile ($\tau=10^{{-{log_precision}}}$)')
            plt.legend(loc='lower right')
            pdf_perf.savefig(fig, bbox_inches='tight')
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params(direction='in', which='both')
            y = np.linspace(1 / len(self.problems), 1.0, len(self.problems))
            y = np.repeat(y, 2)[:-1]
            y = np.r_[0, 0, y, y[-1]]
            for j, solver in enumerate(solvers):
                x = np.repeat(data[:, j], 2)[1:]
                x = np.r_[0, x[0], x, penalty * data_ratio_max]
                plt.plot(x, y, label=solvers[j])
            plt.xlim(0, 1.1 * data_ratio_max)
            plt.ylim(0, 1)
            plt.xlabel(r'Number of simplex gradients')
            plt.ylabel(fr'Data profile ($\tau=10^{{-{log_precision}}}$)')
            plt.legend(loc='lower right')
            pdf_data.savefig(fig, bbox_inches='tight')
            plt.close()
        pdf_perf.close()
        pdf_data.close()

    def run_all(self, solvers, options, load, **kwargs):
        merit_values = np.empty((len(self.problems), len(solvers), self.feature_options['rerun'], self.max_eval))
        result = Parallel(n_jobs=-1)(self.run_one(problem, solver, k, options[j], load, **kwargs) for problem, (j, solver), k in product(self.problems, enumerate(solvers), range(self.feature_options['rerun'])))
        for i in range(len(self.problems)):
            for j in range(len(solvers)):
                for k in range(self.feature_options['rerun']):
                    index = (i * len(solvers) + j) * self.feature_options['rerun'] + k
                    merit_values_run, n_eval = result[index]
                    merit_values[i, j, k, :n_eval] = merit_values_run[:n_eval]
                    merit_values[i, j, k, n_eval:] = merit_values[i, j, k, n_eval - 1]
        return merit_values

    @delayed
    def run_one(self, problem, solver, k, options, load, **kwargs):
        fun_path, constraint_violation_path, var_path = self.get_storage_path(problem, solver, k)
        max_eval = 500 * problem.n
        merit_values, n_eval = None, 0
        if load and fun_path.is_file() and constraint_violation_path.is_file() and var_path.is_file():
            fun_history = np.load(fun_path)
            constraint_violation_history = np.load(constraint_violation_path)
            merit_values = self.merit(fun_history, constraint_violation_history, **kwargs)
            n_eval = merit_values.size
            if max_eval > n_eval:
                var = np.load(var_path)
                if var[0]:
                    merit_values = np.r_[merit_values, np.full(max_eval - n_eval, np.inf)]
                else:
                    merit_values, n_eval = None, 0
            elif max_eval < n_eval:
                n_eval = max_eval
                merit_values = merit_values[:n_eval]
        if merit_values is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                optimizer = Minimizer(problem, solver, max_eval, options, self.noise, k)
                success, fun_history, constraint_violation_history = optimizer()
            n_eval = min(fun_history.size, max_eval)
            merit_values = np.full(max_eval, np.inf)
            merit_values[:n_eval] = self.merit(fun_history[:n_eval], constraint_violation_history[:n_eval], **kwargs)
            np.save(fun_path, fun_history[:n_eval])
            np.save(constraint_violation_path, constraint_violation_history[:n_eval])
            np.save(var_path, np.array([success]))
        logger = get_logger(__name__)
        if not np.all(np.isnan(merit_values)):
            if self.feature_options['rerun'] > 1:
                run_description = f'{solver}({problem.name},{k})'
            else:
                run_description = f'{solver}({problem.name})'
            i = np.argmin(merit_values[:n_eval])
            logger.info('%s: fun = %f, maxcv = %f, n_eval = %d', run_description, fun_history[i], constraint_violation_history[i], n_eval)
        else:
            logger.warning('%s(%s): no value received', solver, problem.name)
        return merit_values, n_eval

    @staticmethod
    def merit(fun_history, constraint_violation_history, **kwargs):
        fun_history = np.atleast_1d(fun_history)
        constraint_violation_history = np.atleast_1d(constraint_violation_history)
        merit_values = np.empty_like(fun_history)
        for i in range(merit_values.size):
            if constraint_violation_history[i] <= kwargs.get('low_cv', 1e-12):
                merit_values[i] = fun_history[i]
            elif kwargs.get('barrier', False) and constraint_violation_history[i] >= kwargs.get('high_cv', 1e-6):
                merit_values[i] = np.inf
            else:
                merit_values[i] = fun_history[i] + kwargs.get('penalty', 1e8) * constraint_violation_history[i]
        return merit_values

    def noise(self, x, f, k=0):
        if self.feature in ['Lq', 'Lh', 'L1']:
            f += self.feature_options['level'] * np.linalg.norm(x, self.feature_options['p'])
        elif self.feature == 'noisy':
            rng = np.random.default_rng(int(1e8 * abs(np.sin(k) + np.sin(self.feature_options['level']) + np.sum(np.sin(np.abs(np.sin(1e8 * x)))))))
            noise = self.feature_options['level'] * rng.standard_normal()
            if self.feature_options['type'] == 'absolute':
                f += noise
            else:
                f *= 1.0 + noise
        elif self.feature == 'digits' and np.isfinite(f):
            if f == 0.0:
                fx_rounded = 0.0
            else:
                fx_rounded = round(f, self.feature_options['digits'] - int(np.floor(np.log10(np.abs(f)))) - 1)
            f = fx_rounded + (f - fx_rounded) * np.abs(np.sin(np.sin(np.sin(self.feature_options['digits'])) + np.sin(1e8 * f) + np.sum(np.sin(np.abs(1e8 * x))) + np.sin(x.size)))
        return f
