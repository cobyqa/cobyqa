import re
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pycutest
from cycler import cycler
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.backends import backend_pdf
from matplotlib.ticker import MaxNLocator

from .optimize import Optimizer
from .utils import get_logger


# Set up matplotlib for plotting the profiles.
prop_cycle = plt.rcParams['axes.prop_cycle']
prop_cycle += cycler(linestyle=[(0, ()), (0, (5, 3)), (0, (1, 1)), (0, (5, 3, 1, 3)), (0, (5, 6)), (0, (1, 3)), (0, (5, 6, 1, 6))])
plt.rc('axes', prop_cycle=prop_cycle)
plt.rc('lines', linewidth=1)
plt.rc('font', size=14)


class Profiles:

    BASE_DIR = Path(__file__).resolve(True).parent.parent
    ARCH_DIR = Path(BASE_DIR, 'archives')
    EXCLUDED_PROBLEMS = {
        # The compilation of the sources is prohibitively time-consuming.
        'BA-L73', 'BA-L73LS', 'BDRY2', 'CHANDHEU', 'CHARDIS0', 'CHARDIS1', 'DMN15102', 'DMN15102LS', 'DMN15103', 'DMN15103LS', 'DMN15332', 'DMN15332LS', 'DMN15333', 'DMN15333LS', 'DMN37142', 'DMN37142LS', 'DMN37143', 'DMN37143LS', 'GPP', 'LEUVEN3', 'LEUVEN4', 'LEUVEN5', 'LEUVEN6', 'LIPPERT2', 'LOBSTERZ', 'PDE1', 'PDE2', 'PENALTY3', 'RDW2D51F', 'RDW2D51U', 'RDW2D52B', 'RDW2D52F', 'RDW2D52U', 'ROSEPETAL', 'WALL100', 'YATP1SQ', 'YATP2SQ', 'BA-L16LS', 'BA-L21', 'BA-L21LS', 'BA-L49', 'BA-L49LS', 'BA-L52LS', 'BA-L52',

        # The starting points contain NaN values.
        'LHAIFAM',

        # The problems contain a lot of NaN.
        'HS62', 'HS112', 'LIN',

        # The problems seem not lower-bounded.
        'INDEF',

        # The problems are known infeasible.
        'ARGLALE', 'ARGLBLE', 'ARGLCLE', 'MODEL', 'NASH',

        # Classical UOBYQA and COBYLA suffer from infinite cycling.
        'GAUSS1LS', 'GAUSS2LS', 'GAUSS3LS', 'MGH17LS', 'MISRA1ALS', 'MISRA1CLS', 'NELSONLS', 'OSBORNEA', 'RAT43LS',

        # Classical COBYLA suffers from infinite cycling.
        'DANWOODLS', 'KOEBHELB',
    }

    def __init__(self, n_min, n_max, constraints, m_min=0, m_max=1000, feature='plain', callback=None, **kwargs):
        # All features:
        # 1. plain: the problems are left unmodified.
        # 2. Lq, Lh, L1: n p-regularization term is added to the objective
        #   functions of all problems, with p = 0.25, 0.5, and 1, respectively.
        #   The following keyword arguments may be supplied:
        #   2.1. regularization: corresponding parameter (default is 1.0).
        # 3. noisy: a Gaussian noise is included in the objective functions of
        #   all problems. The following keyword arguments may be supplied:
        #   3.1. noise_type: noise type (default is relative).
        #   3.2. noise_level: standard deviation of the noise (default is 1e-3).
        #   3.3. rerun: number of experiment runs (default is 10).
        # 4. digits[0-9]+: only the first digits of the objective function
        #   values are significant (the other are randomized).
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
        self.perf_dir = Path(self.ARCH_DIR, 'perf', self.feature, f'n{self.n_min}-{self.n_max}')
        self.data_dir = Path(self.ARCH_DIR, 'data', self.feature, f'n{self.n_min}-{self.n_max}')
        self.cache_dir = Path(self.ARCH_DIR, 'cache', self.feature)
        if self.feature != 'plain':
            # Suffix the feature's directory name with the corresponding
            # feature's options. We exclude the options that are redundant with
            # the feature (e.g., if feature='Lq', then p=0.25).
            options_suffix = dict(self.feature_options)
            if self.feature not in ['noisy', 'nan', 'digits']:
                del options_suffix['rerun']
            if self.feature in ['Lq', 'Lh', 'L1']:
                del options_suffix['p']
            options_details = '_'.join(f'{k}-{v}' for k, v in options_suffix.items())
            self.perf_dir = Path(self.perf_dir, options_details)
            self.data_dir = Path(self.data_dir, options_details)
            self.cache_dir = Path(self.cache_dir, options_details)

        # Fetch the names of the CUTEst problems that match the requirements.
        logger = get_logger(__name__)
        self.problem_names = self.get_problem_names()
        logger.info(f'Number of problems: {len(self.problem_names)}')

    def __call__(self, solvers, labels=None, options=None, load=True, **kwargs):
        if labels is None:
            labels = list(solvers)
        solvers = list(map(str.lower, solvers))
        if options is None:
            options = [{} for _ in range(len(solvers))]

        self.perf_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.generate_profiles(solvers, labels, options, load, **kwargs)

    def get_feature_options(self, **kwargs):
        significant_digits = re.match(r'digits(\d+)', self.feature)
        options = {'rerun': 1}
        if self.feature in ['Lq', 'Lh', 'L1']:
            options['p'] = {'Lq': 0.25, 'Lh': 0.5, 'L1': 1.0}.get(self.feature)
            options['level'] = float(kwargs.get('regularization', 1.0))
        elif self.feature == 'noisy':
            options['type'] = kwargs.get('noise_type', 'relative')
            options['level'] = float(kwargs.get('noise_level', 1e-3))
            options['rerun'] = int(kwargs.get('rerun', 10))
        elif self.feature == 'nan':
            options['rate'] = float(kwargs.get('nan_rate', 0.1))
            options['rerun'] = int(kwargs.get('rerun', 10))
        elif significant_digits:
            self.feature = 'digits'
            options['digits'] = int(significant_digits.group(1))
        elif self.feature != 'plain':
            raise NotImplementedError(f'Unknown feature "{self.feature}"')
        return options

    def get_problem_names(self):
        problem_names = pycutest.find_problems(objective='constant linear quadratic sum of squares other', constraints=self.constraints, regular=True, origin='academic modelling real-world', n=[self.n_min, self.n_max], m=[self.m_min, self.m_max], userM=False)
        return sorted(set(problem_names).difference(self.EXCLUDED_PROBLEMS))

    def get_storage_path(self, problem, solver, k):
        if problem.sifParams is None:
            cache = Path(self.cache_dir, problem.name)
        else:
            sif = '_'.join(f'{k}{v}' for k, v in problem.sifParams.items())
            cache = Path(self.cache_dir, f'{problem.name}_{sif}')
        cache.mkdir(exist_ok=True)
        if self.feature_options['rerun'] == 1:
            fun_path = Path(cache, f'fun-hist-{solver.lower()}.npy')
            resid_path = Path(cache, f'resid-hist-{solver.lower()}.npy')
            success_path = Path(cache, f'success-hist-{solver.lower()}.npy')
        else:
            fun_path = Path(cache, f'fun-hist-{solver.lower()}-{k}.npy')
            resid_path = Path(cache, f'resid-hist-{solver.lower()}-{k}.npy')
            success_path = Path(cache, f'success-hist-{solver.lower()}-{k}.npy')
        return fun_path, resid_path, success_path

    def get_profiles_path(self, solvers, precisions):
        if not isinstance(solvers, str):
            solvers = '_'.join(sorted(solvers))
        constraints = self.constraints.replace(' ', '_')
        pdf_perf_path = Path(self.perf_dir, f'perf-{solvers}-{constraints}.pdf')
        pdf_data_path = Path(self.data_dir, f'data-{solvers}-{constraints}.pdf')
        txt_perf_path = Path(self.perf_dir, f'perf-{solvers}-{constraints}.txt')
        txt_data_path = Path(self.data_dir, f'data-{solvers}-{constraints}.txt')
        eps_perf_path = []
        eps_data_path = []
        for precision in precisions:
            eps_perf_path.append(Path(self.perf_dir, f'perf-{solvers}-{constraints}-{precision}.eps'))
            eps_data_path.append(Path(self.data_dir, f'data-{solvers}-{constraints}-{precision}.eps'))
        return pdf_perf_path, eps_perf_path, txt_perf_path, pdf_data_path, eps_data_path, txt_data_path

    def generate_profiles(self, solvers, labels, options, load, **kwargs):
        # Solve the problems with the given solvers.
        logger = get_logger(__name__)
        logger.info(f'Starting the computations with feature="{self.feature}"')
        merit_values, problem_names, problem_dimensions = self.solve_all(solvers, options, load, **kwargs)
        n_problems, n_solvers, n_run, _ = merit_values.shape

        # Get the merit function values at x0.
        merit_x0 = np.nanmin(merit_values[:, :, :, 0], 1)

        # Determine the least merit function values obtained on each problem.
        merit_min = np.nanmin(merit_values, (1, 2, 3))
        if self.feature in ['digits', 'noisy', 'nan']:
            logger.info('Starting the computations with feature="plain"')
            feature = self.feature
            self.feature_options['rerun'] = 1
            self.feature = 'plain'
            merits_plain, _, _ = self.solve_all(solvers, options, load, **kwargs)
            merit_min_plain = np.nanmin(merits_plain, (1, 2, 3))
            merit_min = np.minimum(merit_min, merit_min_plain)
            self.feature_options['rerun'] = n_run
            self.feature = feature

        # Compute and save the performance and data profiles.
        precisions = np.arange(1, 10)
        pdf_perf_path, eps_perf_path, txt_perf_path, pdf_data_path, eps_data_path, txt_data_path = self.get_profiles_path(solvers, precisions)
        pdf_perf = backend_pdf.PdfPages(pdf_perf_path)
        pdf_data = backend_pdf.PdfPages(pdf_data_path)
        for i_precision, precision in enumerate(precisions):
            logger.info(f"Creating performance and data profiles with tau = 1e-{precision}")
            tau = 10.0 ** (-precision)

            # Determine the number of function evaluations employed by each
            # solver on each problem to reach the given accuracy.
            work = np.full((n_problems, n_solvers, n_run), np.nan)
            for i in range(n_problems):
                for j in range(n_solvers):
                    for k in range(n_run):
                        if np.isfinite(merit_min[i]):
                            threshold = max(tau * merit_x0[i, k] + (1.0 - tau) * merit_min[i], merit_min[i])
                        else:
                            threshold = -np.inf
                        if np.nanmin(merit_values[i, j, k, :]) <= threshold:
                            work[i, j, k] = np.argmax(merit_values[i, j, k, :] <= threshold) + 1

            # Calculate the x-axes performance profiles.
            x_perf = np.full((n_run, n_problems, n_solvers), np.nan)
            for k in range(n_run):
                for i in range(n_problems):
                    if not np.all(np.isnan(work[i, :, k])):
                        x_perf[k, i, :] = work[i, :, k] / np.nanmin(work[i, :, k])
            perf_ratio_max = np.nanmax(x_perf, initial=2.0 ** np.finfo(float).eps)
            x_perf[np.isnan(x_perf)] = 2.0 * perf_ratio_max
            x_perf = np.sort(x_perf, 1)
            x_perf = np.reshape(x_perf, (n_problems * n_run, n_solvers))
            sort_perf = np.argsort(x_perf, 0, 'stable')
            x_perf = np.take_along_axis(x_perf, sort_perf, 0)

            # Calculate the y-axes performance profiles.
            y_perf = np.zeros((n_problems * n_run, n_solvers))
            for k in range(n_run):
                for j in range(n_solvers):
                    y = np.full(n_problems * n_run, np.nan)
                    y[k * n_problems:(k + 1) * n_problems] = np.linspace(1 / n_problems, 1.0, n_problems)
                    y = y[sort_perf[:, j]]
                    for i in range(n_problems * n_run):
                        if np.isnan(y[i]):
                            y[i] = y[i - 1] if i > 0 else 0.0
                    y_perf[:, j] += y
            y_perf /= n_run

            # Calculate the x-axes data profiles.
            x_data = np.full((n_run, n_problems, n_solvers), np.nan)
            for k in range(n_run):
                for i in range(n_problems):
                    if not np.all(np.isnan(work[i, :, k])):
                        x_data[k, i, :] = work[i, :, k] / (problem_dimensions[i] + 1)
            data_ratio_max = np.nanmax(x_data, initial=np.finfo(float).eps)
            x_data[np.isnan(x_data)] = 2.0 * data_ratio_max
            x_data = np.sort(x_data, 1)
            x_data = np.reshape(x_data, (n_problems * n_run, n_solvers))
            sort_data = np.argsort(x_data, 0, 'stable')
            x_data = np.take_along_axis(x_data, sort_data, 0)

            # Calculate the y-axes performance profiles.
            y_data = np.zeros((n_problems * n_run, n_solvers))
            for k in range(n_run):
                for j in range(n_solvers):
                    y = np.full(n_problems * n_run, np.nan)
                    y[k * n_problems:(k + 1) * n_problems] = np.linspace(1 / n_problems, 1.0, n_problems)
                    y = y[sort_data[:, j]]
                    for i in range(n_problems * n_run):
                        if np.isnan(y[i]):
                            y[i] = y[i - 1] if i > 0 else 0.0
                    y_data[:, j] += y
            y_data /= n_run

            # Plot the performance profiles.
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(MaxNLocator(5, prune='lower'))
            ax.yaxis.set_minor_locator(MaxNLocator(10))
            ax.tick_params(direction='in', which='both')
            for j in range(len(solvers)):
                x = np.repeat(x_perf[:, j], 2)[1:]
                x = np.r_[0.0, x[0], x, 2.0 * perf_ratio_max]
                y = np.repeat(y_perf[:, j], 2)[:-1]
                y = np.r_[0.0, 0.0, y, y[-1]]
                plt.semilogx(x, y, base=2, label=labels[j])
            plt.xlim(1.0, 1.1 * perf_ratio_max)
            plt.ylim(0.0, 1.0)
            plt.xlabel('Performance ratio')
            plt.ylabel(fr'Performance profiles ($\tau=10^{{{int(np.log10(tau))}}}$)')
            plt.legend(handlelength=1.3, handletextpad=0.3, labelspacing=0.3, loc='lower right')
            fig.savefig(eps_perf_path[i_precision], bbox_inches='tight', format='eps')
            pdf_perf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Plot the data profiles.
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_major_locator(MaxNLocator(5, prune='lower'))
            ax.yaxis.set_minor_locator(MaxNLocator(10))
            ax.tick_params(direction='in', which='both')
            for j in range(len(solvers)):
                x = np.repeat(x_data[:, j], 2)[1:]
                x = np.r_[0.0, x[0], x, 2.0 * data_ratio_max]
                y = np.repeat(y_data[:, j], 2)[:-1]
                y = np.r_[0.0, 0.0, y, y[-1]]
                plt.plot(x, y, label=labels[j])
            plt.xlim(0.0, 1.1 * data_ratio_max)
            plt.ylim(0.0, 1.0)
            plt.xlabel('Number of simplex gradients')
            plt.ylabel(fr'Data profiles ($\tau=10^{{{int(np.log10(tau))}}}$)')
            plt.legend(handlelength=1.3, handletextpad=0.3, labelspacing=0.3, loc='lower right')
            fig.savefig(eps_data_path[i_precision], bbox_inches='tight', format='eps')
            pdf_data.savefig(fig, bbox_inches='tight')
            plt.close()

        # Save the PDFs and TXTs files.
        pdf_perf.close()
        pdf_data.close()
        with open(txt_perf_path, 'w') as f:
            f.write('\n'.join(problem_names))
        with open(txt_data_path, 'w') as f:
            f.write('\n'.join(problem_names))

    def solve_all(self, solvers, options, load, **kwargs):
        merit_values = []
        problem_names = []
        problem_dimensions = []
        histories = Parallel(n_jobs=-1)(self.solve(i, solvers, options, load, **kwargs) for i in range(len(self.problem_names)))
        for history in histories:
            if history is not None:
                merit_values.append(history[0])
                problem_names.append(history[1])
                problem_dimensions.append(history[2])
        return np.array(merit_values), problem_names, problem_dimensions

    @delayed
    def solve(self, i, solvers, options, load, **kwargs):
        logger = get_logger(__name__)

        # Load the PyCUTEst problem.
        logger.info(f'Loading {self.problem_names[i]} ({i + 1}/{len(self.problem_names)})')
        problem = self.load(self.problem_names[i])
        if problem is None:
            logger.warning(f'{self.problem_names[i]}: failed to load')
            return

        # Solve the problem with the given solvers.
        merit_values = np.full((len(solvers), self.feature_options['rerun'], self.max_eval), np.nan)
        for j, solver in enumerate(solvers):
            for k in range(self.feature_options['rerun']):
                # Attempt to load the history of the solver.
                fun_path, resid_path, success_path = self.get_storage_path(problem, solver, k)
                is_loaded = False
                n_eval = 0
                if load and fun_path.is_file() and resid_path.is_file() and success_path.is_file():
                    fun_values = np.load(fun_path)
                    resid_values = np.load(resid_path)
                    n_eval = fun_values.size
                    if n_eval <= self.max_eval:
                        success = np.load(success_path)
                        if success:
                            merit_values[j, k, :n_eval] = self.merit(fun_values, resid_values, **kwargs)
                            merit_values[j, k, n_eval:] = merit_values[j, k, n_eval - 1]
                            is_loaded = True
                    else:
                        n_eval = self.max_eval
                        merit_values[j, k, :] = self.merit(fun_values[:n_eval], resid_values[:n_eval], **kwargs)
                        is_loaded = True

                # Solve the problem if the history is not loaded.
                if not is_loaded:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        optimizer = Optimizer(problem, solver, self.max_eval, options[j], self.noise, k)
                        success, fun_values, resid_values = optimizer()
                    n_eval = min(fun_values.size, self.max_eval)
                    merit_values[j, k, :n_eval] = self.merit(fun_values[:n_eval], resid_values[:n_eval], **kwargs)
                    merit_values[j, k, n_eval:] = merit_values[j, k, n_eval - 1]
                    np.save(fun_path, fun_values[:n_eval])
                    np.save(resid_path, resid_values[:n_eval])
                    # np.save(success_path, np.array([success]))
                    np.save(success_path, np.array([True]))

                # Log the results.
                if not np.all(np.isnan(merit_values[j, k, :n_eval])):
                    if self.feature_options['rerun'] > 1:
                        header = f'{solver}({problem.name},{k})'
                    else:
                        header = f'{solver}({problem.name})'
                    i_min = np.nanargmin(merit_values[j, k, :n_eval])
                    logger.info(f'{header}: fun = {fun_values[i_min]}, resid = {resid_values[i_min]}, n_eval = {n_eval}')
                else:
                    logger.warning(f'{solver}({problem.name}): no value received')
        return merit_values, self.problem_names[i], problem.n

    def load(self, problem_name):
        logger = get_logger(__name__)
        try:
            # If the dimension of the problem is not fixed, we select the
            # largest possible dimension that matches the requirements. PyCUTEst
            # removes the variables that are fixed by the bound constraints. The
            # dimension of the reduced problem may not satisfy the requirements.
            if pycutest.problem_properties(problem_name)['n'] == 'variable':
                sif_n = [n for n in self.get_sif_n(problem_name) if self.n_min <= n <= self.n_max]
                if len(sif_n) > 0:
                    return pycutest.import_problem(problem_name, sifParams={'N': sif_n[-1]})
            else:
                return pycutest.import_problem(problem_name)
        except Exception as err:
            logger.warning(f'{problem_name}: {err}')

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
        elif self.feature == 'nan':
            rng = np.random.default_rng(int(1e8 * abs(np.sin(k) + np.sin(self.feature_options['rate']) + np.sum(np.sin(np.abs(np.sin(1e8 * x)))))))
            if rng.uniform() <= self.feature_options['rate']:
                f = np.nan
        elif self.feature == 'digits' and np.isfinite(f):
            if f == 0.0:
                fx_rounded = 0.0
            else:
                fx_rounded = round(f, self.feature_options['digits'] - int(np.floor(np.log10(np.abs(f)))) - 1)
            f = fx_rounded + (f - fx_rounded) * np.abs(np.sin(np.sin(np.sin(self.feature_options['digits'])) + np.sin(1e8 * f) + np.sum(np.sin(np.abs(1e8 * x))) + np.sin(x.size)))
        return f

    @staticmethod
    def get_sif_n(name):
        # Get all the available SIF parameters for all variables.
        command = [pycutest.get_sifdecoder_path(), '-show', name]
        process = subprocess.Popen(command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        sif_stdout = process.stdout.read()
        process.wait()

        # Extract all the available SIF parameters for the problem's dimension.
        regex = re.compile(r'^N=(?P<param>\d+)')
        sif_n = []
        for stdout in sif_stdout.split('\n'):
            sif_match = regex.match(stdout)
            if sif_match:
                sif_n.append(int(sif_match.group('param')))
        return sorted(sif_n)

    @staticmethod
    def merit(fun_values, resid_values, **kwargs):
        merit_values = np.empty_like(fun_values)
        for i in range(merit_values.size):
            if resid_values[i] <= kwargs.get('low_resid', 1e-12):
                merit_values[i] = fun_values[i]
            elif kwargs.get('barrier', False) and resid_values[i] >= kwargs.get('high_resid', 1e-6):
                merit_values[i] = np.inf
            else:
                merit_values[i] = fun_values[i] + kwargs.get('penalty', 1e8) * resid_values[i]
        return merit_values
