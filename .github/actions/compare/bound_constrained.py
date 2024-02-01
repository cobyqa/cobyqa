from optiprofiler import (
    set_cutest_problem_options,
    find_cutest_problems,
    run_benchmark,
)


def cobyqa_pypi(fun, x0, xl, xu):
    from cobyqa import minimize
    from scipy.optimize import Bounds

    minimize(fun, x0, bounds=Bounds(xl, xu))


def cobyqa_latest(fun, x0, xl, xu):
    from cobyqa_latest import minimize
    from scipy.optimize import Bounds

    minimize(fun, x0, bounds=Bounds(xl, xu))


if __name__ == "__main__":
    set_cutest_problem_options(n_max=10)
    cutest_problem_names = find_cutest_problems("bound")
    run_benchmark(
        [cobyqa_latest, cobyqa_pypi],
        ["COBYQA Latest", "COBYQA PyPI"],
        cutest_problem_names,
        benchmark_id="bound",
    )
