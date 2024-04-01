from optiprofiler import (
    find_cutest_problems,
    run_benchmark,
)


def cobyqa_pypi(fun, x0):
    from cobyqa import minimize

    minimize(fun, x0)


def cobyqa_latest(fun, x0):
    from cobyqa_latest import minimize

    minimize(fun, x0)


if __name__ == "__main__":
    cutest_problem_names = find_cutest_problems("unconstrained", n_max=10)
    run_benchmark(
        [cobyqa_latest, cobyqa_pypi],
        ["COBYQA Latest", "COBYQA PyPI"],
        cutest_problem_names,
        benchmark_id="unconstrained",
    )
