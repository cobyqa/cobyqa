import numpy as np
from optiprofiler import (
    set_cutest_problem_options,
    find_cutest_problems,
    run_benchmark,
)


def cobyqa_pypi(fun, x0, xl, xu, aub, bub, aeq, beq):
    from cobyqa import minimize
    from scipy.optimize import Bounds, LinearConstraint

    constraints = []
    if bub.size > 0:
        constraints.append(LinearConstraint(aub, -np.inf, bub))
    if beq.size > 0:
        constraints.append(LinearConstraint(aeq, beq, beq))
    minimize(fun, x0, bounds=Bounds(xl, xu), constraints=constraints)


def cobyqa_latest(fun, x0, xl, xu, aub, bub, aeq, beq):
    from cobyqa_latest import minimize
    from scipy.optimize import Bounds, LinearConstraint

    constraints = []
    if bub.size > 0:
        constraints.append(LinearConstraint(aub, -np.inf, bub))
    if beq.size > 0:
        constraints.append(LinearConstraint(aeq, beq, beq))
    minimize(fun, x0, bounds=Bounds(xl, xu), constraints=constraints)


if __name__ == "__main__":
    set_cutest_problem_options(n_max=10)
    cutest_problem_names = find_cutest_problems("adjacency linear")
    run_benchmark(
        [cobyqa_latest, cobyqa_pypi],
        ["COBYQA Latest", "COBYQA PyPI"],
        cutest_problem_names,
        benchmark_id="linear",
    )
