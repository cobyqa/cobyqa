import numpy as np
from optiprofiler import (
    find_cutest_problems,
    run_benchmark,
)


def cobyqa_pypi(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq):
    from cobyqa import minimize
    from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

    constraints = []
    if bub.size > 0:
        constraints.append(LinearConstraint(aub, -np.inf, bub))
    if beq.size > 0:
        constraints.append(LinearConstraint(aeq, beq, beq))
    cub_x0 = cub(x0)
    if cub_x0.size > 0:
        constraints.append(NonlinearConstraint(
            cub,
            -np.inf,
            np.zeros(cub_x0.size),
        ))
    ceq_x0 = ceq(x0)
    if ceq_x0.size > 0:
        constraints.append(NonlinearConstraint(
            ceq,
            np.zeros(ceq_x0.size),
            np.zeros(ceq_x0.size),
        ))
    minimize(fun, x0, bounds=Bounds(xl, xu), constraints=constraints)


def cobyqa_latest(fun, x0, xl, xu, aub, bub, aeq, beq, cub, ceq):
    from cobyqa_latest import minimize
    from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

    constraints = []
    if bub.size > 0:
        constraints.append(LinearConstraint(aub, -np.inf, bub))
    if beq.size > 0:
        constraints.append(LinearConstraint(aeq, beq, beq))
    cub_x0 = cub(x0)
    if cub_x0.size > 0:
        constraints.append(NonlinearConstraint(
            cub,
            -np.inf,
            np.zeros(cub_x0.size),
        ))
    ceq_x0 = ceq(x0)
    if ceq_x0.size > 0:
        constraints.append(NonlinearConstraint(
            ceq,
            np.zeros(ceq_x0.size),
            np.zeros(ceq_x0.size),
        ))
    minimize(fun, x0, bounds=Bounds(xl, xu), constraints=constraints)


if __name__ == "__main__":
    cutest_problem_names = find_cutest_problems(
        "nonlinear",
        n_max=10,
        m_linear_max=50,
        m_nonlinear_max=50,
    )
    run_benchmark(
        [cobyqa_latest, cobyqa_pypi],
        ["COBYQA Latest", "COBYQA PyPI"],
        cutest_problem_names,
        benchmark_id="nonlinear",
    )
