import numpy as np
from OptiProfiler import find_cutest, create_profiles


def cobyqa_pypi(fun, x0, xl, xu, aub, bub, aeq, beq, max_eval):
    from cobyqa import minimize
    minimize(fun, x0, xl=xl, xu=xu, aub=aub, bub=bub, aeq=aeq, beq=beq, options={'max_eval': max_eval})


def cobyqa_latest(fun, x0, xl, xu, aub, bub, aeq, beq, max_eval):
    from cobyqa_latest import minimize
    from scipy.optimize import Bounds, LinearConstraint
    constraints = []
    if bub.size > 0:
        constraints.append(LinearConstraint(aub, -np.inf, bub))
    if beq.size > 0:
        constraints.append(LinearConstraint(aeq, beq, beq))
    minimize(fun, x0, bounds=Bounds(xl, xu), constraints=constraints, options={'max_eval': max_eval})


if __name__ == '__main__':
    problem_names = find_cutest('adjacency linear', n_max=10)
    create_profiles([cobyqa_latest, cobyqa_pypi], ['COBYQA Latest', 'COBYQA PyPI'], problem_names, 'plain', n_max=10)
