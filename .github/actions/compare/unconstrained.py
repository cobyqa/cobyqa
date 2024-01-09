from OptiProfiler import find_cutest, create_profiles


def cobyqa_pypi(fun, x0, max_eval):
    from cobyqa import minimize
    minimize(fun, x0, options={'max_eval': max_eval})


def cobyqa_latest(fun, x0, max_eval):
    from cobyqa_latest import minimize
    minimize(fun, x0, options={'max_eval': max_eval})


if __name__ == '__main__':
    problem_names = find_cutest('unconstrained', n_max=10)
    create_profiles([cobyqa_latest, cobyqa_pypi], ['COBYQA Latest', 'COBYQA PyPI'], problem_names, 'plain', n_max=10)
