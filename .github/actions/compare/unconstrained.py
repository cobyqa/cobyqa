from optiprofiler import set_cutest_problem_options, find_cutest_problems, create_profiles


def cobyqa_pypi(fun, x0):
    from cobyqa import minimize

    minimize(fun, x0)


def cobyqa_latest(fun, x0):
    from cobyqa_latest import minimize

    minimize(fun, x0)


if __name__ == '__main__':
    set_cutest_problem_options(n_max=10)
    cutest_problem_names = find_cutest_problems('unconstrained')
    create_profiles([cobyqa_latest, cobyqa_pypi], ['COBYQA Latest', 'COBYQA PyPI'], cutest_problem_names, benchmark_id='unconstrained')
