class Minimizer:
    def __init__(self, problem, solver, options, callback, *args, **kwargs):
        self.problem = problem
        self.solver = solver
        self.options = dict(options)
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
        self.obj_hist = []
        self.mcv_hist = []
        if not self.validate():
            raise NotImplementedError
