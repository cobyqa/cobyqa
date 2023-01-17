class OptimizeResult(dict):
    """
    Result of the optimization algorithm.

    Attributes
    ----------
    x : numpy.ndarray, shape (n,)
        Solution point.
    success : bool
        Flag indicating whether the optimizer terminated successfully.
    status : int
        Termination status.
    message : str
        Description of the termination status.
    fun : float
        Value of the objective function.
    jac : numpy.ndarray, shape (n,)
        Approximation of the gradient of the objective function based on
        undetermined interpolation. If the value of a component (or more) of the
        gradient is unknown, it is replaced with `numpy.nan`.
    nfev : int
        Number of function evaluations.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        Maximum constraint violation. It is set only if the problem is not
        declared unconstrained by the optimizer.
    """

    def __dir__(self):
        return list(self.keys())

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, key, value):
        super().__setitem__(key, value)

    def __delattr__(self, key):
        super().__delitem__(key)

    def __repr__(self):
        attrs = ", ".join(f"{k}={repr(v)}" for k, v in sorted(self.items()))
        return f"{self.__class__.__name__}({attrs})"

    def __str__(self):
        if self.keys():
            m = max(map(len, self.keys())) + 1
            return "\n".join(f"{k:>{m}}: {v}" for k, v in sorted(self.items()))
        return f"{self.__class__.__name__}()"
