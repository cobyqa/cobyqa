class MaxEvalError(Exception):
    """
    Exception raised when the maximum number of evaluations is reached.
    """
    pass


class TargetSuccess(Exception):
    """
    Exception raised when the target value is reached.
    """
    pass


class FeasibleSuccess(Exception):
    """
    Exception raised when a feasible point of a feasible problem is found.
    """
    pass
