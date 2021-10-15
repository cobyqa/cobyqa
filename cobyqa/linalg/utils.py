import numpy as np
from numpy.testing import assert_


def givens(M, cval, sval, i, j, axis, slicing=None):
    r"""
    Perform a Givens rotation.

    Parameters
    ----------
    M : numpy.ndarray
        Matrix on which the Givens rotation is performed in-place.
    cval : float
        Multiple of the cosine value of the angle of rotation.
    sval : float
        Multiple of the sine value of the angle of rotation.
    i : int
        First index of the Givens rotation procedure.
    j : int
        Second index of the Givens rotation procedure.
    axis : int
        Axis over which to select values. If `M` is a matrix with two
        dimensions, the calculations will be applied to the rows by setting
        ``axis = 0`` and to the columns by setting ``axis = 1``.
    slicing : slice, optional
        Part of the data at which the Givens rotation should be applied.
        Default applies it to to all the components.

    Returns
    -------
    hval : float
        Length of the two-dimensional vector of components `cval` and
        `sval`, given by :math:`\sqrt{ \mathtt{cval}^2 + \mathtt{sval}^2 }`.
    """
    if slicing is None:
        slicing = slice(None)
    hval = np.hypot(cval, sval)
    cosv = cval / hval
    sinv = sval / hval

    # The function numpy.swapaxes returns only a view of M.
    Mc = np.swapaxes(M, 0, axis)
    Gr = np.array([[cosv, -sinv], [sinv, cosv]], dtype=float)
    Mc[[i, j], slicing] = np.matmul(Gr, Mc[[i, j], slicing])

    return hval


def qr(a, overwrite_a=False, pivoting=False, check_finite=True):
    """
    Compute the QR factorization ``a = Q @ R`` where ``Q`` is an orthogonal
    matrix and ``R`` is an upper triangular matrix.

    Parameters
    ----------
    a : array_like, shape (m, n)
        Matrix to be factorized.
    overwrite_a : bool, optional
        Whether to overwrite the data in `a` with the matrix ``R`` (may improve
        the performance by limiting the memory cost).
    pivoting : bool, optional
        Whether the factorization should include column pivoting, in which case
        a permutation vector ``P`` is returned such that ``A[:, P] = Q @ R``.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.

    Returns
    -------
    Q : numpy.ndarray, shape (m, m)
        Above-mentioned orthogonal matrix ``Q``.
    R : numpy.ndarray, shape (m, n)
        Above-mentioned upper triangular matrix ``R``.
    P : numpy.ndarray, shape (n,)
        Indices of the permutations. Not returned if ``pivoting=False``.

    Raises
    ------
    AssertionError
        The matrix `a` is not two-dimensional.
    """
    a = np.asarray_chkfinite(a) if check_finite else np.asarray(a)
    assert_(a.ndim == 2)

    m, n = a.shape
    tiny = np.finfo(float).tiny
    Q = np.eye(m, dtype=a.dtype)
    R = a if overwrite_a else np.copy(a)
    P = np.arange(n, dtype=int)
    for j in range(n):
        if pivoting:
            k = j + np.argmax(np.linalg.norm(R[j:, j:], axis=0))
            P[[j, k]] = P[[k, j]]
            R[:, [j, k]] = R[:, [k, j]]
        for i in range(j + 1, m):
            cval, sval = R[j, j], R[i, j]
            if abs(sval) > tiny * abs(cval):
                givens(Q, cval, sval, i, j, 1)
                givens(R, cval, sval, i, j, 0)
            R[i, j] = 0.
    if pivoting:
        return Q, R, P
    return Q, R


def getact(gq, evalc, argc, resid, iact, mleq, nact, qfac, rfac, delta):
    """
    Pick the current active set.

    The method seeks for the closest vector to ``-gq`` in Euclidean norm,
    subject to the linear constraints whose normalized residuals are upper
    bounded by ``0.2 * delta``. The method selects among these constraints a
    basis for the span their engender.

    Parameters
    ----------
    gq : numpy.ndarray, shape (n,)
        Vector from which the selected direction should be the closest.
    evalc : callable
        Constraint functions to be evaluated.

            ``evalc(i, x, *args) -> float``

        where ``i`` is an integer, ``x`` is an array with shape (n,) and `args`
        is a tuple of parameters to forward to the constraint function.
    argc : tuple
        Parameters to forward to the constraint function.
    resid : numpy.ndarray, shape (m,)
        Normalized residuals of each constraint, starting with the linear
        constraints, followed by the bound constraints.
    mleq : int
        Number of equality constraints.
    iact : numpy.ndarray, shape (n,)
        Indices of the actives constraints, in ``iact[:nact]``.
    nact : numpy.ndarray, shape ()
        Number of active constraints.
    qfac : numpy.ndarray, shape (n, n)
        Orthogonal matrix of the QR factorization of the matrix whose columns
        are the gradients of the active constraints, the first mleq constraints
        being the linear equality constraints.
    rfac : numpy.ndarray, shape (n, n)
        Upper triangular matrix of the QR factorization of the matrix whose
        columns are the gradients of the active constraints, the first mleq
        constraints being the linear equality constraints. Only the first
        ``mleq + nact`` columns of rfac are meaningful.
    delta : float or list of 2-tuple
        Description of the set in which the step will be projected before
        assessing its feasibility. If a float is provided, the step will be
        projected into the ball centered at the origin of radius `delta`.
        Otherwise, if `delta` is of the form ``[(i1, d1), (i2, d2), ...]``, it
        is understood that the first ``i1`` coordinates are bounded in Euclidean
        norm by ``d1``, the following ``i2`` coordinates are bounded in
        Euclidean norm by ``d2``, etc.

    Returns
    -------
    numpy.ndarray, shape (n,)
        The selected direction.

    Notes
    -----
    The selected direction is calculated using the Goldfarb and Idnani algorithm
    for quadratic programming [1]_.

    References
    ----------
    .. [1] D. Goldfarb and A. Idnani. "A numerically stable dual method for
       solving strictly convex quadratic programs." In: Math. Program. 27
       (1983), pp. 1--33.
    """
    eps = np.finfo(float).eps
    tiny = np.finfo(float).tiny
    n = gq.size
    tol = 10. * eps * n
    gqtol = tol * np.max(np.abs(gq), initial=1.)
    deltx = delta if isinstance(delta, (float, np.floating)) else delta[0][1]
    tdel = .2 * deltx

    # Remove from the current active set the constraints that are not considered
    # active anymore, that is those whose residuals exceed tdel.
    for k in range(nact - 1, -1, -1):
        if resid[iact[k]] > tdel:
            rmact(k, mleq, nact, qfac, rfac, iact)

    # The vector vlam stores the Lagrange multipliers of the calculations (only
    # the first nact components are meaningful). Compute these Lagrange
    # multipliers, and remove from the current active set the constraints whose
    # Lagrange multipliers are nonnegative.
    vlam = np.zeros_like(gq)
    k = nact - 1
    while k >= 0:
        kleq = mleq + k
        temp = np.inner(qfac[:, kleq], gq)
        temp -= np.inner(rfac[kleq, kleq + 1:mleq + nact], vlam[k + 1:nact])
        if temp >= 0:
            rmact(k, mleq, nact, qfac, rfac, iact, vlam)
            k = nact - 1
        else:
            vlam[k] = temp / rfac[kleq, kleq]
            k -= 1

    # Start the iterative procedure. The calculations must be stopped if
    # nact + mleq equals n, as n linearly independent constraints would have
    # been found, which would make of the origin the only feasible point.
    stepsq = np.inf
    while nact < n - mleq:
        # Set the new search direction. It is the vector that minimizes the
        # Euclidean norm of gq + step, subject to the active constraints. The
        # calculations are stopped if this vector is zero, of if its norm
        # exceeds the norm of the previous direction.
        temp = np.dot(qfac[:, mleq + nact:].T, gq)
        step = -np.dot(qfac[:, mleq + nact:], temp)
        ssq = np.inner(step, step)
        if ssq >= stepsq or np.sqrt(ssq) <= gqtol:
            return np.zeros_like(step)
        else:
            stepsq = ssq

        # Select the index of the most violated constraint, if any. The step
        # that is considered in these calculations is the one of length delta
        # along the direction in the vector step.
        if isinstance(delta, (float, np.floating)):
            test = np.sqrt(ssq) / delta
        else:
            isav = 0
            test = 0.
            for i, radius in delta:
                test = max(test, np.linalg.norm(step[isav:i]) / radius)
                isav = i
        inext = -1
        violmx = 0.
        for i in range(resid.size):
            if i not in iact[:nact] and resid[i] <= tdel:
                lhs = evalc(i, step, *argc)
                if lhs > max(test * resid[i], violmx):
                    inext = i
                    violmx = lhs

        # Return if no constraint is violated, or if the constraint violation
        # previously calculated is too small, as a positive value of violmx
        # might then be caused by computer rounding errors.
        ctol = 0.
        if 0. < violmx < 1e-2 * deltx:
            for k in range(nact):
                ctol = max(ctol, abs(evalc(iact[k], step, *argc)))
        ctol *= 10.
        if inext == -1 or violmx <= ctol:
            return step

        # Add the inext-th constraint to the active set, by applying Givens
        # rotations to the matrix qfac, and add the appropriate column to rfac.
        sval = 0.
        for k in range(n - 1, -1, -1):
            cval = evalc(inext, qfac[:, k], *argc)
            if k < mleq + nact:
                rfac[k, mleq + nact] = cval
            elif abs(sval) <= tol * abs(cval):
                sval = cval
            else:
                sval = givens(qfac, cval, sval, k + 1, k, 1)
        if sval < 0.:
            qfac[:, mleq + nact] = -qfac[:, mleq + nact]
        rfac[mleq + nact, mleq + nact] = abs(sval)
        iact[nact] = inext
        vlam[nact] = 0.
        nact += 1

        while violmx > ctol:
            # Update the vector of the Lagrange multipliers of the calculations
            # to include the new constraint. When a constraint is added or
            # removed, all the Lagrange multipliers must be updated.
            vmu = np.empty(nact)
            vmu[-1] = 1. / rfac[mleq + nact - 1, mleq + nact - 1] ** 2.
            for k in range(nact - 2, -1, -1):
                kleq = mleq + k
                temp = -np.inner(rfac[kleq, kleq + 1:mleq + nact], vmu[k + 1:])
                vmu[k] = temp / rfac[kleq, kleq]
            vmult = violmx
            imult = np.abs(vmu) > tiny * np.abs(vlam[:nact])
            imult = imult & (vlam[:nact] >= vmult * vmu)
            imult[-1] = False
            k = -1
            if np.any(imult):
                mult = np.copy(vlam[:nact])
                mult[imult] = mult[imult] / vmu[imult]
                mult[np.logical_not(imult)] = np.inf
                k = np.argmin(mult)
                vmult = mult[k]
            vlam[:nact] -= vmult * vmu
            if k >= 0:
                vlam[k] = 0.
                violmx = max(violmx - vmult, 0.)
            else:
                violmx = 0.

            # Remove from the active set the constraints whose Lagrange
            # multipliers are nonnegative. This mechanism ensures the active
            # constraints are linearly independent.
            for k in range(nact - 1, -1, -1):
                if vlam[k] >= 0.:
                    rmact(k, mleq, nact, qfac, rfac, iact, vlam)

    return np.zeros_like(gq)


def rmact(k, mleq, nact, qfac, rfac, *args):
    """
    Remove a constraint from the active set.

    A constraint is removed from the active set by applying a sequence of Givens
    rotations to the factorization matrices of the matrix whose columns are the
    gradients of the active constraints.

    Parameters
    ----------
    k : int
        Index of the constraint to be removed.
    mleq : int
        Number of equality constraints
    nact : numpy.ndarray, shape ()
        Number of active constraints before the removal. It should be provided
        as a zero-dimensional array because the method will decrement it.
    qfac : numpy.ndarray, shape (n, n)
        Orthogonal matrix of the QR factorization of the matrix whose columns
        are the gradients of the active constraints.
    rfac : numpy.ndarray, shape (n, n)
        Upper triangular matrix of the QR factorization of the matrix whose
        columns are the gradients of the active constraints. Only the first
        `nact` columns of rfac are meaningful.
    *args
        List of ``numpy.ndarray, shape (n,)`` that should be modified when a
        constraint is removed from the active set. They could be for example the
        indices of the active constraints, or the Lagrange multipliers.
    """
    for j in range(mleq + k, mleq + nact - 1):
        # Perform a Givens rotation on the matrix rfac that exchange the order
        # of the j-th and the (j + 1)-th constraints. The calculations are done
        # only on the first mleq + nact columns of rfac since the remaining
        # columns are meaningless, to increase the computational efficiency.
        cval, sval = rfac[j + 1, j + 1], rfac[j, j + 1]
        hval = givens(rfac, cval, sval, j, j + 1, 0, slice(j, mleq + nact))
        rfac[[j, j + 1], j:mleq + nact] = rfac[[j + 1, j], j:mleq + nact]
        rfac[:j + 2, [j, j + 1]] = rfac[:j + 2, [j + 1, j]]
        rfac[j, j] = hval
        rfac[j + 1, j] = 0.

        # Perform the corresponding Givens rotations on the matrix qfac.
        givens(qfac, cval, sval, j, j + 1, 1)
        qfac[:, [j, j + 1]] = qfac[:, [j + 1, j]]

    # Rearrange the array's order and decrement nact.
    for array in args:
        array[k:nact - 1] = array[k + 1:nact]
    nact -= 1
