def gridLineSearch(theta, rhoFn, d, lambdaStepSize=0.01, lambdaMax=1):
    lambdas = range(0, lambdaMax, lambdaStepSize)
    rhoVals = lambdas.map(lambda l: rhoFn([t - l * di for t, di in zip(theta, d)]))

    return rhoVals.index(min(rhoVals))


def testConvergence(thetaNew, thetaOld, tolerance=1e-6, relative=False):
    abs_diffs = sum([abs(thetaNew[i] - thetaOld[i] for i in range(len(thetaNew)))])
    rel_diffs = abs_diffs / (sum(abs(thetaOld[i]) for i in range(len(thetaOld))))

    if relative:
        return rel_diffs < tolerance
    else:
        return abs_diffs < tolerance


def gradient_descent(
    theta,
    rhoFn,
    gradFn,
    lineSearchFn,
    testConvergenceFn,
    maxIter=1000,
    toler=1e-6,
    lambdaStepSize=0.1,
    lambdaMax=0.5,
    relative=False,
):
    converged = False
    iter_count = 0

    while not converged and iter_count < maxIter:
        grad = gradFn(theta)
        grad_length = sum(g**2 for g in grad) ** 0.5

        if grad_length > 0:
            d = [g / grad_length for g in grad]
        else:
            d = grad

        step = lineSearchFn(
            theta=theta,
            rhoFn=rhoFn,
            d=d,
            lambdaStepSize=lambdaStepSize,
            lambdaMax=lambdaMax,
        )

        try:
            thetaNew = [t - step * di for t, di in zip(theta, d)]
        except TypeError:
            thetaNew = theta - step * d

        converged = testConvergenceFn(
            thetaNew, theta, tolerance=toler, relative=relative
        )

        theta = thetaNew
        iter_count += 1
