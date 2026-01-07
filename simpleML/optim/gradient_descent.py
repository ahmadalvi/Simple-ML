def gridLineSearch(theta, rhoFn, d, lambdaStepSize=0.01, lambdaMax=1):
    lambdas = [0.0] + [
        k * lambdaStepSize for k in range(1, int(lambdaMax / lambdaStepSize) + 1)
    ]

    values = [rhoFn([t + l * di for t, di in zip(theta, d)]) for l in lambdas]

    best_idx = values.index(min(values))
    return lambdas[best_idx]


def testConvergence(thetaNew, thetaOld, tolerance=1e-10, relative=False):
    diffs = [abs(a - b) for a, b in zip(thetaNew, thetaOld)]
    abs_norm = sum(d**2 for d in diffs) ** 0.5

    if not relative:
        return abs_norm < tolerance

    denom = sum(abs(b) for b in thetaOld)
    if denom == 0:
        return abs_norm < tolerance

    rel_norm = abs_norm / denom
    return rel_norm < tolerance


def gradient_descent(
    theta,
    rhoFn,
    gradFn,
    lineSearchFn=gridLineSearch,
    testConvergenceFn=testConvergence,
    maxIter=1000,
    toler=1e-6,
    lambdaStepSize=0.01,
    lambdaMax=0.5,
    relative=False,
):
    converged = False
    iter_count = 0

    while not converged and iter_count < maxIter:
        grad = gradFn(theta)

        grad_norm = sum(g**2 for g in grad) ** 0.5
        if grad_norm > 0:
            d = [-g / grad_norm for g in grad]
        else:
            d = [g for g in grad]

        if grad_norm < toler:
            converged = True
            break

        step = lineSearchFn(
            theta=theta,
            rhoFn=rhoFn,
            d=d,
            lambdaStepSize=lambdaStepSize,
            lambdaMax=lambdaMax,
        )

        thetaNew = [t + step * di for t, di in zip(theta, d)]

        converged = testConvergenceFn(
            thetaNew, theta, tolerance=toler, relative=relative
        )

        theta = thetaNew
        iter_count += 1

    return [theta, converged, iter_count, rhoFn(theta)]
