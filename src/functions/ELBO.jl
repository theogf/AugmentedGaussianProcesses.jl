function ELBO(model::GP, X, y, pr_mean, kernel)
    setpr_mean!(model.f, pr_mean)
    setkernel!(model.f, kernel)
    state = compute_kernel_matrices(model, (;), X, true)
    return objective(model, state, y)
end

function ELBO(model::AbstractGPModel, X, y, pr_means, kernels, state)
    setpr_means!(model, pr_means)
    setkernels!(model, kernels)
    state = compute_kernel_matrices(model, state, X, true)
    return objective(model, state, y)
end

function ELBO(model::AbstractGPModel, X, y, pr_means, kernels, Zs, state)
    setpr_means!(model, pr_means)
    setkernels!(model, kernels)
    setZs!(model, Zs)
    state = compute_kernel_matrices(model, state, X, true)
    return objective(model, state, y)
end
