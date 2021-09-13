function ELBO(model::GP, pr_mean, kernel)
    setpr_mean!(model.f, pr_mean)
    setkernel!(model.f, kernel)
    compute_kernel_matrices!(model, true)
    return log_py(model)
end

@traitfn function ELBO(
    model::TGP, pr_means, kernels
) where {TGP <: AbstractGPModel; IsFull{TGP}}
    setpr_means!(model, pr_means)
    setkernels!(model, kernels)
    compute_kernel_matrices!(model, true)
    return ELBO(model)
end

@traitfn function ELBO(
    model::TGP, pr_means, kernels, Zs
) where {TGP <: AbstractGPModel; !IsFull{TGP}}
    setpr_means!(model, pr_means)
    setkernels!(model, kernels)
    setZs!(model, Zs)
    compute_kernel_matrices!(model, true)
    return ELBO(model)
end
