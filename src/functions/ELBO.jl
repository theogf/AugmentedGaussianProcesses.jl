function ELBO(
    model::GP{T},
    pr_mean,
    kernel,
) where {T<:Real}
    # setprior!(model, pr_means, kernels, Zs)
    setpr_mean!(model.f, pr_mean)
    setkernel!(model.f, kernel)
    computeMatrices!(model, true)
    return log_py(model)
end



@traitfn function ELBO(
    model::TGP,
    pr_means,
    kernels,
) where {T<:Real, TGP<:AbstractGP{T}; IsFull{TGP}}
    # setprior!(model, pr_means, kernels, Zs)
    setpr_means!(model, pr_means)
    setkernels!(model, kernels)
    computeMatrices!(model, true)
    return ELBO(model)
end

@traitfn function ELBO(
    model::TGP,
    pr_means,
    kernels,
    Zs,
) where {T<:Real, TGP<:AbstractGP{T}; !IsFull{TGP}}
    # setprior!(model, pr_means, kernels, Zs)
    setpr_means!(model, pr_means)
    setkernels!(model, kernels)
    setZs!(model, Zs)
    computeMatrices!(model, true)
    return ELBO(model)
end