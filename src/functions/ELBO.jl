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

# External ELBO call on internal and new data
@traitfn function ELBO(model::TGP) where {TGP <: AbstractGPModel; IsFull{TGP}}
    return ELBO(model, input(model.data), output(model.data))
end

function ELBO(model::AbstractGPModel, X::AbstractMatrix, y::AbstractArray; obsdim=1)
    return ELBO(model, KernelFunctions.vec_of_vecs(X; obsdim), y)
end

function ELBO(model::AbstractGPModel, X::AbstractVector, y::AbstractArray)
    y = treat_labels!(y, likelihood(model))
    state = compute_kernel_matrices(model, (;), X, true)
    if inference(model) isa AnalyticVI
        local_vars = init_local_vars(likelihood(model), length(X))
        local_vars = local_updates!(
            local_vars,
            likelihood(model),
            y,
            mean_f(model, state.kernel_matrices),
            var_f(model, state.kernel_matrices),
        )
        state = merge(state, (; local_vars))
    end
    return ELBO(model, state, y)
end

function ELBO(
    model::OnlineSVGP, state::NamedTuple, X::AbstractMatrix, y::AbstractArray; obsdim=1
)
    return ELBO(model, state, KernelFunctions.vec_of_vecs(X; obsdim), y)
end

function ELBO(model::OnlineSVGP, state::NamedTuple, X::AbstractVector, y::AbstractArray)
    y = treat_labels!(y, likelihood(model))
    state = compute_kernel_matrices(model, state, X, true)
    if inference(model) isa AnalyticVI
        local_vars = init_local_vars(likelihood(model), length(X))
        local_vars = local_updates!(
            local_vars,
            likelihood(model),
            y,
            mean_f(model, state.kernel_matrices),
            var_f(model, state.kernel_matrices),
        )
        state = merge(state, (; local_vars))
    end
    return ELBO(model, state, y)
end
