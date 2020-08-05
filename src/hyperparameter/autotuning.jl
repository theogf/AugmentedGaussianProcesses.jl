include("autotuning_utils.jl")
include("zygote_rules.jl")
include("forwarddiff_rules.jl")

function update_hyperparameters!(m::GP)
    update_hyperparameters!(getf(m), xview(m))
end

@traitfn function update_hyperparameters!(
    m::TGP,
) where {TGP <: AbstractGP; IsFull{TGP}}
    update_hyperparameters!.(m.f, [xview(m)])
    setHPupdated!(m.inference, true)
end

@traitfn function update_hyperparameters!(
    m::TGP,
) where {TGP <: AbstractGP; !IsFull{TGP}}
    update_hyperparameters!.(
        m.f,
        Ref(xview(m)),
        ∇E_μ(m),
        ∇E_Σ(m),
        inference(m),
        inference(m).vi_opt,
    )
    setHPupdated!(m.inference, true)
end

## Update all hyperparameters for the full batch GP models ##
function update_hyperparameters!(
    gp::AbstractLatent,
    X::AbstractVector,
)
    if !isnothing(gp.opt)
        f_l, f_μ₀ = hyperparameter_gradient_function(gp, X)
        aduse = K_ADBACKEND[] == :auto ? ADBACKEND[] : K_ADBACKEND[]
        grads = if aduse == :forward_diff
            ∇L_ρ_forward(f_l, gp, X)
        elseif aduse == :reverse_diff
            ∇L_ρ_reverse(f_l, gp, X)
        end
        grads[pr_mean(gp)] = f_μ₀()
        apply_grads_kernel_params!(gp.opt, kernel(gp), grads) # Apply gradients to the kernel parameters
        apply_gradients_mean_prior!(pr_mean(gp), grads[pr_mean(gp)], X)
    end
end

## Update all hyperparameters for the sparse variational GP models ##
function update_hyperparameters!(
    gp::AbstractLatent,
    X::AbstractVector,
    ∇E_μ::AbstractVector,
    ∇E_Σ::AbstractVector,
    i::Inference,
    vi_opt::InferenceOptimizer,
)
    if !isnothing(gp.opt)
        f_ρ, f_Z, f_μ₀ = hyperparameter_gradient_function(gp)
        k_aduse = K_ADBACKEND[] == :auto ? ADBACKEND[] : K_ADBACKEND[]
        grads = if k_aduse == :forward_diff
            ∇L_ρ_forward(f_ρ, gp, X, ∇E_μ, ∇E_Σ, i, vi_opt)
        elseif k_aduse == :reverse_diff
            ∇L_ρ_reverse(f_ρ, gp, X, ∇E_μ, ∇E_Σ, i, vi_opt)
        end
        # @show grads[kernel(gp).transform.s]
        grads[pr_mean(gp)] = f_μ₀()
    end
    if !isnothing(opt(gp.Z)) && !isnothing(gp.opt)
        Z_aduse = Z_ADBACKEND[] == :auto ? ADBACKEND[] : Z_ADBACKEND[]
        Z_grads = if Z_aduse == :forward_diff
            Z_gradient_forward(gp, f_Z, X, ∇E_μ, ∇E_Σ, i, vi_opt) #Compute the gradient given the inducing points location
        elseif Z_aduse == :reverse_diff
            Z_gradient_reverse(gp, f_Z, X, ∇E_μ, ∇E_Σ, i, vi_opt)
        end
        update!(opt(gp.Z), gp.Z.Z, Z_grads) #Apply the gradients on the location
    end
    if !isnothing(gp.opt)
        apply_grads_kernel_params!(gp.opt, kernel(gp), grads) # Apply gradients to the kernel parameters
        apply_gradients_mean_prior!(pr_mean(gp), grads[pr_mean(gp)], X)
    end
end


## Return the derivative of the KL divergence between the posterior and the GP prior ##
function hyperparameter_KL_gradient(J::AbstractMatrix, A::AbstractMatrix)
    return 0.5 * opt_trace(J, A)
end


function hyperparameter_gradient_function(
    gp::LatentGP{T},
    X::AbstractVector,
) where {T}
    A = (inv(cov(gp)) - mean(gp) * transpose(mean(gp))) # μ = inv(K+σ²)*(y-μ₀)
    return (function (Jnn)
        return -hyperparameter_KL_gradient(Jnn, A)
    end, function ()
        return -mean(gp)
    end)
end

## Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model ##
function hyperparameter_gradient_function(
    gp::VarLatent{T},
    X::AbstractVector,
) where {T<:Real}
    μ₀ = pr_mean(gp, X)
    A = (I - pr_cov(gp) \ (cov(gp) + (mean(gp) - μ₀) * transpose(mean(gp) - μ₀))) / pr_cov(gp)
    return (
        function (Jnn)
            return -hyperparameter_KL_gradient(Jnn, A)
        end,
        function ()
            return pr_cov(gp) \ (mean(gp) - μ₀)
        end,
    )
end

## Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse latent GP ##
function hyperparameter_gradient_function(gp::SparseVarLatent{T}) where {T<:Real}
    μ₀ = pr_mean(gp, gp.Z)
    A = (I(dim(gp)) - pr_cov(gp) \ (cov(gp) + (mean(gp) - μ₀) * transpose(mean(gp) - μ₀))) / cov(gp)
    κΣ = gp.κ * cov(gp)
    return (
        function (Jmm, Jnm, Jnn, ∇E_μ, ∇E_Σ, i, opt)
            return (
                hyperparameter_expec_gradient(
                    gp,
                    ∇E_μ,
                    ∇E_Σ,
                    i,
                    opt,
                    κΣ,
                    Jmm,
                    Jnm,
                    Jnn,
                ) - hyperparameter_KL_gradient(Jmm, A)
            )
        end,
        function (Jmm, Jnm, ∇E_μ, ∇E_Σ, i, opt)
            hyperparameter_expec_gradient(
                gp,
                ∇E_μ,
                ∇E_Σ,
                i,
                opt,
                κΣ,
                Jmm,
                Jnm,
                zero(gp.K̃),
            ) - hyperparameter_KL_gradient(Jmm, A)
        end,
        function ()
            return pr_cov(gp) \ (mean(gp) - μ₀)
        end,
    )
end

function hyperparameter_gradient_function(
    gp::TVarLatent{T},
    X::AbstractVector,
) where {T<:Real}
    μ₀ = pr_mean(gp, X)
    A =
        (
            I(dim(gp)) .-
            (pr_cov(gp)) \ (cov(gp) .+ (mean(gp) - μ₀) * transpose(mean(gp) - μ₀))
        ) / (pr_cov(gp))
    return (
        function (Jnn)
            return -hyperparameter_KL_gradient(prior(gp).χ * Jnn, A)
        end,
        function ()
            return (pr_cov(gp)) \ (mean(gp) - μ₀)
        end,
    )
end


function hyperparameter_gradient_function(gp::OnlineVarLatent{T}) where {T<:Real}
    μ₀ = pr_mean(gp, gp.Z)
    A =
        (
            I(dim(gp)) -
            pr_cov(gp) \ (cov(gp) + (mean(gp) - μ₀) * transpose(mean(gp) - μ₀))
        ) / pr_cov(gp)
    κΣ = gp.κ * cov(gp)
    κₐΣ = gp.κₐ * cov(gp)
    return (
        function (Jmm, Jnm, Jnn, Jab, Jaa, ∇E_μ, ∇E_Σ, i, opt)
            ∇E = hyperparameter_expec_gradient(
                gp,
                ∇E_μ,
                ∇E_Σ,
                i,
                opt,
                κΣ,
                Jmm,
                Jnm,
                Jnn,
            )
            ∇KLₐ = hyperparameter_online_gradient(gp, κₐΣ, Jmm, Jab, Jaa)
            ∇KL = hyperparameter_KL_gradient(Jmm, A)
            return ∇E + ∇KLₐ - ∇KL
        end, # Function gradient given kernel parameters
        function (Jmm, Jnm, Jab, ∇E_μ, ∇E_Σ, i, opt)
            hyperparameter_expec_gradient(
                gp,
                ∇E_μ,
                ∇E_Σ,
                i,
                opt,
                κΣ,
                Jmm,
                Jnm,
                zero(gp.K̃),
            ) + hyperparameter_online_gradient(
                gp,
                κₐΣ,
                Jmm,
                Jab,
                zeros(T, length(gp.Zₐ), length(gp.Zₐ)),
            ) - hyperparameter_KL_gradient(Jmm, A)
        end, # Function gradient given inducing points locations
        function ()
            return -(pr_cov(gp) \ (μ₀ - mean(gp)))
        end,
    ) # Function gradient given mean prior
end

## Gradient with respect to hyperparameter for analytical VI ##
function hyperparameter_expec_gradient(
    gp::AbstractLatent,
    ∇E_μ::AbstractVector{<:Real},
    ∇E_Σ::AbstractVector{<:Real},
    i::AnalyticVI,
    opt::AVIOptimizer,
    κΣ::AbstractMatrix{<:Real},
    Jmm::AbstractMatrix{<:Real},
    Jnm::AbstractMatrix{<:Real},
    Jnn::AbstractVector{<:Real},
)
    ι = (Jnm - gp.κ * Jmm) / pr_cov(gp)
    J̃ = Jnn - (opt_diag(ι, gp.Knm) + opt_diag(gp.κ, Jnm))
    dμ = dot(∇E_μ, ι * mean(gp))
    dΣ = -dot(∇E_Σ, J̃)
    dΣ += -dot(∇E_Σ, 2.0 * (opt_diag(ι, κΣ)))
    dΣ += -dot(∇E_Σ, 2.0 * (ι * mean(gp)) .* (gp.κ * mean(gp)))
    return getρ(i) * (dμ + dΣ)
end


## Gradient with respect to hyperparameters for numerical VI ##
function hyperparameter_expec_gradient(
    gp::AbstractLatent,
    ∇E_μ::AbstractVector{<:Real},
    ∇E_Σ::AbstractVector{<:Real},
    i::NumericalVI,
    opt::NVIOptimizer,
    κΣ::AbstractMatrix{<:Real},
    Jmm::AbstractMatrix{<:Real},
    Jnm::AbstractMatrix{<:Real},
    Jnn::AbstractVector{<:Real},
)
    ι = (Jnm - gp.κ * Jmm) / pr_cov(gp)
    J̃ = Jnn - (opt_diag(ι, gp.Knm) + opt_diag(gp.κ, Jnm))
    dμ = dot(∇E_μ, ι * mean(gp))
    dΣ = dot(∇E_Σ, J̃ + 2.0 * opt_diag(ι, κΣ))
    return getρ(i) * (dμ + dΣ)
end

function hyperparameter_online_gradient(
    gp::AbstractLatent,
    κₐΣ::Matrix{<:Real},
    Jmm::AbstractMatrix{<:Real},
    Jab::AbstractMatrix{<:Real},
    Jaa::AbstractMatrix{<:Real},
)
    ιₐ = (Jab - gp.κₐ * Jmm) / pr_cov(gp)
    trace_term =
        -0.5 * sum(opt_trace.(
            [gp.invDₐ],
            [
                Jaa,
                2 * ιₐ * transpose(κₐΣ),
                -ιₐ * transpose(gp.Kab),
                -gp.κₐ * transpose(Jab),
            ],
        ))
    term_1 = dot(gp.prevη₁, ιₐ * mean(gp))
    term_2 = -dot(ιₐ * mean(gp), gp.invDₐ * gp.κₐ * mean(gp))
    return trace_term + term_1 + term_2
end
