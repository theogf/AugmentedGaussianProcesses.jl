include("autotuning_utils.jl")
include("zygote_rules.jl")
include("forwarddiff_rules.jl")

function update_hyperparameters!(m::GP, state, x, y)
    μ₀ = pr_mean(m.f) # Get prior means
    k = kernel(m.f) # Get kernels
    if !isnothing(opt(m.f))
        if ADBACKEND[] == :Zygote
            hp_state = state.hyperopt_state
            # Compute gradients
            Δμ₀, Δk = Zygote.gradient(μ₀, k) do μ₀, k # Compute gradients for the whole model
                ELBO(m, x, y, μ₀, k)
            end

            # Optimize prior mean
            hp_state = state.hyperopt_state
            if !isnothing(Δμ₀)
                hp_state = update!(μ₀, Δμ₀, hp_state)
            end

            # Optimize kernel parameters
            if isnothing(Δk)
                @warn "Kernel gradients are equal to zero" maxlog = 1
            else
                state_k = update_kernel!(opt(m.f), kernel(m.f), Δk, hp_state)
                hp_state = merge(hp_state, (; state_k))
            end
            state = merge(state, (; hyperopt_state=hp_state))
        elseif ADBACKEND[] == :ForwardDiff
            θ, re = destructure((μ₀, k))
            Δ = ForwardDiff.gradient(θ) do θ
                ELBO(m, re(θ)...)
            end
        end
    end
    # end
    return state
end

# @traitfn function update_hyperparameters!(
#     m::TGP,
# ) where {TGP <: AbstractGPModel; IsFull{TGP}}
#     update_hyperparameters!.(m.f, [xview(m)])
#     setHPupdated!(m.inference, true)
# end

@traitfn function update_hyperparameters!(
    m::TGP, state, x, y
) where {TGP <: AbstractGPModel; IsFull{TGP}}
    if any((!) ∘ isnothing ∘ opt, m.f) # Check there is a least one optimiser
        hp_state = state.hyperopt_state
        μ₀ = pr_means(m) # Get prior means
        ks = kernels(m) # Get kernels
        if ADBACKEND[] == :Zygote
            Δμ₀, Δk = Zygote.gradient(μ₀, ks) do μ₀, ks # Compute gradients for the whole model
                ELBO(m, x, y, μ₀, ks, state)
            end
            # Optimize prior mean
            if !isnothing(Δμ₀)
                hp_state = update!.(μ₀, Δμ₀, hp_state)
            end
            if isnothing(Δk)
                @warn "Kernel gradients are equal to zero" maxlog = 1
            else
                hp_state = map(m.f, Δk, hp_state) do gp, Δ, hp_st
                    if isnothing(opt(gp))
                        return hp_st
                    else
                        state_k = update_kernel!(opt(gp), kernel(gp), Δ, hp_st.state_k)
                        return merge(hp_st, (; state_k))
                    end
                end
            end
        elseif ADBACKEND[] == :ForwardDiff
            θ, re = destructure((μ₀, ks))
            Δ = ForwardDiff.gradient(θ) do θ
                ELBO(m, re(θ)...)
            end
        end
        state = merge(state, (; hyperopt_state=hp_state))
    end
    return state
end

@traitfn function update_hyperparameters!(
    m::TGP, state, x, y
) where {TGP <: AbstractGPModel; !IsFull{TGP}}
    # Check that here is least one optimiser
    if any((!) ∘ isnothing ∘ opt, m.f) || any((!) ∘ isnothing ∘ Zopt, m.f)
        hp_state = state.hyperopt_state
        μ₀ = pr_means(m)
        ks = kernels(m)
        Zs = Zviews(m)
        if ADBACKEND[] == :Zygote
            Δμ₀, Δk, ΔZ = Zygote.gradient(μ₀, ks, Zs) do μ₀, ks, Zs
                ELBO(m, x, y, μ₀, ks, Zs, state)
            end
            # Optimize prior mean
            if !isnothing(Δμ₀)
                hp_state = update!.(μ₀, Δμ₀, hp_state)
            end

            # Optimize kernel parameters
            if isnothing(Δk)
                @warn "Kernel gradients are equal to zero" maxlog = 1
            else
                hp_state = map(m.f, Δk, hp_state) do gp, Δ, hp_st
                    if !isnothing(opt(gp))
                        state_k = update_kernel!(opt(gp), kernel(gp), Δ, hp_st.state_k)
                        return merge(hp_st, (; state_k))
                    else
                        return hp_st
                    end
                end
            end

            # Optimize inducing point locations
            if isnothing(ΔZ)
                @warn "Inducing point locations gradients are equal to zero" maxlog = 1
            else
                hp_state = map(m.f, ΔZ, hp_state) do gp, Δ, hp_st
                    if !isnothing(Zopt(gp))
                        state_Z = update_Z!(Zopt(gp), Zview(gp), Δ, hp_st.state_Z)
                        return merge(hp_st, (; state_Z))
                    else
                        return hp_st
                    end
                end
            end
        elseif ADBACKEND[] == :ForwardDiff
            θ, re = destructure((μ₀, ks, Zs))
            Δ = ForwardDiff.gradient(θ) do θ
                ELBO(m, re(θ)...)
            end
        end
        state = merge(state, (; hyperopt_state=hp_state))
    end
    return state
end

## Update all hyperparameters for the full batch GP models ##
function update_hyperparameters!(gp::AbstractLatent, X::AbstractVector)
    if !isnothing(gp.opt)
        f_l, f_μ₀ = hyperparameter_gradient_function(gp, X)
        ad_backend = K_ADBACKEND[] == :auto ? ADBACKEND[] : K_ADBACKEND[]
        Δμ₀ = f_μ₀()
        Δk = if ad_backend == :forward
            ∇L_ρ_forward(f_l, gp, X)
        elseif ad_backend == :zygote
            ∇L_ρ_zygote(f_l, gp, X)
        end
        apply_Δk!(gp.opt, kernel(gp), Δk) # Apply gradients to the kernel parameters
        apply_gradients_mean_prior!(pr_mean(gp), Δμ₀, X)
    end
end

## Update all hyperparameters for the sparse variational GP models ##
function update_hyperparameters!(
    gp::AbstractLatent,
    X::AbstractVector,
    ∇E_μ::AbstractVector,
    ∇E_Σ::AbstractVector,
    i::AbstractInference,
    vi_opt::InferenceOptimizer,
)
    Δμ₀, Δk = if !isnothing(gp.opt)
        f_ρ, f_Z, f_μ₀ = hyperparameter_gradient_function(gp)
        ad_backend = K_ADBACKEND[] == :auto ? ADBACKEND[] : K_ADBACKEND[]
        Δμ₀ = f_μ₀()
        Δk = if ad_backend == :forward
            ∇L_ρ_forward(f_ρ, gp, X, ∇E_μ, ∇E_Σ, i, vi_opt)
        elseif ad_backend == :zygote
            ∇L_ρ_zygote(f_ρ, gp, X, ∇E_μ, ∇E_Σ, i, vi_opt)
        end
        (Δμ₀, Δk)
    else
        nothing, nothing
    end
    if !isnothing(Zopt(gp))
        ad_backend = Z_ADBACKEND[] == :auto ? ADBACKEND[] : Z_ADBACKEND[]
        Z_grads = if ad_backend == :forward
            Z_gradient_forward(gp, f_Z, X, ∇E_μ, ∇E_Σ, i, vi_opt) #Compute the gradient given the inducing points location
        elseif ad_backend == :zygote
            Z_gradient_zygote(gp, f_Z, X, ∇E_μ, ∇E_Σ, i, vi_opt)
        end
        update_Z!(opt(gp.Z), gp.Z, Z_grads) #Apply the gradients on the location
    end
    if !all([isnothing(Δk), isnothing(Δμ₀)])
        apply_Δk!(gp.opt, kernel(gp), Δk) # Apply gradients to the kernel parameters
        apply_gradients_mean_prior!(pr_mean(gp), Δμ₀, X)
    end
end

function update_hyperparameters!(
    gp::AbstractLatent,
    l::AbstractLikelihood,
    i::AbstractInference,
    X::AbstractVector,
    Y::AbstractVector,
)
    Δμ₀, Δk = if !isnothing(gp.opt)
        ad_backend = K_ADBACKEND[] == :auto ? ADBACKEND[] : K_ADBACKEND[]
        (Δμ₀, Δk) = if ad_backend == :forward
            ∇L_ρ_forward(f_ρ, gp, X, ∇E_μ, ∇E_Σ, i, vi_opt)
        elseif ad_backend == :zygote
            ∇L_ρ_zygote(gp, l, i, X, Y)
        end
    else
        nothing, nothing
    end
    if !isnothing(opt(gp.Z))
        ad_backend = Z_ADBACKEND[] == :auto ? ADBACKEND[] : Z_ADBACKEND[]
        global Z_grads = if ad_backend == :forward
            Z_gradient_forward(gp, f_Z, X, ∇E_μ, ∇E_Σ, i, vi_opt) #Compute the gradient given the inducing points location
        elseif ad_backend == :zygote
            Z_gradient_zygote(gp, l, i, X, Y)
        end
        update!(opt(gp.Z), gp.Z.Z, Z_grads) #Apply the gradients on the location
    end
    if !all([isnothing(Δk), isnothing(Δμ₀)])
        apply!(kernel(gp), Δk, gp.opt) # Apply gradients to the kernel parameters
        update!(pr_mean(gp), Δμ₀, X)
    end
end

## Return the derivative of the KL divergence between the posterior and the GP prior ##
function hyperparameter_KL_gradient(J::AbstractMatrix, A::AbstractMatrix)
    return 0.5 * trace_ABt(J, A)
end

function hyperparameter_gradient_function(gp::LatentGP{T}, ::AbstractVector) where {T}
    A = (inv(cov(gp)) - mean(gp) * transpose(mean(gp))) # μ = inv(K+σ²)*(y-μ₀)
    return (function (Jnn)
        return -hyperparameter_KL_gradient(Jnn, A)
    end, function ()
        return -mean(gp)
    end)
end

## Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model ##
function hyperparameter_gradient_function(
    gp::VarLatent{T}, X::AbstractVector
) where {T<:Real}
    μ₀ = pr_mean(gp, X)
    A =
        (I - pr_cov(gp) \ (cov(gp) + (mean(gp) - μ₀) * transpose(mean(gp) - μ₀))) /
        pr_cov(gp)
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
    A =
        (I(dim(gp)) - pr_cov(gp) \ (cov(gp) + (mean(gp) - μ₀) * transpose(mean(gp) - μ₀))) /
        cov(gp)
    κΣ = gp.κ * cov(gp)
    return (
        function (Jmm, Jnm, Jnn, ∇E_μ, ∇E_Σ, i, opt)
            return (
                hyperparameter_expec_gradient(gp, ∇E_μ, ∇E_Σ, i, opt, κΣ, Jmm, Jnm, Jnn) -
                hyperparameter_KL_gradient(Jmm, A)
            )
        end,
        function (Jmm, Jnm, ∇E_μ, ∇E_Σ, i, opt)
            return hyperparameter_expec_gradient(
                gp, ∇E_μ, ∇E_Σ, i, opt, κΣ, Jmm, Jnm, zero(gp.K̃)
            ) - hyperparameter_KL_gradient(Jmm, A)
        end,
        function ()
            return pr_cov(gp) \ (mean(gp) - μ₀)
        end,
    )
end

function hyperparameter_gradient_function(
    gp::TVarLatent{T}, X::AbstractVector
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
        (I(dim(gp)) - pr_cov(gp) \ (cov(gp) + (mean(gp) - μ₀) * transpose(mean(gp) - μ₀))) /
        pr_cov(gp)
    κΣ = gp.κ * cov(gp)
    κₐΣ = gp.κₐ * cov(gp)
    return (
        function (Jmm, Jnm, Jnn, Jab, Jaa, ∇E_μ, ∇E_Σ, i, opt)
            ∇E = hyperparameter_expec_gradient(gp, ∇E_μ, ∇E_Σ, i, opt, κΣ, Jmm, Jnm, Jnn)
            ∇KLₐ = hyperparameter_online_gradient(gp, κₐΣ, Jmm, Jab, Jaa)
            ∇KL = hyperparameter_KL_gradient(Jmm, A)
            return ∇E + ∇KLₐ - ∇KL
        end, # Function gradient given kernel parameters
        function (Jmm, Jnm, Jab, ∇E_μ, ∇E_Σ, i, opt)
            return hyperparameter_expec_gradient(
                gp, ∇E_μ, ∇E_Σ, i, opt, κΣ, Jmm, Jnm, zero(gp.K̃)
            ) + hyperparameter_online_gradient(
                gp, κₐΣ, Jmm, Jab, zeros(T, length(gp.Zₐ), length(gp.Zₐ))
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
    J̃ = Jnn - (diag_ABt(ι, gp.Knm) + diag_ABt(gp.κ, Jnm))
    dμ = dot(∇E_μ, ι * mean(gp))
    dΣ = -dot(∇E_Σ, J̃)
    dΣ += -dot(∇E_Σ, 2.0 * (diag_ABt(ι, κΣ)))
    dΣ += -dot(∇E_Σ, 2.0 * (ι * mean(gp)) .* (gp.κ * mean(gp)))
    return ρ(i) * (dμ + dΣ)
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
    J̃ = Jnn - (diag_ABt(ι, gp.Knm) + diag_ABt(gp.κ, Jnm))
    dμ = dot(∇E_μ, ι * mean(gp))
    dΣ = dot(∇E_Σ, J̃ + 2.0 * diag_ABt(ι, κΣ))
    return ρ(i) * (dμ + dΣ)
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
        -0.5 * sum(
            trace_ABt.(
                [gp.invDₐ],
                [
                    Jaa,
                    2 * ιₐ * transpose(κₐΣ),
                    -ιₐ * transpose(gp.Kab),
                    -gp.κₐ * transpose(Jab),
                ],
            ),
        )
    term_1 = dot(gp.prevη₁, ιₐ * mean(gp))
    term_2 = -dot(ιₐ * mean(gp), gp.invDₐ * gp.κₐ * mean(gp))
    return trace_term + term_1 + term_2
end
