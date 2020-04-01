"""
```julia
HeteroscedasticLaplaceLikelihood(λ::T=1.0)
```

Gaussian with heteroscedastic noise given by another gp:
```math
    p(y|f,g,β) = N(y|f,(sqrt(2β σ(g)))⁻¹)
```
Where `σ` is the logistic function

Augmentation will be described in a future paper
"""
mutable struct HeteroscedasticLaplaceLikelihood{T<:Real} <:
               RegressionLikelihood{T}
    β::T # Maximum of precision
    c::Vector{T} # Second variational parameter of ω
    ϕ::Vector{T} # Variational paraemter of n
    ψ::Vector{T} # Second var parameter of γ
    θ::Vector{T} # Expectation of ω
    σg::Vector{T} # Expectation of σ(g)
    function HeteroscedasticLaplaceLikelihood{T}(β::T) where {T<:Real}
        new{T}(β)
    end
    function HeteroscedasticLaplaceLikelihood{T}(
        β::T,
        c::AbstractVector{T},
        ϕ::AbstractVector{T},
        ψ::AbstractVector{T},
        θ::AbstractVector{T},
        σg::AbstractVector{T},
    ) where {T<:Real}
        new{T}(β, c, ϕ, ψ, θ, σg)
    end
end

function HeteroscedasticLaplaceLikelihood(β::T=1.0) where {T<:Real}
    @assert β > 0
    HeteroscedasticLaplaceLikelihood{T}(β)
end

implemented(::HeteroscedasticLaplaceLikelihood,::GibbsSampling) = true


function pdf(l::HeteroscedasticLaplaceLikelihood,y::Real,f::AbstractVector)
    pdf(Laplace(y,inv(sqrt(2*l.β*logistic(f[2])))),f[1])
end

function logpdf(l::HeteroscedasticLaplaceLikelihood,y::Real,f::AbstractVector)
    logpdf(Normal(y,inv(sqrt(2*l.β*logistic(f[2])))),f[1])
end

function Base.show(io::IO,model::HeteroscedasticLaplaceLikelihood{T}) where T
    print(io,"Laplace likelihood with heteroscedastic noise")
end

num_latent(::HeteroscedasticLaplaceLikelihood) = 2

function treat_labels!(y::AbstractVector{T},likelihood::L) where {T,L<:HeteroscedasticLaplaceLikelihood}
    @assert T<:Real "For regression target(s) should be real valued"
    return y, 2, likelihood
end

function init_likelihood(
    likelihood::HeteroscedasticLaplaceLikelihood{T},
    inference::Inference{T},
    nLatent::Integer,
    nMinibatch::Integer,
    nFeatures::Integer,
) where {T<:Real}
    β = likelihood.β
    c = ones(T, nMinibatch)
    ϕ = ones(T, nMinibatch)
    ψ = ones(T, nMinibatch)
    θ = ones(T, nMinibatch)
    σg = ones(T, nMinibatch)
    HeteroscedasticLaplaceLikelihood{T}(β, c, ϕ, ψ, θ, σg)
end

function compute_proba(
    l::HeteroscedasticLaplaceLikelihood{T},
    μ::AbstractVector{<:AbstractVector},
    Σ::AbstractVector{<:AbstractVector},
) where {T}
    return μ[1], max.(Σ[1], zero(T)) .+ inv.(l.β * logistic.(μ[2]))
end

function local_updates!(l::HeteroscedasticLaplaceLikelihood{T},y::AbstractVector,μ::NTuple{2,<:AbstractVector},diag_cov::NTuple{2,<:AbstractVector}) where {T}
    # gp[1] is f and gp[2] is g (for approximating the noise)
    l.σg .= expectation.(logistic, μ[2], diag_cov[2])
    l.c .= sqrt.(abs2.(μ[2]) + diag_cov[2])
    l.ψ .= 0.5 * (abs2.(μ[1] - y) + diag_cov[1])
    l.ϕ .= 0.5 * l.β * l.σg ./ l.ψ
    l.θ .= 0.5 * (1.0 .+ l.γ) ./ l.c .* tanh.(0.5 * l.c)
    # l.λ = 0.5 * length(l.ϕ) / dot(l.ϕ, l.σg)
end

function local_autotuning!(model::VGP{T,<:HeteroscedasticLaplaceLikelihood}) where {T}
    Jnn = kernelderivativematrix.([model.X],model.likelihood.kernel)
    f_l,f_v,f_μ₀ = hyperparameter_local_gradient_function(model)
    grads_l = map(compute_hyperparameter_gradient,model.likelihood.kernel,fill(f_l,model.nLatent),Jnn,1:model.nLatent)
    grads_v = map(f_v,model.likelihood.kernel,1:model.nPrior)
    grads_μ₀ = map(f_μ₀,1:model.nLatent)

    apply_gradients_lengthscale!.(model.likelihood.kernel,grads_l) #Send the derivative of the matrix to the specific gradient of the model
    apply_gradients_variance!.(model.likelihood.kernel,grads_v) #Send the derivative of the matrix to the specific gradient of the model
    update!.(model.likelihood.μ₀,grads_μ₀)

    model.inference.HyperParametersUpdated = true
end

function variational_updates!(
    model::AbstractGP{T,<:HeteroscedasticLaplaceLikelihood,<:AnalyticVI},
) where {T,L}
    local_updates!(
        model.likelihood,
        get_y(model),
        mean_f(model),
        diag_cov_f(model),
    )
    natural_gradient!(
        ∇E_μ(model.likelihood, opt_type(model.inference), get_y(model))[2],
        ∇E_Σ(model.likelihood, opt_type(model.inference), get_y(model))[2],
        getρ(model.inference),
        opt_type(model.inference),
        get_Z(model, 2),
        model.f[2],
    )
    global_update!(model.f[2], opt_type(model.inference), model.inference)
    heteroscedastic_expectations!(
        model.likelihood,
        mean_f(model.f[2]),
        diag_cov_f(model.f[2]),
    )
    natural_gradient!(
        ∇E_μ(model.likelihood, opt_type(model.inference), get_y(model))[1],
        ∇E_Σ(model.likelihood, opt_type(model.inference), get_y(model))[1],
        getρ(model.inference),
        opt_type(model.inference),
        get_Z(model, 1),
        model.f[1],
    )
    global_update!(model.f[1], opt_type(model.inference), model.inference)
end

function heteroscedastic_expectations!(l::HeteroscedasticLaplaceLikelihood{T},μ::AbstractVector,Σ::AbstractVector) where {T}
    l.σg .= expectation.(logistic,μ,Σ)
    # l.β = 0.5*length(l.ϕ)/dot(l.ϕ,l.σg)
end

@inline ∇E_μ(l::HeteroscedasticLaplaceLikelihood,::AOptimizer,y::AbstractVector) where {T} = (0.5 * l.ψ .* y , 0.5 * (1.0 .- l.ϕ))

@inline ∇E_Σ(l::HeteroscedasticLaplaceLikelihood,::AOptimizer,y::AbstractVector) where {T} = (0.5 * l.ψ, 0.5 * l.θ)

function proba_y(
    model::AbstractGP{T,HeteroscedasticLaplaceLikelihood{T},AnalyticVI{T}},
    X_test::AbstractMatrix{T},
) where {T<:Real}
    (μf, σ²f), (μg, σ²g) = predict_f(model, X_test, covf = true)
    return μf,
    σ²f + expectation.(x -> inv(2 * model.likelihood.β * logistic(x)), μg, σ²g)
end

function expec_log_likelihood(l::HeteroscedasticLaplaceLikelihood{T},i::AnalyticVI,y::AbstractVector,μ,diag_cov) where {T}
    tot = length(y)*(0.5*log(l.λ)-log(2*sqrt(twoπ)))
    tot += 0.5*(dot(μ[2],(0.5 .- l.γ)) - dot(abs2.(μ[2]),l.θ)-dot(diag_cov[2],l.θ))
    tot -= PoissonKL(l,y,μ[1],diag_cov[1])
    return tot
end

AugmentedKL(l::HeteroscedasticLaplaceLikelihood,::AbstractVector) = PolyaGammaKL(l)

function PoissonKL(l::HeteroscedasticLaplaceLikelihood{T},y::AbstractVector,μ::AbstractVector,Σ::AbstractVector) where {T}
    return PoissonKL(l.γ,0.5*l.β*(abs2.(y-μ)+Σ),log.(0.5*l.λ*(abs2.(μ-y)+Σ)))
end

function PolyaGammaKL(l::HeteroscedasticLaplaceLikelihood{T}) where {T}
    PolyaGammaKL(0.5.+l.γ,l.c,l.θ)
end
