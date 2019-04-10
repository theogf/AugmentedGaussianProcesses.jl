"""
**Laplace likelihood**

Laplace likelihood for regression: ``\\frac{1}{2\\beta}\\exp\\left(-\\frac{|y-f|}{\\beta}\\right)``
see [wiki page](https://en.wikipedia.org/wiki/Laplace_distribution)

---

For the analytical solution, it is augmented via:
```math
#TODO
```

"""
struct LaplaceLikelihood{T<:Real} <: RegressionLikelihood{T}
    β::T
    θ::LatentArray{Vector{T}}
    λ::LatentArray{Vector{T}}
    function LaplaceLikelihood{T}(β::T) where {T<:Real}
        new{T}(β)
    end
    function LaplaceLikelihood{T}(β::T,λ::AbstractVector{<:AbstractVector{T}},θ::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
        new{T}(β,λ,θ)
    end
end

function LaplaceLikelihood(β::T=1.0) where {T<:Real}
    LaplaceLikelihood{T}(β)
end

function init_likelihood(likelihood::LaplaceLikelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int) where T
    if inference isa AnalyticVI || inference isa GibbsSampling
        LaplaceLikelihood{T}(likelihood.β,[abs2.(T.(rand(T,nSamplesUsed))) for _ in 1:nLatent],[zeros(T,nSamplesUsed) for _ in 1:nLatent])
    else
        LaplaceLikelihood{T}(likelihood.β)
    end
end

function pdf(l::LaplaceLikelihood,y::Real,f::Real)
    pdf(Laplace(f,l.β),y)
end

function Base.show(io::IO,model::LaplaceLikelihood{T}) where T
    print(io,"Laplace likelihood")
end

function compute_proba(l::LaplaceLikelihood{T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
    N = length(μ)
    nSamples = 2000
    μ_pred = zeros(T,N)
    σ²_pred = zeros(T,N)
    temp_array = zeros(T,nSamples)
    for i in 1:N
        # e = expectation(Normal(μ[i],sqrt(β²[i])))
        # μ_pred[i] = μ[i]
        #
        # β²_pred[i] = e(x->pdf(LocationScale(x,1.0,st))^2) - e(x->pdf(LocationScale(x,1.0,st)))^2
        if σ²[i] <= 1e-3
            pyf =  LocationScale(μ[i],l.β,st)
            for j in 1:nSamples
                temp_array[j] = rand(Laplace(μ[i],l.β))
            end
        else
            d = Normal(μ[i],sqrt(σ²[i]))
            for j in 1:nSamples
                temp_array[j] = rand(Laplace(rand(d),l.β))
            end
        end
        μ_pred[i] = μ[i];
        σ²_pred[i] = cov(temp_array)
    end
    return μ_pred,σ²_pred
end

###############################################################################

function local_updates!(model::VGP{<:LaplaceLikelihood,<:AnalyticVI})
    model.likelihood.λ .= broadcast((Σ,μ,y)->0.25*(Σ+abs2.(μ-y).+1.0./model.likelihood.β^2),diag.(model.Σ),model.μ,model.y)
end

function local_updates!(model::SVGP{<:LaplaceLikelihood,<:AnalyticVI})
    model.likelihood.λ .= broadcast((K̃,κ,Σ,μ,y)->0.25*(K̃ + opt_diag(κ*Σ,κ) + abs2.(κ*μ-y[model.inference.MBIndices]) .+1.0./model.likelihood.β^2),model.K̃,model.κ,model.Σ,model.μ,model.y)
end

function sample_local!(model::VGP{<:LaplaceLikelihood,<:GibbsSampling})
    model.likelihood.λ .= broadcast((μ::AbstractVector{<:Real})->rand.(InverseGamma.([0.5],0.5*(μ.^2.+1.0./model.likelihood.β^2))),model.μ)
    return nothing
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{<:LaplaceLikelihood,<:AnalyticVI},index::Integer)
    return model.likelihood.λ[index].*model.y[index]
end

function ∇μ(model::VGP{<:LaplaceLikelihood,<:AnalyticVI})
    return hadamard.(model.likelihood.λ,model.y)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:LaplaceLikelihood,<:AnalyticVI},index::Integer)
    return model.likelihood.λ[index].*model.y[index][model.inference.MBIndices]
end

function ∇μ(model::SVGP{<:LaplaceLikelihood,<:AnalyticVI})
    return hadamard.(model.likelihood.λ,getindex.(model.y,[model.inference.MBIndices]))
end

function expec_Σ(model::AbstractGP{<:LaplaceLikelihood,<:AnalyticVI},index::Integer)
    return 0.5*model.likelihood.λ[index]
end

function ∇Σ(model::AbstractGP{<:LaplaceLikelihood,<:AnalyticVI})
    return 0.5*model.likelihood.λ
end

function ELBO(model::AbstractGP{<:LaplaceLikelihood,<:AnalyticVI})
    return expecLogLikelihood(model) - InverseGammaKL(model) - GaussianKL(model)
end

function expecLogLikelihood(model::VGP{LaplaceLikelihood{T},AnalyticVI{T}}) where T
    tot = -0.5*model.nLatent*model.nSample*log(twoπ)
    # tot -= 0.5.*sum(broadcast(β->sum(log.(β).-model.nSample*digamma(model.likelihood.α)),model.likelihood.β))
    # tot -= 0.5.*sum(broadcast((β,Σ,μ,y)->dot(model.likelihood.α./β,Σ+abs2.(μ)-2.0*μ.*y-abs2.(y)),model.likelihood.β,diag.(model.Σ),model.μ,model.y))
    return tot
end

function expecLogLikelihood(model::SVGP{LaplaceLikelihood{T},AnalyticVI{T}}) where T
    # tot = -0.5*model.nLatent*model.inference.nSamplesUsed*log(twoπ)
    # tot -= 0.5.*sum(broadcast(β->sum(log.(β).-model.inference.nSamplesUsed*digamma(model.likelihood.α)),model.likelihood.β))
    # tot -= 0.5.*sum(broadcast((β,K̃,κ,Σ,μ,y)->dot(model.likelihood.α./β,(K̃+opt_diag(κ*Σ,κ)+abs2.(κ*μ)-2.0*(κ*μ).*y[model.inference.MBIndices]-abs2.(y[model.inference.MBIndices]))),model.likelihood.β,model.K̃,model.κ,model.Σ,model.μ,model.y))
    return model.inference.ρ*tot
end

function gradpdf(::LaplaceLikelihood,y::Int,f::T) where {T<:Real}
    @error "Not implemented yet"
end

function hessiandiagpdf(::LaplaceLikelihood,y::Int,f::T) where {T<:Real}
    @error "Not implemented yet"
end
