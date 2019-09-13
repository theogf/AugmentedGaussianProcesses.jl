"""
**Gaussian Likelihood**

Classical Gaussian noise : ``p(y|f) = \\mathcal{N}(y|f,\\epsilon)``

```julia
GaussianLikelihood(ϵ::T=1e-3) #ϵ is the variance
```

There is no augmentation needed for this likelihood which is already conjugate
Note that the variable ϵ is optimized over time!
"""
struct GaussianLikelihood{T<:Real} <: RegressionLikelihood{T}
    ϵ::LatentArray{T}
    θ::LatentArray{Vector{T}}
    function GaussianLikelihood{T}(ϵ::AbstractVector{T}) where {T<:Real}
        new{T}(ϵ)
    end
    function GaussianLikelihood{T}(ϵ::AbstractVector{T},θ::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
        new{T}(ϵ,θ)
    end
end

function GaussianLikelihood(ϵ::T=1e-3) where {T<:Real}
    GaussianLikelihood{T}([ϵ])
end

function GaussianLikelihood(ϵ::AbstractVector{T}) where {T<:Real}
    GaussianLikelihood{T}(ϵ)
end

function pdf(l::GaussianLikelihood,y::Real,f::Real)
    pdf(Normal(y,l.ϵ[1]),f) #WARNING multioutput invalid
end

function logpdf(l::GaussianLikelihood,y::Real,f::Real)
    logpdf(Normal(y,l.ϵ[1]),f) #WARNING multioutput invalid
end

function Base.show(io::IO,model::GaussianLikelihood{T}) where T
    print(io,"Gaussian likelihood")
end

function compute_proba(l::GaussianLikelihood{T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
    return μ,max.(σ²,0.0).+ l.ϵ[1]
end

function init_likelihood(likelihood::GaussianLikelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int,nFeatures::Int) where {T<:Real}
    if length(likelihood.ϵ) ==1 && length(likelihood.ϵ) != nLatent
        return GaussianLikelihood{T}([likelihood.ϵ[1] for _ in 1:nLatent],[fill(inv(likelihood.ϵ[1]),nSamplesUsed) for _ in 1:nLatent])
    elseif length(likelihood.ϵ) != nLatent
        @warn "Wrong dimension of ϵ : $(length(likelihood.ϵ)), using first value only"
        return GaussianLikelihood{T}([likelihood.ϵ[1] for _ in 1:nLatent])
    else
        return GaussianLikelihood{T}(likelihood.ϵ,[fill(likelihood.ϵ[i],nSamplesUsed) for i in 1:nLatent])
    end
end

function local_updates!(model::GP{T,<:GaussianLikelihood}) where {T}
    if model.opt_noise
        model.likelihood.ϵ .= inv(model.nFeatures)*broadcast((y,μ,Σ)->sum(abs2,y)-2*dot(y,μ)+sum(abs2,μ)+tr(Σ),model.y,model.Knn.*model.invKnn.*model.y,model.Knn.*([I].-model.invKnn.*model.Knn))
    end
end

function local_updates!(model::SVGP{T,<:GaussianLikelihood}) where {T}
    if model.inference.Stochastic
        #TODO make it a moving average
        ρ = inv(sqrt(1+model.inference.nIter))
        model.likelihood.ϵ .= (1-ρ)*model.likelihood.ϵ + ρ/model.inference.nSamplesUsed *broadcast((y,κ,μ,Σ,K̃)->sum(abs2.(y-κ*μ))+opt_trace(κ*Σ,κ)+sum(K̃),model.inference.y,model.κ,model.μ,model.Σ,model.K̃)
    else
        model.likelihood.ϵ .= 1.0/model.inference.nSamplesUsed *broadcast((y,κ,μ,Σ,K̃)->sum(abs2.(y-κ*μ))+opt_trace(κ*Σ,κ)+sum(K̃),model.y,model.κ,model.μ,model.Σ,model.K̃)
    end
    model.likelihood.θ .= broadcast(ϵ->fill(inv(ϵ),model.inference.nSamplesUsed),model.likelihood.ϵ)
end

function local_updates!(model::VStP{T,<:GaussianLikelihood}) where {T}
    model.likelihood.ϵ .= 1.0/model.inference.nSamples *broadcast((y,μ,Σ)->sum(abs2,y-μ)+Σ,model.y,model.μ,tr.(model.Σ))
    model.likelihood.θ .= broadcast(ϵ->fill(inv(ϵ),model.inference.nSamplesUsed),model.likelihood.ϵ)
end

@inline ∇E_μ(model::AbstractGP{T,<:GaussianLikelihood,<:AnalyticVI}) where {T} = model.inference.y./model.likelihood.ϵ
@inline ∇E_μ(model::AbstractGP{T,<:GaussianLikelihood,<:AnalyticVI},i::Int) where {T} = model.inference.y[i]./model.likelihood.ϵ[i]
@inline ∇E_Σ(model::AbstractGP{T,<:GaussianLikelihood,<:AnalyticVI}) where {T} = 0.5*model.likelihood.θ
@inline ∇E_Σ(model::AbstractGP{T,<:GaussianLikelihood,<:AnalyticVI},i::Int) where {T} = 0.5*model.likelihood.θ[i]



function predict_f(model::GP{T,GaussianLikelihood{T},Analytic{T}},X_test::AbstractMatrix{T};covf::Bool=true,fullcov::Bool=false) where {T}
    k_star = kernelmatrix.([X_test],[model.inference.x],model.kernel)
    μf = k_star.*model.invKnn.*model.inference.y
    if !covf
        return model.nLatent == 1 ? μf[1] : μf
    end
    if fullcov
        Σf = Symmetric.(kernelmatrix.([X_test],model.kernel) .- k_star.*model.invKnn.*transpose.(k_star))
        i = 0
        ϵ = 1e-16
        while count(isposdef.(Σf))!=model.nLatent
            Σf .= ifelse.(isposdef.(Σf),Σf,Σf.+ϵ.*[I])
            if i > 100
                println("DAMN")
                break;
            end
            ϵ *= 2
            i += 1
        end
        @assert count(isposdef.(Σf))==model.nLatent
        return model.nLatent == 1 ? (μf[1],Σf[1]) : (μf,Σf)
    else
        σ²f = kerneldiagmatrix.([X_test],model.kernel) .- opt_diag.(k_star.*model.invKnn,k_star)
        return model.nLatent == 1 ? (μf[1],σ²f[1]) : (μf,σ²f)
    end
end


function proba_y(model::GP{T,GaussianLikelihood{T},Analytic{T}},X_test::AbstractMatrix{T}) where {T}
    μf, σ²f = predict_f(model,X_test,covf=true)
    σ²f .+= model.likelihood.ϵ
    return μf,σ²f
end

function proba_y(model::SVGP{T,GaussianLikelihood{T},AnalyticVI{T}},X_test::AbstractMatrix{T}) where {T}
    μf, σ²f = predict_f(model,X_test,covf=true)
    σ²f .+= model.likelihood.ϵ
    return μf,σ²f
end

### Special case where the ELBO is equal to the marginal likelihood
function ELBO(model::GP{T,GaussianLikelihood{T}}) where {T}
    return -0.5*sum(broadcast((y,invK)->dot(y,invK*y) - logdet(invK)+ model.nFeatures*log(twoπ),model.y,model.invKnn))
end

function ELBO(model::SVGP{T,GaussianLikelihood{T}}) where {T}
    return expecLogLikelihood(model) - GaussianKL(model)
end

function expecLogLikelihood(model::SVGP{T,GaussianLikelihood{T}}) where {T}
    return -0.5*model.inference.ρ*sum(broadcast((y,ϵ,κ,Σ,μ,K̃)->(sum(abs2,y-κ*μ)+sum(K̃)+opt_trace(κ*Σ,κ))/ϵ+model.inference.nSamplesUsed*(log(twoπ)+log(ϵ)),model.inference.y,model.likelihood.ϵ,model.κ,model.Σ,model.μ,model.K̃))
end

function hyperparameter_gradient_function(model::GP{T,GaussianLikelihood{T}}) where {T}
    A = ([I].-model.invKnn.*(model.y.*transpose.(model.y))).*model.invKnn
    if model.IndependentPriors
        return (function(Jnn,index)
                    return -0.5*hyperparameter_KL_gradient(Jnn,A[index])
                end,
                function(kernel,index)
                    return -0.5/getvariance(kernel)*opt_trace(model.Knn[index],A[index])
                end,
                function(index)
                    return -model.invKnn[index]*(model.μ₀[index]-model.y[index])
                end)
    else
        return (function(Jnn,index)
            return -0.5*sum(hyperparameter_KL_gradient(Jnn,A[i]) for i in 1:model.nLatent)
                end,
                function(kernel,index)
                    return -0.5/getvariance(kernel)*sum(opt_trace(model.Knn[1],A[i]) for i in 1:model.nLatent)
                end,
                function(index)
                    return -sum(model.invKnn.*(model.μ₀.-model.μ))
                end)
    end
end
