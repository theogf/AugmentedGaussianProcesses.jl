"""
**Gaussian Likelihood**

Classical Gaussian noise : ``p(y|f) = \\mathcal{N}(y|f,\\epsilon)``

```julia
GaussianLikelihood(ϵ::T=1e-3) #ϵ is the variance
```

There is no augmentation needed for this likelihood which is already conjugate
"""
mutable struct GaussianLikelihood{T<:Real} <: RegressionLikelihood{T}
    ϵ::T
    opt_noise::Bool
    θ::Vector{T}
    function GaussianLikelihood{T}(ϵ::T,opt_noise::Bool) where {T<:Real}
        new{T}(ϵ,opt_noise)
    end
    function GaussianLikelihood{T}(ϵ::T,opt_noise::Bool,θ::AbstractVector{T}) where {T<:Real}
        new{T}(ϵ,opt_noise,θ)
    end
end

function GaussianLikelihood(ϵ::T=1e-3;opt_noise::Bool=true) where {T<:Real}
    GaussianLikelihood{T}(ϵ,opt_noise)
end

function pdf(l::GaussianLikelihood,y::Real,f::Real)
    pdf(Normal(y,l.ϵ),f)
end

function logpdf(l::GaussianLikelihood,y::Real,f::Real)
    logpdf(Normal(y,l.ϵ),f)
end

function Base.show(io::IO,model::GaussianLikelihood{T}) where T
    print(io,"Gaussian likelihood")
end

function compute_proba(l::GaussianLikelihood{T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
    return μ,max.(σ²,zero(σ²)).+ l.ϵ
end

function init_likelihood(likelihood::GaussianLikelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int,nFeatures::Int) where {T<:Real}
    return GaussianLikelihood{T}(likelihood.ϵ,likelihood.opt_noise,fill(inv(likelihood.ϵ),nSamplesUsed))
end

function local_updates!(l::GaussianLikelihood{T},y::AbstractVector,μ::AbstractVector,Σ::AbstractVector) where {T}
    if l.opt_noise
        # ρ = inv(sqrt(1+model.inference.nIter))
        l.ϵ = sum(abs2.(y-μ)+Σ)/length(y)
    end
    l.θ .= inv(l.ϵ)
end

@inline ∇E_μ(l::GaussianLikelihood{T},::AVIOptimizer,y::AbstractVector) where {T} = y./l.ϵ
@inline ∇E_Σ(l::GaussianLikelihood{T},::AVIOptimizer,y::AbstractVector) where {T} = 0.5*l.θ

function predict_f(model::GP{T,GaussianLikelihood{T},Analytic{T}},X_test::AbstractMatrix{T};covf::Bool=true,fullcov::Bool=false) where {T}
    k_star = kernelmatrix.([X_test],[model.X],model.kernel)
    μf = k_star.*model.invKnn.*model.y
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
    return -0.5*model.inference.ρ*sum(broadcast((y,ϵ,κ,Σ,μ,K̃)->1.0/ϵ*(sum(abs2.(y[model.inference.MBIndices]-κ*μ))+sum(K̃)+opt_trace(κ*Σ,κ))+model.inference.nSamplesUsed*(log(twoπ)+log(ϵ)),model.y,model.likelihood.ϵ,model.κ,model.Σ,model.μ,model.K̃))
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
