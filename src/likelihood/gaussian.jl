"""
Gaussian likelihood : ``p(y|f) = 𝓝(y|f,ϵ) ``
"""
struct GaussianLikelihood{T<:Real} <: RegressionLikelihood{T}
    ϵ::AbstractVector{T}
    function GaussianLikelihood{T}(ϵ::Real) where {T<:Real}
        new{T}([ϵ])
    end
    function GaussianLikelihood{T}(ϵ::AbstractVector) where {T<:Real}
        new{T}(ϵ)
    end
end

function GaussianLikelihood(ϵ::T=1e-3) where {T<:Real}
    GaussianLikelihood{T}(ϵ)
end

function GaussianLikelihood(ϵ::AbstractVector{T}) where {T<:Real}
    GaussianLikelihood{T}(ϵ)
end

function pdf(l::GaussianLikelihood,y::Real,f::Real)
    pdf(Normal(y,l.ϵ[1]),f)
end

function logpdf(l::GaussianLikelihood,y::Real,f::Real)
    logpdf(Normal(y,l.ϵ[1]),f)
end

function Base.show(io::IO,model::GaussianLikelihood{T}) where T
    print(io,"Gaussian likelihood")
end


function init_likelihood(likelihood::GaussianLikelihood{T},nLatent::Integer,nSamples::Integer) where {T<:Real}
    if length(likelihood.ϵ) ==1 && length(likelihood.ϵ) != nLatent
        return GaussianLikelihood{T}([likelihood.ϵ[1] for _ in 1:nLatent])
    elseif length(likelihood.ϵ) != nLatent
        @warn "Wrong dimension of ϵ : $(length(likelihood.ϵ)), using first value only"
        return GaussianLikelihood{T}([likelihood.ϵ[1] for _ in 1:nLatent])
    else
        return likelihood
    end
end

function local_updates!(model::GP{GaussianLikelihood{T}}) where {T<:Real}
end

function local_updates!(model::SVGP{GaussianLikelihood{T}}) where {T<:Real}
    model.likelihood.ϵ .= 1.0/model.inference.nSamplesUsed *broadcast((y,κ,μ,Σ,K̃)->sum(abs2.(y[model.inference.MBIndices]-κ*μ))+opt_trace(κ*Σ,κ)+sum(K̃),model.y,model.κ,model.μ,model.Σ,model.K̃)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{GaussianLikelihood{T},AnalyticInference{T}},index::Integer) where {T<:Real}
    return model.y[index][model.inference.MBIndices]./model.likelihood.ϵ[index]
end

function ∇μ(model::SVGP{GaussianLikelihood{T},AnalyticInference{T}}) where {T<:Real}
    return getindex.(model.y,[model.inference.MBIndices])./model.likelihood.ϵ
end

function expec_Σ(model::SVGP{GaussianLikelihood{T},AnalyticInference{T}},index::Integer) where {T<:Real}
    return fill(0.5/model.likelihood.ϵ[index],model.inference.nSamplesUsed)
end

function ∇Σ(model::SVGP{GaussianLikelihood{T},AnalyticInference{T}}) where {T<:Real}
    return [fill(0.5/model.likelihood.ϵ[i],model.inference.nSamplesUsed) for i in 1:model.nLatent]
end


function predict_f(model::GP{GaussianLikelihood{T},AnalyticInference{T}},X_test::AbstractMatrix{T};covf::Bool=true,fullcov::Bool=false) where {T<:Real}
    k_star = kernelmatrix.([X_test],[model.X],model.kernel)
    μf = k_star.*model.invKnn.*model.y
    if !covf
        return model.nLatent == 1 ? μf[1] : μf
    end
    if fullcov
        Σf = Symmetric.(kernelmatrix.([X_test],model.kernel) .- k_star.*model.invKnn.*transpose.(k_star))
        i = 0
        while count(isposdef.(Σf))!=model.nLatent
            Σf .= ifelse.(isposdef.(Σf),Σf,Σf.+0.01.*[I])
            if i > 100
                println("DAMN")
                break;
            end
            i += 1
        end
        @assert count(isposdef.(Σf))==model.nLatent
        println(typeof(Σf))
        return model.nLatent == 1 ? (μf[1],Σf[1]) : (μf,Σf)
    else
        σ²f = kerneldiagmatrix.([X_test],model.kernel) .- opt_diag.(k_star.*model.invKnn,k_star)
        return model.nLatent == 1 ? (μf[1],σ²f[1]) : (μf,σ²f)
    end
end


function proba_y(model::AbstractGP{GaussianLikelihood{T},AnalyticInference{T}},X_test::AbstractMatrix{T}) where {T<:Real}
    μf, σ²f = predict_f(model,X_test,covf=true)
end

### Special case where the ELBO is equal to the marginal likelihood
function ELBO(model::GP{GaussianLikelihood{T}}) where {T<:Real}
    return -0.5*sum(broadcast((y,invK)->dot(y,invK*y) - logdet(invK)+ model.nFeature*log(twoπ),model.y,model.invKnn))
end

function ELBO(model::SVGP{GaussianLikelihood{T}}) where {T<:Real}
    return expecLogLikelihood(model) - GaussianKL(model)
end

function expecLogLikelihood(model::SVGP{GaussianLikelihood{T}}) where T
    return -0.5*model.inference.ρ*sum(broadcast((y,ϵ,κ,Σ,μ,K̃)->1.0/ϵ*(sum(abs2.(y[model.inference.MBIndices]-κ*μ))+sum(K̃)+opt_trace(κ*Σ,κ))+model.inference.nSamplesUsed*(log(twoπ)+log(ϵ)),model.y,model.likelihood.ϵ,model.κ,model.Σ,model.μ,model.K̃))
end

function hyperparameter_gradient_function(model::GP{GaussianLikelihood{T}}) where {T<:Real}
    A = ([I].-model.invKnn.*(model.y.*transpose.(model.y))).*model.invKnn
    if model.IndependentPriors
        return (function(Jnn,index)
                    return -0.5*hyperparameter_KL_gradient(Jnn,A[index])
                end,
                function(kernel,index)
                    return -0.5/getvariance(kernel)*opt_trace(model.Knn[index],A[index])
                end)
    else
        return (function(Jnn,index)
            return -0.5*sum(hyperparameter_KL_gradient(Jnn,A[i]) for i in 1:model.nLatent)
                end,
                function(kernel,index)
                    return -0.5/getvariance(kernel)*sum(opt_trace(model.Knn[1],A[i]) for i in 1:model.nLatent)
                end)
    end
end
