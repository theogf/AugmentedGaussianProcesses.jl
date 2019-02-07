"""
Logistic likelihood : ``p(y|f) = σ(yf) = (1+exp(-yf))⁻¹ ``
"""
abstract type AbstractLogisticLikelihood{T<:Real} <: Likelihood{T} end;

struct LogisticLikelihood{T<:Real} <: AbstractLogisticLikelihood{T}
    function LogisticLikelihood{T}() where {T<:Real}
        new{T}()
    end
end

struct AugmentedLogisticLikelihood{T<:Real} <: Likelihood{T}
    c::AbstractVector{AbstractVector{T}}
    θ::AbstractVector{AbstractVector{T}}
    function AugmentedLogisticLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function AugmentedLogisticLikelihood{T}(c::AbstractVector{AbstractVector{<:Real}},θ::AbstractVector{AbstractVector{<:Real}})
        new{T}(c,θ)
    end
end

function LogisticLikelihood(Augmented::Bool=true)
    if Augmented
        AugmentedLogisticLikelihood{Float64}()
    else
        LogisticLikelihood{Float64}()
    end
end

function AugmentedLogisticLikelihood()
    AugmentedLogisticLikelihood{Float64}()
end

function pdf(l::AbstractLogisticLikelihood,y::Real,f::Real)
    logit(y*f)
end

function init_likelihood(likelihood::AugmentedLogisticLikelihood{T},nLatent::Integer,nSamplesUsed) where T
    LogisticLikelihood{T}([abs.(rand(T,nSamplesUsed)) for _ in 1:nLatent],[zeros(T,nSamplesUsed) for _ in 1:nLatent])
end

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractArray{T,N},likelihood::L) where {T,N,L<:AbstractLogisticLikelihood}
    @assert T<:Real "For classification target(s) should be real valued (Bool,Integer or Float)"
    @assert N <= 2 "Target should be a matrix or a vector"
    labels = Int64.(unique(y))
    @assert count(labels) <= 2 && (labels == [0 1] || labels == [-1 1]) "Labels of y should be binary {-1,1} or {0,1}"
    if N == 1
        return [y]
    else
        return [y[:,i] for i in 1:size(y,2)]
    end
end

function local_update!(model::VGP{AugmentedLogisticLikelihood{T}}) where T
    model.c .= broadcast((μ,Σ)->sqrt.(diag(Σ)+μ.^2),model.μ,model.Σ)
    model.θ .= broadcast(c->0.5*tanh.(0.5*c)./c,model.c)
end

function local_update!(model::SVGP{AugmentedLogisticLikelihood{T}}) where T
    model.c .= broadcast((μ,Σ,K̃,κ)->sqrt.(K̃+opt_diag(κ*Σ,κ')+(κ*μ).^2),model.μ,model.Σ,model.K̃,model.κ)
    model.θ .= broadcast(c->0.5*tanh.(0.5*c)./c,model.c)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{<:AugmentedLogisticLikelihood},index::Integer)
    return 0.5.*model.y[index]
end

function expec_μ(model::VGP{<:AugmentedLogisticLikelihood})
    return 0.5.*model.y
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:AugmentedLogisticLikelihood},index::Integer)
    return 0.5.*model.y[index][model.inference.MBIndices]
end

function expec_μ(model::SVGP{<:AugmentedLogisticLikelihood})
    return 0.5.*getindex.(model.y,[model.inference.MBIndices])
end

function expec_Σ(model::GP{<:AugmentedLogisticLikelihood},index::Integer)
    return 0.5*model.θ[index]
end

function expec_Σ(model::GP{<:AugmentedLogisticLikelihood})
    return 0.5*model.θ
end

function ELBO(model::GP{<:AugmentedLogisticLikelihood})
    return expecLogLikelihood(model) - GaussianKL(model) - PolyaGammaKL(model)
end

function expecLogLikelihood(model::VGP{<:AugmentedLogisticLikelihood)
    tot = -model.nLatent*(0.5*model.nSamples*log(2))
    tot += sum(broadcast((μ,y,θ,Σ)->0.5.*(sum(μ.*y)-opt_trace(θ,(diag(Σ)+μ.^2))),
                        model.μ,model.y,model.θ,model.Σ))
    return tot
end

function expecLogLikelihood(model::SVGP{<:AugmentedLogisticLikelihood)
    tot = -model.nLatent*(0.5*model.nSamples*log(2))
    tot += sum(broadcast((κμ,y,θ,κΣκ,K̃)->0.5.*(sum(κμ.*y)-opt_trace(θ,K̃+κΣκ+κμ.^2))),
                        model.κ.*model.μ,model.y,model.θ,opt_diag(model.κ*model.Σ,model.κ'),model.K̃)
    return model.inference.ρ*tot
end
