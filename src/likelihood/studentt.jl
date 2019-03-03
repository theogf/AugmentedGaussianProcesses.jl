"""
Student-t likelihood : ``Γ((ν+1)/2)/(√(νπ)Γ(ν/2)) (1+t²/ν)^(-(ν+1)/2)``
"""
abstract type AbstractStudentTLikelikelihood{T<:Real} <: RegressionLikelihood{T} end

function pdf(l::AbstractStudentTLikelikelihood,y::Real,f::Real)
    tdistpdf(l.ν,y-f)
end

function Base.show(io::IO,model::AbstractStudentTLikelikelihood{T}) where T
    print(io,"Student-t likelihood")
end

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractArray{T,N},likelihood::L) where {T,N,L<:AbstractStudentTLikelikelihood}
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
###############################################################################

struct AugmentedStudentTLikelihood{T<:Real} <:AbstractStudentTLikelikelihood{T}
    ν::T
    α::T
    β::Vector{T}
    θ::Vector{T}
    function AugmentedStudentTLikelihood{T}(ν::T) where {T<:Real}
        new{T}(ν)
    end
    function AugmentedStudentTLikelihood{T}(ν::T,β::AbstractVector{T},θ::AbstractVector{T}) where {T<:Real}
        new{T}(ν,(ν+one(T))/2.0,β,θ)
    end
end

isaugmented(::AugmentedStudentTLikelihood{T}) where T = true

function AugmentedStudentTLikelihood(ν::T) where {T<:Real}
    AugmentedStudentTLikelihood{T}(ν)
end

function init_likelihood(likelihood::AugmentedStudentTLikelihood{T},nLatent::Int,nSamplesUsed::Int) where T
    AugmentedStudentTLikelihood{T}(likelihood.ν,abs2(T.(rand(T,nSamplesUsed))),zeros(T,nSamplesUsed))
end

function local_updates!(model::VGP{<:AugmentedStudentTLikelihood,<:AnalyticInference})
    model.likelihood.β .= 0.5*(diag(model.Σ)+abs2.(model.μ.-model.y).+model.likelihood.ν)
    model.likelihood.θ .= 0.5*(model.likelihood.ν+1.0)./model.likelihood.β
end

function local_updates!(model::SVGP{<:AugmentedStudentTLikelihood,<:AnalyticInference})
    model.β .= 0.5*(model.K̃ + opt_diag(model.κ*model.Σ,model.κ) + abs2.(model.κ*model.μ-model.y[model.MBIndices]) .+model.likelihood.ν)
    model.θ .= 0.5*(model.likelihood.ν+1.0)./model.likelihood.β
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{<:AugmentedStudentTLikelihood},index::Integer)
    return model.likelihood.θ*model.y[index]
end

function expec_μ(model::VGP{<:AugmentedStudentTLikelihood})
    return model.likelihood.θ*model.y
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:AugmentedStudentTLikelihood},index::Integer)
    return 0.5.*model.y[index][model.inference.MBIndices]
end

function expec_μ(model::SVGP{<:AugmentedStudentTLikelihood})
    return 0.5.*getindex.(model.y,[model.inference.MBIndices])
end

function expec_Σ(model::GP{<:AugmentedStudentTLikelihood},index::Integer)
    return 0.5*model.likelihood.θ[index]
end

function expec_Σ(model::GP{<:AugmentedStudentTLikelihood})
    return 0.5*model.likelihood.θ
end

function compute_proba(l::AugmentedStudentTLikelihood,μ::AbstractVector{AbstractVector},σ²::AbstractVector{AbstractVector})
    K = length(μ)
    N = length(μ[1])
    pred = [zeros(N) for _ in 1:K]
    for k in 1:model.K
        for i in 1:N
            if σ²[k][i] <= 0.0
                pred[k][i] = logit(μ[k][i])
            else
                pred[k][i] = expectation(logit,Normal(μ[k][i],sqrt(σ²[k][i])))
            end
        end
    end
    return pred
end

function ELBO(model::GP{<:AugmentedStudentTLikelihood})
    return expecLogLikelihood(model) - InverseGammaKL(model)
end

function expecLogLikelihood(model::VGP{AugmentedStudentTLikelihood{T}}) where T
    return -Inf #TODO
end

function expecLogLikelihood(model::SVGP{AugmentedStudentTLikelihood{T}}) where T
    return -Inf #TODO
end
