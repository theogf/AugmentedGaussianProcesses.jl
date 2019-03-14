"""
Student-t likelihood : ``Γ((ν+1)/2)/(√(νπ)Γ(ν/2)) (1+t²/ν)^(-(ν+1)/2)``
"""
abstract type AbstractStudentTLikelihood{T<:Real} <: RegressionLikelihood{T} end

function pdf(l::AbstractStudentTLikelihood,y::Real,f::Real)
    tdistpdf(l.ν,y-f)
end

function Base.show(io::IO,model::AbstractStudentTLikelihood{T}) where T
    print(io,"Student-t likelihood")
end


function compute_proba(l::AbstractStudentTLikelihood,μ::AbstractVector{AbstractVector},σ²::AbstractVector{AbstractVector})
    K = length(μ)
    N = length(μ[1])
    @error "Not implemented for StudentT likelihood yet"
    return pred
end

###############################################################################

struct AugmentedStudentTLikelihood{T<:Real} <:AbstractStudentTLikelihood{T}
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
    AugmentedStudentTLikelihood{T}(likelihood.ν,abs2.(T.(rand(T,nSamplesUsed))),zeros(T,nSamplesUsed))
end

function local_updates!(model::VGP{<:AugmentedStudentTLikelihood,<:AnalyticInference})
    model.likelihood.β .= 0.5*(diag(model.Σ)+abs2.(model.μ.-model.y).+model.likelihood.ν)
    model.likelihood.θ .= 0.5*(model.likelihood.ν+1.0)./model.likelihood.β
end

function local_updates!(model::SVGP{<:AugmentedStudentTLikelihood,<:AnalyticInference})
    model.likelihood.β .= 0.5*(model.K̃ + opt_diag(model.κ*model.Σ,model.κ) + abs2.(model.κ*model.μ-model.y[model.likelihood.MBIndices]) .+model.likelihood.ν)
    model.likelihood.θ .= 0.5*(model.likelihood.ν+1.0)./model.likelihood.β
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

function ELBO(model::GP{<:AugmentedStudentTLikelihood})
    return expecLogLikelihood(model) - InverseGammaKL(model)
end

function expecLogLikelihood(model::VGP{AugmentedStudentTLikelihood{T}}) where T
    return -Inf #TODO
end

function expecLogLikelihood(model::SVGP{AugmentedStudentTLikelihood{T}}) where T
    return -Inf #TODO
end


##########################################

struct StudentTLikelihood{T<:Real} <:AbstractStudentTLikelihood{T}
    ν::T
    α::T
    function StudentTLikelihood{T}(ν::T) where {T<:Real}
        new{T}(ν,(ν+one(T))/2.0)
    end
end

function StudentTLikelihood(ν::T) where {T<:Real}
    StudentTLikelihood{T}(ν)
end

function gradpdf(::StudentTLikelihood,y::Int,f::T) where {T<:Real}
    @error "Not implemented yet"
end

function hessiandiagpdf(::StudentTLikelihood,y::Int,f::T) where {T<:Real}
    @error "Not implemented yet"
end
