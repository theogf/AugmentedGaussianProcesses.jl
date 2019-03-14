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
    β::LatentArray{Vector{T}}
    θ::LatentArray{Vector{T}}
    function AugmentedStudentTLikelihood{T}(ν::T) where {T<:Real}
        new{T}(ν,(ν+one(T))/2.0)
    end
    function AugmentedStudentTLikelihood{T}(ν::T,β::AbstractVector{<:AbstractVector{T}},θ::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
        new{T}(ν,(ν+one(T))/2.0,β,θ)
    end
end

isaugmented(::AugmentedStudentTLikelihood{T}) where T = true

function AugmentedStudentTLikelihood(ν::T) where {T<:Real}
    AugmentedStudentTLikelihood{T}(ν)
end

function init_likelihood(likelihood::AugmentedStudentTLikelihood{T},nLatent::Int,nSamplesUsed::Int) where T
    AugmentedStudentTLikelihood{T}(likelihood.ν,[abs2.(T.(rand(T,nSamplesUsed))) for _ in 1:nLatent],[zeros(T,nSamplesUsed) for _ in 1:nLatent])
end

function local_updates!(model::VGP{<:AugmentedStudentTLikelihood,<:AnalyticInference})
    model.likelihood.β .= broadcast((Σ,μ,y)->0.5*(Σ+abs2.(μ-y).+model.likelihood.ν),diag.(model.Σ),model.μ,model.y)
    model.likelihood.θ .= broadcast(β->0.5*(model.likelihood.ν+1.0)./β,model.likelihood.β)
end

function local_updates!(model::SVGP{<:AugmentedStudentTLikelihood,<:AnalyticInference})
    model.likelihood.β .= broadcast((K̃,κ,Σ,μ,y)->0.5*(K̃ + opt_diag(κ*Σ,κ) + abs2.(κ*μ-y[model.likelihood.MBIndices]) .+model.likelihood.ν),model.K̃,model.κ,model.Σ,model.μ,model.y)
    model.likelihood.θ .= broadcast(β->0.5*(model.likelihood.ν+1.0)./β,model.likelihood.β)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{<:AugmentedStudentTLikelihood},index::Integer)
    return model.likelihood.θ[index].*model.y[index]
end

function ∇μ(model::VGP{<:AugmentedStudentTLikelihood})
    return hadamard.(model.likelihood.θ,model.y)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:AugmentedStudentTLikelihood},index::Integer)
    return model.likelihood.θ[index].*model.y[index][model.inference.MBIndices]
end

function ∇μ(model::SVGP{<:AugmentedStudentTLikelihood})
    return hadamard.(model.likelihood.θ,model.y[model.inference.MBIndices])
end

function expec_Σ(model::GP{<:AugmentedStudentTLikelihood},index::Integer)
    return 0.5*model.likelihood.θ[index]
end

function ∇Σ(model::GP{<:AugmentedStudentTLikelihood})
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
