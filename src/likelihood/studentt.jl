"""
Student-t likelihood : ``\\Gamma((\\nu+1)/2)/(\\sqrt{\\nu\\pi}\\Gamma(\\nu/2)) (1+t\\^2/\\nu)^(-(\\nu+1)/2)``
"""
abstract type AbstractStudentTLikelihood{T<:Real} <: RegressionLikelihood{T} end

function pdf(l::AbstractStudentTLikelihood,y::Real,f::Real)
    tdistpdf(l.ν,y-f)
end

function Base.show(io::IO,model::AbstractStudentTLikelihood{T}) where T
    print(io,"Student-t likelihood")
end


function compute_proba(l::AbstractStudentTLikelihood{T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
    N = length(μ)
    st = TDist(l.ν)
    nSamples = 2000
    μ_pred = zeros(T,N)
    σ²_pred = zeros(T,N)
    temp_array = zeros(T,nSamples)
    for i in 1:N
        # e = expectation(Normal(μ[i],sqrt(σ²[i])))
        # μ_pred[i] = μ[i]
        #
        # σ²_pred[i] = e(x->pdf(LocationScale(x,1.0,st))^2) - e(x->pdf(LocationScale(x,1.0,st)))^2
        if σ²[i] <= 1e-3
            pyf =  LocationScale(μ[i],1.0,st)
            for j in 1:nSamples
                temp_array[j] = rand(pyf)
            end
        else
            d = Normal(μ[i],sqrt(σ²[i]))
            for j in 1:nSamples
                temp_array[j] = rand(LocationScale(rand(d),1.0,st))
            end
        end
        μ_pred[i] = μ[i];
        σ²_pred[i] = cov(temp_array)
    end
    return μ_pred,σ²_pred
end

###############################################################################
"""
Student-t likelihood : ``\\frac{\\Gamma((\\nu+1)/2)}{\\sqrt{\\nu\\pi}\\Gamma(\\nu/2)}\\left(1+t^2/\\nu\\right)^{(-(\\nu+1)/2)}``

"""
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

function local_updates!(model::VGP{<:AugmentedStudentTLikelihood,<:AnalyticVI})
    model.likelihood.β .= broadcast((Σ,μ,y)->0.5*(Σ+abs2.(μ-y).+model.likelihood.ν),diag.(model.Σ),model.μ,model.y)
    model.likelihood.θ .= broadcast(β->0.5*(model.likelihood.ν+1.0)./β,model.likelihood.β)
end

function local_updates!(model::SVGP{<:AugmentedStudentTLikelihood,<:AnalyticVI})
    model.likelihood.β .= broadcast((K̃,κ,Σ,μ,y)->0.5*(K̃ + opt_diag(κ*Σ,κ) + abs2.(κ*μ-y[model.inference.MBIndices]) .+model.likelihood.ν),model.K̃,model.κ,model.Σ,model.μ,model.y)
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
    return hadamard.(model.likelihood.θ,getindex.(model.y,[model.inference.MBIndices]))
end

function expec_Σ(model::AbstractGP{<:AugmentedStudentTLikelihood},index::Integer)
    return 0.5*model.likelihood.θ[index]
end

function ∇Σ(model::AbstractGP{<:AugmentedStudentTLikelihood})
    return 0.5*model.likelihood.θ
end

function ELBO(model::AbstractGP{<:AugmentedStudentTLikelihood})
    return expecLogLikelihood(model) - InverseGammaKL(model) - GaussianKL(model)
end

function expecLogLikelihood(model::VGP{AugmentedStudentTLikelihood{T}}) where T
    tot = -0.5*model.nLatent*model.nSample*log(twoπ)
    tot -= 0.5.*sum(broadcast(β->sum(log.(β).-model.nSample*digamma(model.likelihood.α)),model.likelihood.β))
    tot -= 0.5.*sum(broadcast((β,Σ,μ,y)->dot(model.likelihood.α./β,Σ+abs2.(μ)-2.0*μ.*y-abs2.(y)),model.likelihood.β,diag.(model.Σ),model.μ,model.y))
    return tot
end

function expecLogLikelihood(model::SVGP{AugmentedStudentTLikelihood{T}}) where T
    tot = -0.5*model.nLatent*model.inference.nSamplesUsed*log(twoπ)
    tot -= 0.5.*sum(broadcast(β->sum(log.(β).-model.inference.nSamplesUsed*digamma(model.likelihood.α)),model.likelihood.β))
    tot -= 0.5.*sum(broadcast((β,K̃,κ,Σ,μ,y)->dot(model.likelihood.α./β,(K̃+opt_diag(κ*Σ,κ)+abs2.(κ*μ)-2.0*(κ*μ).*y[model.inference.MBIndices]-abs2.(y[model.inference.MBIndices]))),model.likelihood.β,model.K̃,model.κ,model.Σ,model.μ,model.y))
    return model.inference.ρ*tot
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
