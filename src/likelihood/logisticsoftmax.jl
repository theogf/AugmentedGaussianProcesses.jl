"""
Softmax likelihood : ``p(y=i|{fₖ}) = exp(fᵢ)/ ∑ exp(fₖ) ``
"""
struct AugmentedLogisticSoftMaxLikelihood{T<:Real} <: AbstractLogisticSoftMaxLikelihood{T}
    Y::Abstract{Vector{SparseVector{Int64}}} #Mapping from instances to classes
    class_mapping::AbstractVector{Any} # Classes labels mapping
    ind_mapping::Dict{Any,Int} # Mapping from label to index
    c::AbstractVector{AbstractVector{T}} # Second moment of fₖ
    α::AbstractVector{T} # Variational parameter of Gamma distribution
    β::AbstractVector{T} # Variational parameter of Gamma distribution
    θ::AbstractVector{AbstractVector{T}} # Variational parameter of Polya-Gamma distribution
    γ::AbstractVector{AbstractVector{T}} # Variational parameter of Poisson distribution
    function AugmentedLogisticSoftMaxLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function AugmentedLogisticSoftMaxLikelihood{T}(Y::Abstract{Vector{SparseVector{<:Integer}}},
    class_mapping::AbstractVector{Any}, ind_mapping::Dict{Any,Int})
        new{T}(Y,class_mapping,ind_mapping)
    end
    function AugmentedLogisticSoftMaxLikelihood{T}(Y::Abstract{Vector{SparseVector{<:Integer}}},
    class_mapping::AbstractVector{Any}, ind_mapping::Dict{Any,Int},
    c::AbstractVector{AbstractVector{T}}, α::AbstractVector{T},
    β::AbstractVector{T}, θ::AbstractVector{AbstractVector{T}},γ::AbstractVector{AbstractVector{T}})
        new{T}(Y,class_mapping,ind_mapping,c,α,β,θ,γ)
    end
end

function AugmentedLogisticSoftMaxLikelihood()
    SoftMaxLikelihood{Float64}()
end

function init_likelihood(likelihood::AugmentedLogisticSoftMaxLikelihood{T},nLatent::Integer,nSamplesUsed::Integer) where T
    c = [ones(T,nSamplesUsed) for i in 1:nLatent]
    α = nLatent*ones(T,nSamplesUsed)
    β = nLatent*ones(T,nSamplesUsed)
    θ = [abs.(rand(T,nSamplesUsed))*2 for i in 1:nLatent]
    γ = [abs.(rand(T,nSamplesUsed)) for i in 1:nLatent]
    SoftMaxLikelihood{T}(likelihood.Y,likelihood.class_mapping,likelihood.ind_mapping,c,α,β,θ,γ)
end

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractArray{T,N},likelihood::L) where {T,N,L<:SoftMaxLikelihood}
    @assert N <= 1 "Target should be a vector of values"
    likelihood = init_multiclass_likelihood(y,likelihood)

end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{<:AugmentedLogisticSoftMaxLikelihood},index::Integer)
end

function expec_μ(model::VGP{<:AugmentedLogisticSoftMaxLikelihood})
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:AugmentedLogisticSoftMaxLikelihood},index::Integer)
end

function expec_μ(model::SVGP{<:AugmentedLogisticSoftMaxLikelihood})
end

function expec_Σ(model::GP{<:AugmentedLogisticSoftMaxLikelihood},index::Integer)
end

function expec_Σ(model::GP{<:AugmentedLogisticSoftMaxLikelihood})
end

function ELBO(model::GP{<:AugmentedLogisticSoftMaxLikelihood})
    return expecLogLikelihood(model) - GaussianKL(model) - PolyaGammaKL(model)
end

function expecLogLikelihood(model::VGP{<:AugmentedLogisticSoftMaxLikelihood})
    tot = -model.nLatent*(0.5*model.nSamples*log(2))
    tot += sum(broadcast((μ,y,θ,Σ)->0.5.*(sum(μ.*y)-opt_trace(θ,(diag(Σ)+μ.^2))),
                        model.μ,model.y,model.θ,model.Σ))
    return tot
end

function expecLogLikelihood(model::SVGP{<:AugmentedLogisticSoftMaxLikelihood})
    tot = -model.nLatent*(0.5*model.nSamples*log(2))
    tot += sum(broadcast((κμ,y,θ,κΣκ,K̃)->0.5.*(sum(κμ.*y)-opt_trace(θ,K̃+κΣκ+κμ.^2))),
                        model.κ.*model.μ,model.y,model.θ,opt_diag(model.κ*model.Σ,model.κ'),model.K̃)
    return model.inference.ρ*tot
end

function treat_samples(model::GP{<:LogisticSoftMaxMultiClassLikelihood},samples::AbstractMatrix,index::Integer)
    class = model.likelihood.ind_mapping[model.y[index]]
    grad_μ = zeros(model.nLatent)
    grad_Σ = zeros(model.nLatent)
    for i in 1:size(samples,1)
        σ = logistic(samples[i,:])
        samples[i,:]  .= logisticsoftmax(samples[i,:])
        s = samples[i,class]
        g_μ = grad_logisticsoftmax(samples[i,:],σ,class)
        grad_μ .+= g_μ./s
        grad_Σ .+= hessian_logisticsoftmax(samples[i,:],σ,class)./s .- g_μ.^2 ./s^2
    end
    for k in 1:model.nLatent
        model.∇μE[k][index] = grad_μ[k]/nSamples
        model.∇ΣE[k][index] = 0.5.*grad_Σ[k]/nSamples
    end
end
