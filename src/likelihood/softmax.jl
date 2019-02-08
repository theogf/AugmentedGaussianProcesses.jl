"""
Softmax likelihood : ``p(y=i|{fₖ}) = exp(fᵢ)/ ∑ exp(fₖ) ``
"""
struct SoftMaxLikelihood{T<:Real} <: MultiClassLikelihood{T}
    Y::AbstractVector{SparseVector{Int64}} #Mapping from instances to classes
    class_mapping::AbstractVector{Any} # Classes labels mapping
    ind_mapping::Dict{Any,Int} # Mapping from label to index
    y_class::AbstractVector{Int64} #GP Index for each sample
    function SoftMaxLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function SoftMaxLikelihood{T}(Y::AbstractVector{SparseVector{<:Integer}},
    class_mapping::AbstractVector{Any}, ind_mapping::Dict{Any,Int},y_class::AbstractVector{Int}) where {T<:Real}
        new{T}(Y,class_mapping,ind_mapping,y_class)
    end
end

function SoftMaxLikelihood()
    SoftMaxLikelihood{Float64}()
end

function pdf(l::SoftMaxLikelihood,f::AbstractVector)
    softmax(f)
end

function init_likelihood(likelihood::SoftMaxLikelihood{T},nLatent::Integer,nSamplesUsed::Integer) where T
    return likelihood
end


""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{<:SoftMaxLikelihood},index::Integer)
end

function expec_μ(model::VGP{<:SoftMaxLikelihood})
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:SoftMaxLikelihood},index::Integer)
end

function expec_μ(model::SVGP{<:SoftMaxLikelihood})
end

function expec_Σ(model::GP{<:SoftMaxLikelihood},index::Integer)
end

function expec_Σ(model::GP{<:SoftMaxLikelihood})
end

function ELBO(model::GP{<:SoftMaxLikelihood})
    return expecLogLikelihood(model) - GaussianKL(model) - PolyaGammaKL(model)
end

function expecLogLikelihood(model::VGP{SoftMaxLikelihood{T}}) where T
    tot = -model.nLatent*(0.5*model.nSamples*log(2))
    tot += sum(broadcast((μ,y,θ,Σ)->0.5.*(sum(μ.*y)-opt_trace(θ,(diag(Σ)+μ.^2))),
                        model.μ,model.y,model.θ,model.Σ))
    return tot
end

function expecLogLikelihood(model::SVGP{SoftMaxLikelihood{T}}) where T
    tot = -model.nLatent*(0.5*model.nSamples*log(2))
    tot += sum(broadcast((κμ,y,θ,κΣκ,K̃)->0.5.*(sum(κμ.*y)-opt_trace(θ,K̃+κΣκ+κμ.^2))),
                        model.κ.*model.μ,model.y,model.θ,opt_diag(model.κ*model.Σ,model.κ'),model.K̃)
    return model.inference.ρ*tot
end

function treat_samples(model::GP{<:SoftMaxLikelihood},samples::AbstractMatrix,index::Integer)
    class = model.likelihood.ind_mapping[model.y[model.inference.MBIndices[index]]]
    grad_μ = zeros(model.nLatent)
    grad_Σ = zeros(model.nLatent)
    samples .= mapslices(softmax,samples,dims=2)
    for i in 1:size(samples,1)
        s = samples[i,class]
        g_μ = grad_softmax(samples[i,:],σ,class)
        grad_μ .+= g_μ./s
        grad_Σ .+= diaghessian_softmax(samples[i,:],σ,class)./s .- g_μ.^2 ./s^2
    end
    for k in 1:model.nLatent
        model.inference.∇μE[k][index] = grad_μ[k]/nSamples
        model.inference.∇ΣE[k][index] = 0.5.*grad_Σ[k]/nSamples
    end
end


function softmax(f::AbstractVector{<:Real})
    s = exp.(f)
    return s./sum(s)
end

function softmax(f::AbstractVector{<:Real},i::Integer)
    return softmax(f)[i]
end

function grad_softmax(s::AbstractVector{<:Real},i::Integer)
    base_grad = -s.*s[i]
    base_grad[i] += s[i]
    return base_grad
end

function diaghessian_softmax(s::AbstractVector{<:Real},i::Integer)
    m = length(s)
    hessian = zeros(m)
    for j in 1:m
            hessian[j] = s[i]*((δ(i,j)-s[j])^2-s[j]*(1.0-s[j]))
    end
    return hessian
end

function hessian_softmax(s::AbstractVector{<:Real},i::Integer)
    m = length(s)
    hessian = zeros(m,m)
    for j in 1:m
        for k in 1:m
            hessian[j,k] = s[i]*((δ(i,k)-s[k])*(δ(i,j)-s[j])-s[j]*(δ(j,k)-s[k]))
        end
    end
    return hessian
end
