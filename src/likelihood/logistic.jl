"""
Logistic likelihood : ``p(y|f) = σ(yf) = (1+exp(-yf))⁻¹ ``
"""
#TODO Separate into a numerical and analytic version
struct LogisticLikelihood{T<:Real} <: ClassificationLikelihood{T}
    c::AbstractVector{AbstractVector{T}}
    θ::AbstractVector{AbstractVector{T}}
    function LogisticLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function LogisticLikelihood{T}(c::AbstractVector{AbstractVector{<:Real}},θ::AbstractVector{AbstractVector{<:Real}}) where {T<:Real}
        new{T}(c,θ)
    end
end

function LogisticLikelihood()
    LogisticLikelihood{Float64}()
end

function pdf(l::LogisticLikelihood,y::Real,f::Real)
    logit(y*f)
end

function Base.show(io::IO,model::LogisticLikelihood{T}) where T
    print(io,"Bernoulli logistic likelihood")
end


function init_likelihood(likelihood::LogisticLikelihood{T},nLatent::Integer,nSamplesUsed) where T
    LogisticLikelihood{T}([abs.(rand(T,nSamplesUsed)) for _ in 1:nLatent],[zeros(T,nSamplesUsed) for _ in 1:nLatent])
end

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractArray{T,N},likelihood::L) where {T,N,L<:LogisticLikelihood}
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

function local_updates!(model::VGP{<:LogisticLikelihood,<:AnalyticInference})
    model.likelihood.c .= broadcast((μ,Σ)->sqrt.(diag(Σ)+μ.^2),model.μ,model.Σ)
    model.likelihood.θ .= broadcast(c->0.5*tanh.(0.5*c)./c,model.c)
end

function local_updates!(model::SVGP{<:LogisticLikelihood,<:AnalyticInference})
    model.likelihood.c .= broadcast((μ,Σ,K̃,κ)->sqrt.(K̃+opt_diag(κ*Σ,κ')+(κ*μ).^2),model.μ,model.Σ,model.K̃,model.κ)
    model.likelihood.θ .= broadcast(c->0.5*tanh.(0.5*c)./c,model.c)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{<:LogisticLikelihood},index::Integer)
    return 0.5.*model.y[index]
end

function expec_μ(model::VGP{<:LogisticLikelihood})
    return 0.5.*model.y
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:LogisticLikelihood},index::Integer)
    return 0.5.*model.y[index][model.inference.MBIndices]
end

function expec_μ(model::SVGP{<:LogisticLikelihood})
    return 0.5.*getindex.(model.y,[model.inference.MBIndices])
end

function expec_Σ(model::GP{<:LogisticLikelihood},index::Integer)
    return 0.5*model.likelihood.θ[index]
end

function expec_Σ(model::GP{<:LogisticLikelihood})
    return 0.5*model.likelihood.θ
end

function compute_proba(l::LogisticLikelihood,μ::AbstractVector{AbstractVector},σ²::AbstractVector{AbstractVector})
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

function ELBO(model::GP{<:LogisticLikelihood})
    return expecLogLikelihood(model) - GaussianKL(model) - PolyaGammaKL(model)
end

function expecLogLikelihood(model::VGP{LogisticLikelihood{T}}) where T
    tot = -model.nLatent*(0.5*model.nSamples*log(2))
    tot += sum(broadcast((μ,y,θ,Σ)->0.5.*(sum(μ.*y)-opt_trace(θ,(diag(Σ)+μ.^2))),
                        model.μ,model.y,model.θ,model.Σ))
    return tot
end

function expecLogLikelihood(model::SVGP{LogisticLikelihood{T}}) where T
    tot = -model.nLatent*(0.5*model.nSamples*log(2))
    tot += sum(broadcast((κμ,y,θ,κΣκ,K̃)->0.5.*(sum(κμ.*y)-opt_trace(θ,K̃+κΣκ+κμ.^2))),
                        model.κ.*model.μ,model.y,model.θ,opt_diag(model.κ*model.Σ,model.κ'),model.K̃)
    return model.inference.ρ*tot
end
