"""
**MultinomialLogistic**

Multiclass likelihood with Multinomial logistic link : ``p(y=i|{fₖ}) = \\prod_{i\\neq k} \\Theta(f_i-f_k) ``
"""
struct MultinomialLogisticLikelihood{T<:Real} <: MultinomialLogisticLikelihood{T}
    Y::AbstractVector{BitVector} #Mapping from instances to classes
    class_mapping::AbstractVector{Any} # Classes labels mapping
    ind_mapping::Dict{Any,Int} # Mapping from label to index
    y_class::AbstractVector{Int64} # GP Index for each sample
    c::AbstractVector{AbstractVector{T}} # Second moment of fₖ
    θ::AbstractVector{AbstractVector{T}} # Variational parameter of Polya-Gamma distribution
    function MultinomialLogisticLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function MultinomialLogisticLikelihood{T}(Y::AbstractVector{<:BitVector},
    class_mapping::AbstractVector, ind_mapping::Dict{<:Any,<:Int},y_class::AbstractVector{<:Int}) where {T<:Real}
        new{T}(Y,class_mapping,ind_mapping,y_class)
    end
    function MultinomialLogisticLikelihood{T}(Y::AbstractVector{<:BitVector},
    class_mapping::AbstractVector, ind_mapping::Dict{<:Any,<:Int},y_class::AbstractVector{<:Int},
    c::AbstractVector{<:AbstractVector{<:Real}}, θ::AbstractVector{<:AbstractVector}) where {T<:Real}
        new{T}(Y,class_mapping,ind_mapping,y_class,c,θ)
    end
end

function MultinomialLogisticLikelihood()
    MultinomialLogisticLikelihood{Float64}()
end

function pdf(l::MultinomialLogisticLikelihood,f::AbstractVector)
    multinomiallogistic(f)
end

function pdf(l::MultinomialLogisticLikelihood,y::Integer,f::AbstractVector)
    multinomiallogistic(f)[y]
end

function multinomiallogistic(f::AbstractVector)
    y = zero(f)
    for (i,x) in enumerate(f)
        y[i]=prod(logistic.(x.-f[1:length(f).!=i]))
    end
    return y
end

function multinomialheaviside(f::AbstractVector)
    y = zero(f)
    for (i,x) in enumerate(f)
        y[i]=prod(x.>f[1:length(f).!=i])
    end
    return y
end

function multinomialprobit(f::AbstractVector)
    y = zero(f)
    for (i,x) in enumerate(f)
        y[i]=prod(cdf.(Normal(0,1),x.-f[1:length(f).!=i]))
    end
    return y
end

function Base.show(io::IO,model::AugmentedMultinomialLogisticLikelihood{T}) where T
    print(io,"Augmented Multinomial Logistic likelihood")
end


function init_likelihood(likelihood::AugmentedMultinomialLogisticLikelihood{T},nLatent::Integer,nSamplesUsed::Integer) where T
    c = [ones(T,nSamplesUsed) for i in 1:nLatent]
    θ = [abs.(rand(T,nSamplesUsed))*2 for i in 1:nLatent]
    AugmentedMultinomialLogisticLikelihood{T}(likelihood.Y,likelihood.class_mapping,likelihood.ind_mapping,likelihood.y_class,c,θ)
end

struct MultinomialLogisticLikelihood{T<:Real} <: AbstractMultinomialLogisticLikelihood{T}
    Y::AbstractVector{BitVector} #Mapping from instances to classes
    class_mapping::AbstractVector{Any} # Classes labels mapping
    ind_mapping::Dict{Any,Int} # Mapping from label to index
    y_class::AbstractVector{Int64} #GP Index for each sample
    function MultinomialLogisticLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function MultinomialLogisticLikelihood{T}(Y::AbstractVector{<:BitVector},
    class_mapping::AbstractVector, ind_mapping::Dict{<:Any,<:Int},y_class::AbstractVector{<:Int}) where {T<:Real}
        new{T}(Y,class_mapping,ind_mapping,y_class)
    end
end

function MultinomialLogisticLikelihood()
    MultinomialLogisticLikelihood{Float64}()
end

function Base.show(io::IO,model::MultinomialLogisticLikelihood{T}) where T
    print(io,"logistic-softmax likelihood")
end

function init_likelihood(likelihood::MultinomialLogisticLikelihood{T},nLatent::Integer,nSamplesUsed::Integer) where T
    return likelihood
end

function local_updates!(model::VGP{AugmentedMultinomialLogisticLikelihood{T},AnalyticVI{T},T,V}) where {T<:Real,V<:AbstractVector{T}}
    model.likelihood.c .= broadcast((Σ::V,μ::V)->Σ.+abs2.(μ),diag.(model.Σ),model.μ)
    c = (x->sqrt.(x)).(model.likelihood.Y.*model.likelihood.c .+ model.likelihood.c)
    model.likelihood.θ .= broadcast((c::V)->1.0./c.*tanh.(0.5*c),model.likelihood.c)
end

function natural_gradient!(model::VGP{AugmentedMultinomialLogisticLikelihood{T},AnalyticVI{T},T,V}) where {T<:Real,V<:AbstractVector{T}}
    up_y = 0.5*((model.nLatent-1).+2*dot(!y,μ.*θ))
    up_f = 0.5*(1+2*μ[model.likelihood.class_y[i]]*θ[i])
    upsig_y = -0.5*(Diagonal([dot(!getindexmodel(model.likelihood.Y,[i]),getindexmodel(model.likelihood.θ,[i])) for i in 1:model.nSamples]) + invKmm) - model.η₂
    upsig_g = θ + invKmm
end


function local_updates!(model::SVGP{AugmentedMultinomialLogisticLikelihood{T},AnalyticVI{T},T,V}) where {T<:Real,V<:AbstractVector{T}}
    model.likelihood.c .= broadcast((μ::V,Σ::Symmetric{T,Matrix{T}},κ::Matrix{T},K̃::V)->sqrt.(K̃+opt_diag(κ*Σ,κ)+abs2.(κ*μ)),
                                    model.μ,model.Σ,model.κ,model.K̃)
    for _ in 1:5
        model.likelihood.γ .= broadcast((c::V,κμ::V,ψα::V)->(0.5/(model.likelihood.β[1]))*exp.(ψα).*safe_expcosh.(-0.5*κμ,0.5*c),
                                    model.likelihood.c,model.κ.*model.μ,[digamma.(model.likelihood.α)])
        model.likelihood.α .= 1.0.+(model.likelihood.γ...,)
    end
    model.likelihood.θ .= broadcast((y::BitVector,γ::V,c::V)->0.5.*(y[model.inference.MBIndices]+γ)./c.*tanh.(0.5.*c),
                                    model.likelihood.Y,model.likelihood.γ,model.likelihood.c)
    return nothing
end

function sample_local!(model::VGP{<:AugmentedMultinomialLogisticLikelihood,<:GibbsSampling})
    if model.inference.nIter <= 1
        # model.likelihood.α .= 10.0.*model.likelihood.α./model.likelihood.β
    end
    model.likelihood.γ .= broadcast(μ::AbstractVector{<:Real}->rand.(Poisson.(0.5*model.likelihood.α.*safe_expcosh.(-0.5*μ,0.5*μ))), model.μ)
    model.likelihood.α .= rand.(Gamma.(1.0.+(model.likelihood.γ...,),1.0./model.likelihood.β))
    model.likelihood.θ .= broadcast((y::BitVector,γ::AbstractVector{<:Real},μ::AbstractVector{<:Real})->PolyaGammaDist().draw.(y.+γ,μ),model.likelihood.Y,model.likelihood.γ,model.μ)
    return nothing
end

function sample_global!(model::VGP{<:AugmentedMultinomialLogisticLikelihood,<:GibbsSampling})
    model.Σ .= inv.(Symmetric.(Diagonal.(model.likelihood.θ).+model.invKnn))
    model.μ .= rand.(MvNormal.(0.5.*model.Σ.*(model.likelihood.Y.-model.likelihood.γ),model.Σ))
    return nothing
end


""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::VGP{<:AugmentedMultinomialLogisticLikelihood},index::Int)
    0.5.*(model.likelihood.Y[index]-model.likelihood.γ[index])
end

function ∇μ(model::VGP{<:AugmentedMultinomialLogisticLikelihood})
    0.5.*(model.likelihood.Y.-model.likelihood.γ)
end

""" Return the gradient of the expectation for latent GP `index` """
function expec_μ(model::SVGP{<:AugmentedMultinomialLogisticLikelihood},index::Integer)
    0.5.*(model.likelihood.Y[index][model.inference.MBIndices]-model.likelihood.γ[index])
end

function ∇μ(model::SVGP{<:AugmentedMultinomialLogisticLikelihood})
    0.5.*(getindex.(model.likelihood.Y,[model.inference.MBIndices]).-model.likelihood.γ)
end

function expec_Σ(model::AbstractGP{<:AugmentedMultinomialLogisticLikelihood},index::Integer)
    0.5.*model.likelihood.θ[index]
end

function ∇Σ(model::AbstractGP{AugmentedMultinomialLogisticLikelihood{T}}) where {T<:Real}
    Diagonal{T}.(0.5.*model.likelihood.θ)
end

function ELBO(model::AbstractGP{<:AugmentedMultinomialLogisticLikelihood})
    return expecLogLikelihood(model) - GaussianKL(model) - GammaImproperKL(model) - PoissonKL(model) - PolyaGammaKL(model)
end

function expecLogLikelihood(model::VGP{<:AugmentedMultinomialLogisticLikelihood})
    model.likelihood.c .= broadcast((Σ,μ)->sqrt.(Σ.+abs2.(μ)),diag.(model.Σ),model.μ)
    tot = -model.nSample*logtwo
    tot += -sum(sum.(model.likelihood.γ))*logtwo
    tot +=  0.5*sum(broadcast((y,μ,γ,θ,c)->sum(μ.*(y-γ)-θ.*abs2.(c)),
                    model.likelihood.Y,model.μ,model.likelihood.γ,model.likelihood.θ,model.likelihood.c))
    return tot
end

function expecLogLikelihood(model::SVGP{<:AugmentedMultinomialLogisticLikelihood})
    model.likelihood.c .= broadcast((μ::AbstractVector,Σ::AbstractMatrix,κ::AbstractMatrix,K̃::AbstractVector)->sqrt.(K̃+opt_diag(κ*Σ,κ)+abs2.(κ*μ)),
                                    model.μ,model.Σ,model.κ,model.K̃)
    tot = -model.inference.nSamplesUsed*logtwo
    tot += -sum(sum.(model.likelihood.γ))*logtwo
    tot += 0.5*sum(broadcast((y,κμ,γ,θ,c)->sum((κμ).*(y[model.inference.MBIndices]-γ)-θ.*abs2.(c)),
                    model.likelihood.Y,model.κ.*model.μ,model.likelihood.γ,model.likelihood.θ,model.likelihood.c))
    return model.inference.ρ*tot
end

function grad_samples(model::AbstractGP{<:MultinomialLogisticLikelihood,<:NumericalVI,T},samples::AbstractMatrix{T},index::Integer) where {T<:Real}
    class = model.likelihood.y_class[index]::Int64
    grad_μ = zeros(T,model.nLatent)
    grad_Σ = zeros(T,model.nLatent)
    g_μ = similar(grad_μ)
    nSamples = size(samples,1)
    @views @inbounds for i in 1:nSamples
        σ = logistic.(samples[i,:])
        samples[i,:]  .= MultinomialLogistic(samples[i,:])
        s = samples[i,class]
        g_μ .= grad_MultinomialLogistic(samples[i,:],σ,class)/s
        grad_μ += g_μ
        grad_Σ += diaghessian_MultinomialLogistic(samples[i,:],σ,class)/s - abs2.(g_μ)
    end
    for k in 1:model.nLatent
        model.inference.∇μE[k][index] = grad_μ[k]/nSamples
        model.inference.∇ΣE[k][index] = 0.5.*grad_Σ[k]/nSamples
    end
end

function log_like_samples(model::AbstractGP{<:MultinomialLogisticLikelihood,<:Inference,T},samples::AbstractMatrix,index::Integer) where {T<:Real}
    class = model.likelihood.y_class[index]
    nSamples = size(samples,1)
    loglike = zero(T)
    for i in 1:nSamples
        σ = logistic.(samples[i,:])
        loglike += log(σ[class])-log(sum(σ))
    end
    return loglike/nSamples
end

function remove_augmentation(l::Type{AugmentedMultinomialLogisticLikelihood{T}}) where T
    return MultinomialLogisticLikelihood{T}(l.Y,l.class_mapping,l.ind_mapping)
end

function grad_MultinomialLogistic(s::AbstractVector{<:Real},σ::AbstractVector{<:Real},i::Integer)
    s[i]*(δ.(i,eachindex(σ)).-s).*(1.0.-σ)
end

function diaghessian_MultinomialLogistic(s::AbstractVector{<:Real},σ::AbstractVector{<:Real},i::Integer)
    s[i]*(1.0.-σ).*(
    abs2.(δ.(i,eachindex(σ))-s).*(1.0.-σ)
    -s.*(1.0.-s).*(1.0.-σ)
    -σ.*(δ.(i,eachindex(σ))-s))
end

function hessian_MultinomialLogistic(s::AbstractVector{T},σ::AbstractVector{T},i::Integer) where {T<:Real}
    m = length(s)
    hessian = zeros(T,m,m)
    @inbounds for j in 1:m
        for k in 1:m
            hessian[j,k] = (1-σ[j])*s[i]*(
            (δ(i,k)-s[k])*(1.0-σ[k])*(δ(i,j)-s[j])
            -s[j]*(δ(j,k)-s[k])*(1.0-σ[k])
            -δ(k,j)*σ[j]*(δ(i,j)-s[j]))
        end
    end
    return hessian
end
