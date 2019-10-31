"""
**Heteroscedastic Likelihood**

Gaussian with heteroscedastic noise given by another gp: ``p(y|f,g) = \\mathcal{N}(y|f,(\\lambda\\sigma(g))^{-1})``

```julia
HeteroscedasticLikelihood([kernel=RBFKernel(),[priormean=0.0]])
```
Augmentation is described here (#TODO)
"""
struct HeteroscedasticLikelihood{T<:Real} <: RegressionLikelihood{T}
    λ::T
    c::Vector{T}
    ϕ::Vector{T}
    γ::Vector{T}
    θ::Vector{T}
    σg::Vector{T}
    function HeteroscedasticLikelihood{T}(λ::T) where {T<:Real}
            new{T}(λ)
    end
    function HeteroscedasticLikelihood{T}(λ::T,c::AbstractVector{T},ϕ::AbstractVector{T},γ::AbstractVector{T},θ::AbstractVector{T},σg::AbstractVector{T}) where {T<:Real}
         new{T}(λ,c,ϕ,γ,θ,σg)
    end
end

function HeteroscedasticLikelihood(λ::T=1.0) where {T<:Real}
        HeteroscedasticLikelihood{T}(λ)
end

function pdf(l::HeteroscedasticLikelihood,y::Real,f::Real,g::Real)
    pdf(Normal(y,inv(sqrt(l.λ*logistic(g)))),f)
end

function logpdf(l::HeteroscedasticLikelihood,y::Real,f::Real,g::Real)
    logpdf(Normal(y,inv(sqrt(l.λ*logistic(g)))),f)
end

function Base.show(io::IO,model::HeteroscedasticLikelihood{T}) where T
    print(io,"Gaussian likelihood with heteroscedastic noise")
end

function treat_labels!(y::AbstractVector{T},likelihood::L) where {T,L<:HeteroscedasticLikelihood}
    @assert T<:Real "For regression target(s) should be real valued"
    return y,2,likelihood
end

function init_likelihood(likelihood::HeteroscedasticLikelihood{T},inference::Inference{T},nLatent::Integer,nMinibatch::Integer,nFeatures::Integer) where {T<:Real}
    λ = likelihood.λ
    c = ones(T,nMinibatch)
    ϕ = ones(T,nMinibatch)
    γ = ones(T,nMinibatch)
    θ = ones(T,nMinibatch)
    σg = ones(T,nMinibatch)
    HeteroscedasticLikelihood{T}(λ,c,ϕ,γ,θ,σg)
end

function local_updates!(l::HeteroscedasticLikelihood{T},y::AbstractVector,μ::Tuple{2,<:AbstractVector},diag_cov::Tuple{2,<:AbstractVector}) where {T}
    l.ϕ .= 0.5*(abs2.(μ[1]-y)+diag_cov[1])
    l.c .=  sqrt.(abs2.(μ[2])+diag_cov[2])
    l.γ .= 0.5*l.λ*l.ϕ.*safe_expcosh.(-0.5*μ[2],0.5*c)
    l.θ .= 0.5*(0.5.+l.γ)./l.c.*tanh.(0.5*l.c)
    model.likelihood.μ .= broadcast((Σ,invK,μ₀,γ)->Σ*(invK*μ₀+0.5*(0.5.-γ)),model.likelihood.Σ,model.likelihood.invK,model.likelihood.μ₀,model.likelihood.γ)
end

function local_autotuning!(model::VGP{T,<:HeteroscedasticLikelihood}) where {T}
    Jnn = kernelderivativematrix.([model.X],model.likelihood.kernel)
    f_l,f_v,f_μ₀ = hyperparameter_local_gradient_function(model)
    grads_l = map(compute_hyperparameter_gradient,model.likelihood.kernel,fill(f_l,model.nLatent),Jnn,1:model.nLatent)
    grads_v = map(f_v,model.likelihood.kernel,1:model.nPrior)
    grads_μ₀ = map(f_μ₀,1:model.nLatent)

    apply_gradients_lengthscale!.(model.likelihood.kernel,grads_l) #Send the derivative of the matrix to the specific gradient of the model
    apply_gradients_variance!.(model.likelihood.kernel,grads_v) #Send the derivative of the matrix to the specific gradient of the model
    update!.(model.likelihood.μ₀,grads_μ₀)

    model.inference.HyperParametersUpdated = true
end

function variational_updates!(model::AbstractGP{T,<:HeteroscedasticLikelihood,<:AnalyticVI}) where {T,L}
    local_updates!(model.likelihood,get_y(model),mean_f(model),diag_cov_f(model))
    natural_gradient_g!(model.likelihood,model.inference.model.inference.vi_opt[2],model.f[2])
    heteroscedastic_expectations!(model.likelihood,model.f[2].μ,diag(model.f[2].Σ))
    natural_gradient!(model.likelihood,model.inference,model.inference.vi_opt[1],[get_y(model)],model.f[1])
    global_update!(model.f[1],model.inference.vi_opt[1],model.inference)
end

function natural_gradient_g!(l::HeteroscedasticLikelihood{T},gp::_VGP,opt::AVIOptimizer,i::AnalyticVI) where {T}
    gp.η₁ .= 0.5*(0.5.-l.γ)+gp.K/(gp.μ-gp.μ₀)
    gp.η₂ .= -0.5*(Diagonal(l.θ)+inv(gp.K))
    global_update!(gp)
end

function natural_gradient_g!(l::HeteroscedasticLikelihood{T},gp::_SVGP,opt::AVIOptimizer,i::AnalyticVI) where {T}
    opt.∇η₁ .= ∇η₁(0.5*(0.5.-l.γ),i.ρ,gp.κ,gp.K,gp.μ₀,gp.η₁)
    opt.∇η₂ .= ∇η₂(0.5*l.θ,i.ρ,gp.κ,gp.K,gp.η₂)
    global_update!(gp,opt,i)
end

function heteroscedastic_expectations(l::HeteroscedasticLikelihood{T},μ::AbstractVector,Σ::AbstractVector) where {T}
    l.σg .= expectation.(logistic,μ,Σ)
    l.λ .= 0.5*length(l.ϕ)/dot(l.ϕ,l.σg)
end

function expectation(f::Function,μ::Real,σ²::Real)
    x = pred_nodes*sqrt(max(σ²,zero(σ²))).+μ
    dot(pred_weights,f.(x))
end

@inline ∇E_μ(l::HeteroscedasticLikelihood,::AVIOptimizer,y::AbstractVector) where {T} = 0.5*y.*l.λ.*l.σg

@inline ∇E_Σ(l::HeteroscedasticLikelihood,::AVIOptimizer,y::AbstractVector) where {T} = 0.5*l.λ.*l.σg

function proba_y(model::AbstractGP{T,HeteroscedasticLikelihood{T},AnalyticVI{T}},X_test::AbstractMatrix{T}) where {T<:Real}
    (μf, σ²f),(μg, σ²g) = predict_f(model,X_test,covf=true)
    return μf,σ²f + expectation.(x->inv(model.likelihood.λ*logistic(x)),μg,σ²g)
end

function ELBO(model::AbstractGP{T,HeteroscedasticLikelihood{T},<:AnalyticVI}) where {T}
    return model.inference.ρ*expec_logpdf(model.likelihood, get_y(model), mean_f(model.f[2]), diag_cov_f(model.f[2])) - GaussianKL(model) - model.inference.ρ.*(PoissonKL(model.likelihood,mean_f(model.f[1]),diag_cov_f(model.f[2])) - PolyaGammaKL(model))
end


function expec_logpdf(l::HeteroscedasticLikelihood{T},μ::AbstractVector,diag_cov::AbstractVector) where {T}
    tot = length(μ)*(0.5*log(l.λ)-log(2*sqrt(twoπ)))
    tot += 0.5*(dot(μ[2],(0.5 .- l.γ)) - dot(abs2.(μ[2]),θ)-dot(diag_cov[2],θ))
    return tot
end

function PoissonKL(l::HeteroscedasticLikelihood{T},y::AbstractVector,μ::AbstractVector,Σ::AbstractVector) where {T}
    return PoissonKL(l.γ,0.5*l.λ*(abs2.(y-μ)+Σ),log.(0.5*l.λ*(abs2.(μ-y)+Σ)))
end

function PolyaGammaKL(l::HeteroscedasticLikelihood{T}) where {T}
    PolyaGammaKL(0.5.+l.γ,l.c,l.θ)
end
