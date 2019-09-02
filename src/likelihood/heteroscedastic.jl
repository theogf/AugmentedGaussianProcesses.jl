"""
**Heteroscedastic Likelihood**

Gaussian with heteroscedastic noise given by another gp: ``p(y|f,g) = \\mathcal{N}(y|f,(\\lambda\\sigma(g))^{-1})``

```julia
HeteroscedasticLikelihood([kernel=RBFKernel(),[priormean=0.0]])
```
Augmentation is described here (#TODO)
"""
struct HeteroscedasticLikelihood{T<:Real} <: RegressionLikelihood{T}
    kernel::LatentArray{Kernel{T}}
    μ₀::LatentArray{PriorMean{T}}
    λ::LatentArray{T}
    c::LatentArray{Vector{T}}
    ϕ::LatentArray{Vector{T}}
    γ::LatentArray{Vector{T}}
    θ::LatentArray{Vector{T}}
    μ::LatentArray{Vector{T}}
    Σ::LatentArray{Symmetric{T,Matrix{T}}}
    σg::LatentArray{Vector{T}}
    K::LatentArray{Symmetric{T,Matrix{T}}}
    invK::LatentArray{Symmetric{T,Matrix{T}}}
    κ::LatentArray{Matrix{T}}
    K̃::LatentArray{Vector{T}}
    function HeteroscedasticLikelihood{T}(kernel::Vector{<:Kernel{T}},priormean::Vector{<:PriorMean}) where {T<:Real}
            new{T}(kernel,priormean)
    end
    function HeteroscedasticLikelihood{T}(kernel::AbstractVector{<:Kernel},μ₀::AbstractVector{<:PriorMean},λ::AbstractVector{T},c::AbstractVector{<:AbstractVector{T}},ϕ::AbstractVector{<:AbstractVector{T}},γ::AbstractVector{<:AbstractVector{T}},θ::AbstractVector{<:AbstractVector{T}},μ::AbstractVector{<:AbstractVector{T}},Σ::AbstractVector{<:Symmetric{T,Matrix{T}}},σg::AbstractVector{<:AbstractVector{T}},K::AbstractVector{<:Symmetric{T,Matrix{T}}},invK::AbstractVector{<:Symmetric{T,Matrix{T}}}) where {T<:Real}
         new{T}(kernel,μ₀,λ,c,ϕ,γ,θ,μ,Σ,σg,K,invK)
     end
     function HeteroscedasticLikelihood{T}(kernel::AbstractVector{<:Kernel},μ₀::AbstractVector{<:PriorMean},λ::AbstractVector{T},c::AbstractVector{<:AbstractVector{T}},ϕ::AbstractVector{<:AbstractVector{T}},γ::AbstractVector{<:AbstractVector{T}},θ::AbstractVector{<:AbstractVector{T}},μ::AbstractVector{<:AbstractVector{T}},Σ::AbstractVector{<:Symmetric{T,Matrix{T}}},σg::AbstractVector{<:AbstractVector{T}},K::AbstractVector{<:Symmetric{T,Matrix{T}}},invK::AbstractVector{<:Symmetric{T,Matrix{T}}},κ::AbstractVector{<:AbstractMatrix{T}},K̃::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
          new{T}(kernel,μ₀,λ,c,ϕ,γ,θ,μ,Σ,σg,K,invK,κ,K̃)
      end
end

function HeteroscedasticLikelihood(kernel::Union{Kernel{T},AbstractVector{<:Kernel{T}}}=RBFKernel(),priormean::PriorMean{T}=ConstantMean(0.0)) where {T<:Real}
    if typeof(kernel) <: AbstractVector
        HeteroscedasticLikelihood{T}(kernel,[priormean])
    else
        HeteroscedasticLikelihood{T}([kernel],[priormean])
    end
end

function pdf(l::HeteroscedasticLikelihood,y::Real,f::Real,g::Real)
    pdf(Normal(y,inv(sqrt(l.λ[1]*logistic(g)))),f) #WARNING multioutput invalid
end

function logpdf(l::HeteroscedasticLikelihood,y::Real,f::Real,g::Real)
    logpdf(Normal(y,inv(sqrt(l.λ[1]*logistic(g)))),f) #WARNING multioutput invalid
end

function Base.show(io::IO,model::HeteroscedasticLikelihood{T}) where T
    print(io,"Gaussian likelihood with heteroscedastic noise")
end

function init_likelihood(likelihood::HeteroscedasticLikelihood{T},inference::Inference{T},nLatent::Integer,nSamplesUsed::Integer,nFeatures::Integer) where {T<:Real}
    kernel = Vector{Kernel}(undef,nLatent)
    if length(likelihood.kernel) != nLatent
        kernel[:] = [deepcopy(likelihood.kernel[1]) for _ in 1:nLatent]
    else
        kernel[:] = likelihood.kernel
    end
    μ₀ = [deepcopy(likelihood.μ₀[1]) for _ in 1:nLatent]
    λ = ones(T,nLatent)
    c = [ones(T,nSamplesUsed) for _ in 1:nLatent]
    ϕ = [ones(T,nSamplesUsed) for _ in 1:nLatent]
    γ = [ones(T,nSamplesUsed) for _ in 1:nLatent]
    θ = [ones(T,nSamplesUsed) for _ in 1:nLatent]
    μ = [zeros(T,nSamplesUsed) for _ in 1:nLatent]
    Σ = [Symmetric(Matrix(Diagonal(one(T)*I,nFeatures))) for _ in 1:nLatent]
    σg = [ones(T,nSamplesUsed) for _ in 1:nLatent]
    K = [Symmetric(Matrix(Diagonal(one(T)*I,nFeatures))) for _ in 1:nLatent]
    invK = [Symmetric(Matrix(Diagonal(one(T)*I,nFeatures))) for _ in 1:nLatent]
    κ = [ones(T,nSamplesUsed,nFeatures) for _ in 1:nLatent]
    K̃ = [ones(T,nSamplesUsed) for _ in 1:nLatent]
    HeteroscedasticLikelihood{T}(kernel,μ₀,λ,c,ϕ,γ,θ,μ,Σ,σg,K,invK,κ,K̃)
end

function local_updates!(model::VGP{T,HeteroscedasticLikelihood{T}}) where {T}
    model.likelihood.ϕ .= broadcast((μ,Σ,y)->0.5*(abs2.(μ-y)+Σ),model.μ,diag.(model.Σ),model.y)
    model.likelihood.c .= broadcast((μ,Σ)->sqrt.(Σ+abs2.(μ)),model.likelihood.μ,diag.(model.likelihood.Σ))
    model.likelihood.γ .= broadcast((λ,ϕ,μ,c)->0.5*λ*ϕ.*safe_expcosh.(-0.5*μ,0.5*c),model.likelihood.λ,model.likelihood.ϕ,model.likelihood.μ,model.likelihood.c)
    model.likelihood.θ .= broadcast((γ,c)->0.5*(0.5.+γ)./c.*tanh.(0.5*c),model.likelihood.γ,model.likelihood.c)
    model.likelihood.K .= broadcast(kernel->Symmetric(kernelmatrix(model.X,kernel)+getvariance(kernel)*T(jitter)*I),model.likelihood.kernel)
    model.likelihood.invK .= inv.(model.likelihood.K)
    model.likelihood.Σ .= broadcast((θ,invK)->Symmetric(inv(Diagonal(θ)+invK)),model.likelihood.θ,model.likelihood.invK)
    model.likelihood.μ .= broadcast((Σ,invK,μ₀,γ)->Σ*(invK*μ₀+0.5*(0.5.-γ)),model.likelihood.Σ,model.likelihood.invK,model.likelihood.μ₀,model.likelihood.γ)
    model.likelihood.σg .=  broadcast((μ,Σ)->expectation.(logistic,Normal.(μ,sqrt.(diag(Σ)))),model.likelihood.μ,model.likelihood.Σ)
    model.likelihood.λ .= 0.5*broadcast((ϕ,σg)->model.inference.nSamples/dot(ϕ,σg),model.likelihood.ϕ,model.likelihood.σg)
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

function local_updates!(model::SVGP{T,HeteroscedasticLikelihood{T}}) where {T}
    model.likelihood.ϕ .= broadcast((κ,μ,Σ,y,K̃)->0.5*(abs2.(κ*μ-y)+opt_diag(κ*Σ,κ)+K̃),model.κ,model.μ,diag.(model.Σ),model.inference.y,model.K̃)
    model.likelihood.c .= broadcast((κ,μ,Σ,K̃)->sqrt.(opt_diag(κ*Σ,κ)+abs2.(κ*μ)+K̃),model.likelihood.κ,model.likelihood.μ,diag.(model.likelihood.Σ),model.likelihood.K̃)
    model.likelihood.γ .= broadcast((λ,ϕ,κμ,c)->0.5*λ*ϕ.*safe_expcosh.(-0.5*μ,0.5*c),model.likelihood.λ,model.likelihood.ϕ,model.likelihood.κ.*model.likelihood.μ,model.likelihood.c)
    model.likelihood.θ .= broadcast((γ,c)->0.5*(0.5.+γ)./c.*tanh.(0.5*c),model.likelihood.γ,model.likelihood.c)
    model.likelihood.K .= broadcast(kernel->Symmetric(kernelmatrix(model.Z,kernel)+getvariance(kernel)*T(jitter)*I),model.likelihood.kernel)
    model.likelihood.invK .= inv.(model.likelihood.K)
    model.likelihood.Knm .= broadcast(kernel->kernelmatrix(model.inference.x,model.Z,kernel),model.likelihood.kernel)
    model.likelihood.κ .= model.likelihood.Knm.*model.invK
    model.likelihood.K̃ .=  broadcast(kernel->kerneldiagmatrix(model.inference.x,kernel),model.likelihood.kernel)
    model.likelihood.Σ .= broadcast((θ,κ,invK)->Symmetric(inv(model.inference.ρ*transpose(κ)*Diagonal(θ)*κ+invK)),model.likelihood.θ,model.likelihood.κ,model.likelihood.invK)
    model.likelihood.μ .= broadcast((Σ,invK,μ₀,γ)->Σ*(invK*μ₀+0.5*transpose(κ)*(0.5.-γ)),model.likelihood.Σ,model.likelihood.invK,model.likelihood.μ₀,model.likelihood.γ)
    model.likelihood.σg .=  broadcast((μ,Σ)->expectation.(logistic,Normal.(μ,sqrt.(diag(Σ)))),model.likelihood.μ,model.likelihood.Σ)
    model.likelihood.λ .= 0.5*broadcast((ϕ,σg)->model.inference.nSamples/dot(ϕ,σg),model.likelihood.ϕ,model.likelihood.σg)
end

@inline ∇E_μ(model::AbstractGP{T,<:HeteroscedasticLikelihood,<:AnalyticVI}) where {T} = 0.5*hadamard.(model.inference.y,model.likelihood.λ.*model.likelihood.σg)
@inline ∇E_μ(model::AbstractGP{T,<:HeteroscedasticLikelihood,<:AnalyticVI},i::Int) where {T} = 0.5*model.likelihood.λ[i]*model.inference.y[i].*model.likelihood.σg[i]

@inline ∇E_Σ(model::AbstractGP{T,<:HeteroscedasticLikelihood,<:AnalyticVI}) where {T} = 0.5*model.likelihood.λ.*model.likelihood.σg
@inline ∇E_Σ(model::AbstractGP{T,<:HeteroscedasticLikelihood,<:AnalyticVI},i::Int) where {T} = 0.5*model.likelihood.λ[i].*model.likelihood.σg[i]

function proba_y(model::VGP{T,HeteroscedasticLikelihood{T},AnalyticVI{T}},X_test::AbstractMatrix{T}) where {T<:Real}
    μf, σ²f = predict_f(model,X_test,covf=true)
    μg, σ²g = _predict_f.(model.likelihood.μ,model.likelihood.Σ,model.likelihood.invK,model.likelihood.kernel,[X_test],[model.X],covf=true)[1]#WARNING Only valid for 1D output
    return μf,σ²f.+broadcast((λ,μ,σ)->expectation.(x->inv(λ*logistic(x)),Normal.(μ,sqrt.(σ))),model.likelihood.λ,μg,σ²g)
end

function proba_y(model::SVGP{T,HeteroscedasticLikelihood{T},AnalyticVI{T}},X_test::AbstractMatrix{T}) where {T}
    μf, σ²f = predict_f(model,X_test,covf=true)
    μg, σ²g = _predict_f.(model.likelihood.μ,model.likelihood.Σ,model.likelihood.invK,model.likelihood.kernel,[X_test],model.Z,covf=true)

    return μf,σ²f.+broadcast((λ,μ,σ)->dot.([pred_weights],eachcol(inv.(λ*logistic.(pred_nodes.*sqrt2*sqrt.(σ)'.+μ')))),model.likelihood.λ,μg,σ²g)
end

function ELBO(model::AbstractGP{T,HeteroscedasticLikelihood{T},<:AnalyticVI}) where {T}
    return expecLogLikelihood(model) - GaussianKL(model) - GaussianKL_g(model) - PoissonKL(model) - PolyaGammaKL(model)
end


function expecLogLikelihood(model::VGP{T,HeteroscedasticLikelihood{T}}) where {T}
    tot = model.nFeatures*(0.5*sum(log,model.likelihood.λ)-model.nLatent*(log(2*sqrt(twoπ))))
    tot += 0.5*sum(broadcast((μ,γ,Σ,θ)->dot(μ,(0.5 .- γ)) - dot(abs2.(μ),θ)-dot(Σ,θ),model.likelihood.μ,model.likelihood.γ,diag.(model.likelihood.Σ),model.likelihood.θ))
    return tot
end

function GaussianKL_g(model::AbstractGP{T,<:HeteroscedasticLikelihood}) where {T}
    sum(broadcast(GaussianKL,model.likelihood.μ,model.likelihood.μ₀,model.likelihood.Σ,model.likelihood.invK))
end

function PoissonKL(model::VGP{T,<:HeteroscedasticLikelihood}) where {T}
    return sum(broadcast(PoissonKL,model.likelihood.γ,
            broadcast((λ,y,μ,Σ)->0.5*λ*(abs2.(y-μ)+Σ),model.likelihood.λ,model.y,model.μ,diag.(model.Σ)),
            broadcast((λ,y,μ,Σ)->log.(0.5*λ*(abs2.(μ-y)+Σ)),model.likelihood.λ,model.y,model.μ,diag.(model.Σ)))) #TODO
end

function PolyaGammaKL(model::VGP{T,<:HeteroscedasticLikelihood}) where {T}
    sum(broadcast(PolyaGammaKL,broadcast(γ->0.5.+γ,model.likelihood.γ),model.likelihood.c,model.likelihood.θ))
end
