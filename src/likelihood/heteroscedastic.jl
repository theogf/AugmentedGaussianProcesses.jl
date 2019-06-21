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
    K::LatentArray{Symmetric{T,Matrix{T}}}
    invK::LatentArray{Symmetric{T,Matrix{T}}}
    σg::LatentArray{Vector{T}}
    function HeteroscedasticLikelihood{T}(kernel::Vector{<:Kernel{T}},priormean::Vector{<:PriorMean}) where {T<:Real}
            new{T}(kernel,priormean)
    end
    function HeteroscedasticLikelihood{T}(kernel::AbstractVector{<:Kernel},μ₀::AbstractVector{<:PriorMean},λ::AbstractVector{T},c::AbstractVector{<:AbstractVector{T}},ϕ::AbstractVector{<:AbstractVector{T}},γ::AbstractVector{<:AbstractVector{T}},θ::AbstractVector{<:AbstractVector{T}},μ::AbstractVector{<:AbstractVector{T}},Σ::AbstractVector{<:Symmetric{T,Matrix{T}}},K::AbstractVector{<:Symmetric{T,Matrix{T}}},invK::AbstractVector{<:Symmetric{T,Matrix{T}}},σg::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
         new{T}(kernel,μ₀,λ,c,ϕ,γ,θ,μ,Σ,K,invK,σg)
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

function init_likelihood(likelihood::HeteroscedasticLikelihood{T},inference::Inference{T},nLatent::Integer,nSamplesUsed::Integer) where {T<:Real}
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
    Σ = [Symmetric(Matrix(Diagonal(one(T)*I,nSamplesUsed))) for _ in 1:nLatent] #WARNING Temp fix (not valid for SVGP)
    K = [Symmetric(Matrix(Diagonal(one(T)*I,nSamplesUsed))) for _ in 1:nLatent]
    invK = [Symmetric(Matrix(Diagonal(one(T)*I,nSamplesUsed))) for _ in 1:nLatent]
    σg = [ones(T,nSamplesUsed) for _ in 1:nLatent]
    HeteroscedasticLikelihood{T}(kernel,μ₀,λ,c,ϕ,γ,θ,μ,Σ,K,invK,σg)
end

function local_updates!(model::VGP{HeteroscedasticLikelihood{T}}) where {T<:Real}
    model.likelihood.ϕ .= broadcast((μ,Σ,y)->0.5*(abs2.(μ-y)+Σ),model.μ,diag.(model.Σ),model.y)
    model.likelihood.c .= broadcast((μ,Σ)->sqrt.(Σ+abs2.(μ)),model.likelihood.μ,diag.(model.likelihood.Σ))
    model.likelihood.γ .= broadcast((λ,ϕ,μ,c)->0.5*λ*ϕ.*safe_expcosh.(-0.5*μ,0.5*c),model.likelihood.λ,model.likelihood.ϕ,model.likelihood.μ,model.likelihood.c)
    model.likelihood.θ .= broadcast((γ,c)->0.5*(one(T).+γ)./c.*tanh.(0.5*c),model.likelihood.γ,model.likelihood.c)
    model.likelihood.K .= broadcast(kernel->Symmetric(kernelmatrix(model.X,kernel)+getvariance(kernel)*T(jitter)*I),model.likelihood.kernel)
    model.likelihood.invK .= inv.(model.likelihood.K)
    model.likelihood.Σ .= broadcast((θ,invK)->Symmetric(inv(Diagonal(θ)+invK)),model.likelihood.θ,model.likelihood.invK)
    model.likelihood.μ .= broadcast((Σ,invK,μ₀,γ)->Σ*(invK*μ₀+0.5*(one(T).-γ)),model.likelihood.Σ,model.likelihood.invK,model.likelihood.μ₀,model.likelihood.γ)
    model.likelihood.σg .=  broadcast((μ,Σ)->expectation.(logistic,Normal.(μ,sqrt.(diag(Σ)))),model.likelihood.μ,model.likelihood.Σ)
    # model.likelihood.λ .= broadcast((ϕ,σg)->model.inference.nSamples/dot(ϕ,σg),model.likelihood.ϕ,model.likelihood.σg)
end

function local_updates!(model::SVGP{HeteroscedasticLikelihood{T}}) where {T<:Real}
    model.likelihood.c .= broadcast((μ,Σ)->sqrt.(Σ+abs2.(μ)),model.likelihood.μ,diag.(model.likelihood.Σ))
    model.likelihood.γ .= broadcast((λ,μ,c)->0.5*λ*safe_expcosh.(-0.5*μ,0.5*c),model.likelihood.λ,model.likelihood.μ,model.likelihood.c)
    model.likelihood.θ .= broadcast((γ,c)->0.5*(one(T).+γ)./c*tanh.(0.5*c),model.likelihood.γ,model.likelihood.c)
    model.likelihood.K = broadcast((kernel)->Symmetric(kernelmatrix(model.X,kernel)+getvariance(kernel)*T(jitter)*I),model.likelihood.kernel)
    model.likelihood.invK = inv.(model.likelihood.K)
    model.likelihood.Σ .= broadcast((θ,invK)->Symmetric(inv(Diagonal(θ)+invK)),model.likelihood.θ,model.likelihood.invK)
    model.likelihood.μ .= broadcast((Σ,invK,μ₀,γ)->Σ*(invK*μ₀+0.5*(one(T).-γ)),model.likelihood.Σ,model.likelihood.invK,model.likelihood.μ₀,model.likelihood.γ)
    model.likelihood.λσg .=  broadcast((λ,μ,Σ)->λ*expectation.(logistic,Normal.(μ,sqrt.(diag(Σ)))),model.likelihood.λ,model.likelihood.μ,model.likelihood.Σ)
end

function cond_mean(model::VGP{HeteroscedasticLikelihood{T},AnalyticVI{T}},index::Integer) where {T<:Real}
    return model.likelihood.λ[index]*model.y[index].*model.likelihood.σg[index]
end

function ∇μ(model::VGP{HeteroscedasticLikelihood{T},AnalyticVI{T}}) where {T<:Real}
    return 0.5*hadamard.(model.y,model.likelihood.λ.*model.likelihood.σg)
end

function cond_mean(model::SVGP{HeteroscedasticLikelihood{T},AnalyticVI{T}},index::Integer) where {T<:Real}
    return model.likelihood.λ[index]*model.y[index][model.inference.MBIndices].*σg[index]
end

function ∇μ(model::SVGP{HeteroscedasticLikelihood{T},AnalyticVI{T}}) where {T<:Real}
    return hadamard.(getindex.(model.y,[model.inference.MBIndices]),model.likelihood.λ.*model.likelihood.σg)
end

function ∇Σ(model::AbstractGP{HeteroscedasticLikelihood{T},AnalyticVI{T}}) where {T<:Real}
    return model.likelihood.λ.*model.likelihood.σg
end

function proba_y(model::VGP{HeteroscedasticLikelihood{T},AnalyticVI{T}},X_test::AbstractMatrix{T}) where {T<:Real}
    μf, σ²f = predict_f(model,X_test,covf=true)
    μg, σ²g = _predict_f.(model.likelihood.μ,model.likelihood.Σ,model.likelihood.invK,model.likelihood.kernel,[X_test],[model.X],covf=true)[1]#WARNING Only valid for 1D output
    return μf,σ²f.+broadcast((λ,μ,σ)->expectation.(x->inv(λ*logistic(x)),Normal.(μ,sqrt.(σ))),model.likelihood.λ,μg,σ²g)
end

function proba_y(model::SVGP{HeteroscedasticLikelihood{T},AnalyticVI{T}},X_test::AbstractMatrix{T}) where {T<:Real}
    μf, σ²f = predict_f(model,X_test,covf=true)
    μg, σ²g = _predict_f.(model.likelihood.μ,model.likelihood.Σ,model.likelihood.invK,model.likelihood.kernel,[X_test],model.Z,covf=true)
    return μf,σ²f.+broadcast((λ,μ,σ)->expectation.(x->inv(λ*logistic(x)),Normal.(μ,sqrt.(σ))),model.likelihood.λ,μg,σ²g)
end

### Special case where the ELBO is equal to the marginal likelihood
function ELBO(model::VGP{HeteroscedasticLikelihood{T}}) where {T<:Real}
    # return -0.5*sum(broadcast((y,invK)->dot(y,invK*y) - logdet(invK)+ model.nFeature*log(twoπ),model.y,model.invKnn))
    return NaN
end

function ELBO(model::SVGP{HeteroscedasticLikelihood{T}}) where {T<:Real}
    return expecLogLikelihood(model) - GaussianKL(model)
end

function expecLogLikelihood(model::SVGP{HeteroscedasticLikelihood{T}}) where T
    return NaN
end
