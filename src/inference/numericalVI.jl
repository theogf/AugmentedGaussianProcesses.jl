"""
Solve any non-conjugate likelihood using Variational Inference
by making a numerical approximation (quadrature or MC integration)
of the expected log-likelihood ad its gradients
Gradients are computed as in "The Variational Gaussian Approximation
Revisited" by Opper and Archambeau 2009
"""
abstract type NumericalVI{T<:Real} <: Inference{T} end

include("quadratureVI.jl")
include("MCVI.jl")


""" `NumericalVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1))`

General constructor for Variational Inference via numerical approximation.

**Argument**

    -`integration_technique::Symbol` : Method of approximation can be `:quad` for quadrature see [QuadratureVI](@ref) or `:mc` for MC integration see [MCIntegrationVI](@ref)

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation (for the MCIntegrationVI)
    - `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
    - `optimizer::Optimizer` : Optimizer used for the variational updates. Should be an Optimizer object from the [GradDescent.jl]() package. Default is `Adam()`
"""
function NumericalVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer::Optimizer=Momentum(η=1e-5)) where {T<:Real}
    if integration_technique == :quad
        QuadratureVI{T}(ϵ,nGaussHermite,optimizer,false,0)
    elseif integration_technique == :mc
        MCIntegrationVI{T}(ϵ,nMC,optimizer,false,0)
    else
        @error "Only possible integration techniques are quadrature : :quad or mcmc integration :mcmc"
    end
end

""" `NumericalSVI(integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=1000,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1))`

General constructor for Stochastic Variational Inference via numerical approximation.

**Argument**

    -`nMinibatch::Integer` : Number of samples per mini-batches
    -`integration_technique::Symbol` : Method of approximation can be `:quad` for quadrature see [QuadratureVI](@ref) or `:mc` for MC integration see [MCIntegrationVI](@ref)

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation (for the MCIntegrationVI)
    - `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
    - `optimizer::Optimizer` : Optimizer used for the variational updates. Should be an Optimizer object from the [GradDescent.jl]() package. Default is `Adam()`
"""
function NumericalSVI(nMinibatch::Integer,integration_technique::Symbol=:quad;ϵ::T=1e-5,nMC::Integer=200,nGaussHermite::Integer=20,optimizer::Optimizer=Momentum(η=1e-5)) where {T<:Real}
    if integration_technique == :quad
        QuadratureVI{T}(ϵ,nGaussHermite,optimizer,true,nMinibatch)
    elseif integration_technique == :mc
        MCIntegrationVI{T}(ϵ,nMC,optimizer,true,nMinibatch)
    else
        @error "Only possible integration techniques are quadrature : :quad or mcmc integration :mc"
    end
end

function Base.show(io::IO,inference::NumericalVI{T}) where T
    print(io,"$(inference.Stochastic ? "Stochastic numerical" : "Numerical") inference by $(isa(inference,MCIntegrationVI) ? "Monte Carlo Integration" : "Quadrature")")
end

∇E_μ(::Likelihood,i::Union{<:NVIOptimizer,<:NumericalVI},::AbstractVector) = -i.ν
∇E_Σ(::Likelihood,i::Union{<:NVIOptimizer,<:NumericalVI},::AbstractVector) = 0.5.*i.λ

function variational_updates!(model::AbstractGP{T,L,<:NumericalVI}) where {T,L}
    compute_grad_expectations!(model)
    natural_gradient!.(model.likelihood,model.inference,model.inference.vi_opt,[get_y(model)],model.f)
    global_update!(model)
end

function natural_gradient!(l::Likelihood,i::NumericalVI,opt::NVIOptimizer,y::AbstractVector,gp::_VGP{T}) where {T,L}
    opt.∇η₂ .= Diagonal(∇E_Σ(l,opt,y)) - 0.5*inv(gp.K) - gp.η₂
    opt.∇η₁ .= ∇E_μ(l,opt,y) - gp.K \ (gp.μ - gp.μ₀) - 2 * opt.∇η₂ * gp.μ
end

function natural_gradient!(l::Likelihood,i::NumericalVI,opt::NVIOptimizer,y::AbstractVector,gp::_SVGP{T}) where {T,L}
    opt.∇η₂ .= i.ρ*transpose(gp.κ)*Diagonal(∇E_Σ(l,opt,y))*gp.κ - 0.5*inv(gp.K) - gp.η₂
    opt.∇η₁ .= i.ρ * transpose(gp.κ) * ∇E_μ(l,opt,y) - gp.K \ (gp.μ - gp.μ₀) - 2 * opt.∇η₂ * gp.μ  #TO CHECK
end

function global_update!(model::AbstractGP{T,L,<:NumericalVI}) where {T,L}
    # model.η₁ .= model.η₁ .+ update.(model.inference.optimizer_η₁,model.inference.∇η₁)
    for (gp,opt) in zip(model.f,model.inference.vi_opt)
        Δ = update(opt.optimizer,vcat(opt.∇η₁,vec(opt.∇η₂)))
        Δ₁ = Δ[1:model.nFeatures]
        Δ₂ = reshape(Δ[model.nFeatures+1:end],model.nFeatures,model.nFeatures)
        # global C = Matrix(cholesky(-gp.η₂).L)
        # global Lk = eliminate_matrix(gp.dim)
        # global Kk = commute_transpose(gp.dim)
        # global Dk = duplicate_matrix(gp.dim)
        # global J = Lk*(kron(C,diagm(ones(T,gp.dim)))+kron(diagm(ones(T,gp.dim)),C)*Kk)*Dk
        # @show J
        # Δ₂ = Symmetric(reshape(Δ[model.nFeatures+1:end],gp.dim,gp.dim))
        α = 1.0
        # Loop to verify update keeps positive definiteness
        while !isposdef(-(gp.η₂ + α*Δ₂)) &&  α > 1e-7
            α *= 0.1
        end
        if α <= 1e-7
            @error "α too small, positive definiteness could not be achieved"
        end

        ## Passed the pos. def. test, now update parameters
        gp.η₂ = Symmetric(gp.η₂+α*Δ₂)
        gp.η₁ .+= α*Δ₁

        ## Save the new scaling on the optimizer
        if isa(opt.optimizer,Adam)
            opt.optimizer.α = min(opt.optimizer_η₂.α * α* 2.0,0.1)
        elseif isa(opt.optimizer,Union{VanillaGradDescent,Momentum,RMSprop})
            opt.optimizer.η = min(opt.optimizer.η*α*1.5,1.0)
        elseif isa(opt.optimizer,ALRSVI)
        elseif isa(opt.optimizer,InverseDecay)
        end

        ## Reparametrize to the normal distribution
        global_update!.(model.f)
    end
end

function ELBO(model::AbstractGP{T,L,<:NumericalVI}) where {T,L}
    return expec_logpdf(model) - GaussianKL(model)
end

# function convert(::Type{T1},x::T2) where {T1<:VGP{<:Likelihood,T3} where {T3<:NumericalVI},T2<:VGP{<:Real,<:Likelihood,<:AnalyticVI}}
#     #TODO Check if likelihood is compatible
#     inference = T3(x.inference.ϵ,x.inference.nIter,x.inference.optimizer,defaultn(T3),x.inference.Stochastic,x.inference.nSamples,x.inference.nSamplesUsed,x.inference.MBIndices,x.inference.ρ,x.inference.HyperParametersUpdated,x.inference.∇η₁,x.inference.∇η₂,copy(expec_μ(x)),copy(expec_Σ(x)))
#     likelihood =isaugmented(x.likelihood) ? remove_augmentation(x.likelihood) : likelihood
#     return T1(x.X,x.y,x.nSample,x.nDim,x.nFeatures,x.nLatent,x.IndependentPriors,x.nPrior,x.μ,x.Σ,x.η₁,x.η₂,x.Knn,x.invKnn,x.kernel,likelihood,inference,x.verbose,x.optimizer,x.atfrequency,x.Trained)
# end
