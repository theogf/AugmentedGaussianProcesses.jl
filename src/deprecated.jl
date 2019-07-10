### Contains all old constructors for compatibility issues

deprecation_warning = "Deprecated constructor, use VGP(X,y,kernel,likelihood,inference) or SVGP(X,y,kernel,likelihood,inference,m) in the future"

function BatchGPRegression(X::AbstractArray{T},y::AbstractArray;Autotuning::Bool=true,nEpochs::Integer = 10, kernel=0,noise::T=1e-3,verbose::Integer=0) where {T<:Real}
    @warn deprecation_warning
    model = VGP(X,y,kernel,GaussianLikelihood(noise),AnalyticVI(ϵ=ϵ),verbose=verbose,optimizer=Autotuning,atfrequency=AutotuningFrequency,IndependentPriors=IndependentGPs)
    for i in 1:model.nLatent
        model.μ[i] = μ_init[1]*ones(length(model.μ[i]))
    end
    return model
end

function SparseGPRegression(X::AbstractArray{T},y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true, Autotuning::Bool=false,OptimizeIndPoints::Bool=false, nEpochs::Integer = 10000,batchsize::Integer=-1,κ_s::T=1.0,τ_s::Integer=100, kernel=0,noise::T=1e-3,m::Integer=0,AutotuningFrequency::Integer=2, ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5, verbose::Integer=0) where {T<:Real}
    @warn deprecation_warning
    model = SVGP(X,y,kernel,GaussianLikelihood(noise),Stochastic ? AnalyticSVI(batchsize,ϵ=ϵ,optimizer=AdaptiveLearningRate ? ALRSVI() : InverseDecay(τ=τ_s,κ=κ_s)) : AnalyticVI(ϵ=ϵ),m,verbose=verbose,optimizer=Autotuning,atfrequency=AutotuningFrequency,IndependentPriors=IndependentGPs)
    for i in 1:model.nLatent
        model.μ[i] = μ_init[1]*ones(length(model.μ[i]))
    end
    return model
end

function MultiClass(X::AbstractArray{T},y::AbstractArray;Autotuning::Bool=false,nEpochs::Integer = 200, KStochastic::Bool = false, nClassesUsed::Int=0,
                                kernel=0,AutotuningFrequency::Integer=2,IndependentGPs::Bool=false,
                                ϵ::Real=T(1e-5),μ_init::Vector{T}=zeros(T,1),verbose::Integer=0) where {T<:Real}
    @warn deprecation_warning
    model = VGP(X,y,kernel,LogisticSoftMaxLikelihood(),AnalyticVI(ϵ=ϵ),verbose=verbose,optimizer=Autotuning,atfrequency=AutotuningFrequency,IndependentPriors=IndependentGPs)
    for i in 1:model.nLatent
        model.μ[i] = μ_init[1]*ones(length(model.μ[i]))
    end
    return model
end

function SparseMultiClass(X::AbstractArray{T},y::AbstractArray;Stochastic::Bool=false,KStochastic::Bool=false,nClassesUsed::Int=0,AdaptiveLearningRate::Bool=true, Autotuning::Bool=false,OptimizeIndPoints::Bool=false, IndependentGPs::Bool=true, nEpochs::Integer = 10000,KSize::Int64=-1,batchsize::Integer=-1,κ_s::T=T(0.51),τ_s::Integer=1, kernel=0,m::Integer=0, AutotuningFrequency::Integer=2, ϵ::Real=1e-5,μ_init::Vector{T}=zeros(T,1),SmoothingWindow::Integer=5, verbose::Integer=0) where {T<:Real}
    @warn deprecation_warning
    model = SVGP(X,y,kernel,LogisticSoftMaxLikelihood(),Stochastic ? AnalyticSVI(batchsize,ϵ=ϵ,optimizer=AdaptiveLearningRate ? ALRSVI() : InverseDecay(τ=τ_s,κ=κ_s)) : AnalyticVI(ϵ=ϵ),m,verbose=verbose,optimizer=Autotuning,atfrequency=AutotuningFrequency,IndependentPriors=IndependentGPs)
    for i in 1:model.nLatent
        model.μ[i] = μ_init[1]*ones(length(model.μ[i]))
    end
    return model
end

function BatchBSVM(X::AbstractArray{T},y::AbstractArray;Autotuning::Bool=false,nEpochs::Integer = 200, kernel=0,AutotuningFrequency::Integer=1, ϵ::T=1e-5,μ_init::Vector{T}=ones(T,1),verbose::Integer=0) where {T<:Real}
    @warn deprecation_warning
    model = VGP(X,y,kernel,BayesianSVM(),AnalyticVI(ϵ=ϵ),verbose=verbose,optimizer=Autotuning,atfrequency=AutotuningFrequency,IndependentPriors=IndependentGPs)
    for i in 1:model.nLatent
        model.μ[i] = μ_init[1]*ones(length(model.μ[i]))
    end
    return model
end

function SparseBSVM(X::AbstractArray{T},y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true,Autotuning::Bool=false,OptimizeIndPoints::Bool=false, nEpochs::Integer = 10000,batchsize::Integer=-1,κ_s::Real=1.0,τ_s::Integer=100,kernel=0,m::Integer=0,AutotuningFrequency::Integer=1,ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,verbose::Integer=0) where {T<:Real}
    @warn deprecation_warning
    model = SVGP(X,y,kernel,BayesianSVM(),Stochastic ? AnalyticSVI(batchsize,ϵ=ϵ,optimizer=AdaptiveLearningRate ? ALRSVI() : InverseDecay(τ=τ_s,κ=κ_s)) : AnalyticVI(ϵ=ϵ),m,verbose=verbose,optimizer=Autotuning,atfrequency=AutotuningFrequency,IndependentPriors=IndependentGPs)
    for i in 1:model.nLatent
        model.μ[i] = μ_init[1]*ones(length(model.μ[i]))
    end
    return model
end

function BatchXGPC(X::AbstractArray{T},y::AbstractArray;Autotuning::Bool=false,nEpochs::Integer = 200, kernel=0,AutotuningFrequency::Integer=1, ϵ::T=1e-5,μ_init::Vector{T}=ones(T,1),verbose::Integer=0) where {T<:Real}
    @warn deprecation_warning
    model = VGP(X,y,kernel,LogisticLikelihood(),AnalyticVI(ϵ=ϵ),verbose=verbose,optimizer=Autotuning,atfrequency=AutotuningFrequency,IndependentPriors=IndependentGPs)
    for i in 1:model.nLatent
        model.μ[i] = μ_init[1]*ones(length(model.μ[i]))
    end
    return model
end

function SparseXGPC(X::AbstractArray{T},y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true,Autotuning::Bool=false,OptimizeIndPoints::Bool=false, nEpochs::Integer = 10000,batchsize::Integer=-1,κ_s::Real=1.0,τ_s::Integer=100,kernel=0,m::Integer=0,AutotuningFrequency::Integer=1,ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,verbose::Integer=0) where {T<:Real}
    @warn deprecation_warning
    model = SVGP(X,y,kernel,LogisticLikelihood(),Stochastic ? AnalyticSVI(batchsize,ϵ=ϵ,optimizer=AdaptiveLearningRate ? ALRSVI() : InverseDecay(τ=τ_s,κ=κ_s)) : AnalyticVI(ϵ=ϵ),m,verbose=verbose,optimizer=Autotuning,atfrequency=AutotuningFrequency,IndependentPriors=IndependentGPs)
    for i in 1:model.nLatent
        model.μ[i] = μ_init[1]*ones(length(model.μ[i]))
    end
    return model
end

function BatchStudentT(X::AbstractArray{T},y::AbstractArray;Autotuning::Bool=false, nEpochs::Integer = 200, kernel=0,AutotuningFrequency::Integer=1, ϵ::Real=1e-5,μ_init::Vector{T}=ones(T,1),verbose::Integer=0,ν::T=5.0) where {T<:Real}
    @warn deprecation_warning
    model = VGP(X,y,kernel,StudentTLikelihood(ν),AnalyticVI(ϵ=ϵ),verbose=verbose,optimizer=Autotuning,atfrequency=AutotuningFrequency,IndependentPriors=IndependentGPs)
    for i in 1:model.nLatent
        model.μ[i] = μ_init[1]*ones(length(model.μ[i]))
    end
    return model
end

function SparseStudentT(X::AbstractArray{T},y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true, Autotuning::Bool=false,OptimizeIndPoints::Bool=false, nEpochs::Integer = 10000,batchsize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100, kernel=0,m::Integer=0,AutotuningFrequency::Integer=1, ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5, verbose::Integer=0,ν::Real=5.0) where {T<:Real}
    @warn deprecation_warning
    model = SVGP(X,y,kernel,StudentTLikelihood(ν),Stochastic ? AnalyticSVI(batchsize,ϵ=ϵ,optimizer=AdaptiveLearningRate ? ALRSVI() : InverseDecay(τ=τ_s,κ=κ_s)) : AnalyticVI(ϵ=ϵ),m,verbose=verbose,optimizer=Autotuning,atfrequency=AutotuningFrequency,IndependentPriors=IndependentGPs)
    for i in 1:model.nLatent
        model.μ[i] = μ_init[1]*ones(length(model.μ[i]))
    end
    return model
end
