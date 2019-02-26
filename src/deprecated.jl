### Contains all old constructors for compatibility issues

deprecation_warning = "Deprecated constructor, use VGP(X,y,kernel,likelihood,inference) or SVGP(X,y,kernel,likelihood,inference,m) in the future"

function MultiClass(X::AbstractArray{T},y::AbstractArray;Autotuning::Bool=false,nEpochs::Integer = 200, KStochastic::Bool = false, nClassesUsed::Int=0,
                                kernel=0,AutotuningFrequency::Integer=2,IndependentGPs::Bool=false,
                                ϵ::Real=T(1e-5),μ_init::Vector{T}=zeros(T,1),verbose::Integer=0) where T
    @warn deprecation_warning
    model = VGP(X,y,kernel,AugmentedLogisticSoftMaxLikelihood(),AnalyticInference(ϵ=ϵ),verbose=verbose,Autotuning=Autotuning,atfrequency=AutotuningFrequency,IndependentPriors=IndependentGPs)
    for i in 1:model.nLatent
        model.μ[i] = μ_init[1]*ones(length(model.μ[i]))
    end
    return model
end

function SparseMultiClass(X::AbstractArray{T},y::AbstractArray;Stochastic::Bool=false,KStochastic::Bool=false,nClassesUsed::Int=0,AdaptiveLearningRate::Bool=true,
                                Autotuning::Bool=false,OptimizeIndPoints::Bool=false, IndependentGPs::Bool=true,
                                nEpochs::Integer = 10000,KSize::Int64=-1,batchsize::Integer=-1,κ_s::T=T(0.51),τ_s::Integer=1,
                                kernel=0,m::Integer=0,AutotuningFrequency::Integer=2,
                                ϵ::Real=1e-5,μ_init::Vector{T}=zeros(T,1),SmoothingWindow::Integer=5,
                                verbose::Integer=0) where {T<:Real}
    @warn deprecation_warning
    model = SVGP(X,y,kernel,AugmentedLogisticSoftMaxLikelihood(),Stochastic ? StochasticAnalyticInference(batchsize,ϵ=ϵ,optimizer=AdaptiveLearningRate ? ALRSVI() : InverseDecay(τ=τ_s,κ=κ_s)) : AnalyticInference(ϵ=ϵ),m,verbose=verbose,Autotuning=Autotuning,atfrequency=AutotuningFrequency,IndependentPriors=IndependentGPs)
    for i in 1:model.nLatent
        model.μ[i] = μ_init[1]*ones(length(model.μ[i]))
    end
    return model
end
