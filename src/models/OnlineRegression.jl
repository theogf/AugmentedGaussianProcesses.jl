
#Online Gaussian Process Regression

mutable struct OnlineGPRegression{T<:Real} <: OnlineGPModel{T}
    @commonfields
    @functionfields
    @stochasticfields
    @kernelfields
    @gaussianparametersfields
    @onlinefields
    noise::T
    function OnlineGPRegression(X::AbstractArray{T},y::AbstractArray;kmeansalg::KMeansAlg=StreamOnline(),Sequential::Bool=false,AdaptiveLearningRate::Bool=false,
                                    Autotuning::Bool=false,OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,batchsize::Integer=-1,κ_s::T=1.0,τ_s::Integer=100,
                                    kernel=0,noise::T=1e-3,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Vector{T}=ones(T,1),SmoothingWindow::Integer=5,
                                    verbose::Integer=0) where {T<:Real}
            this = new{T}();
            this.ModelType = Regression;
            this.Name = "Online Sparse Gaussian Process Regression";
            initCommon!(this,X,y,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency);
            this.nSamplesUsed = batchsize
            initKernel!(this,kernel);
            initFunctions!(this);
            initOnline!(this,kmeansalg,Sequential,m)
            initStochastic!(this,AdaptiveLearningRate,batchsize,κ_s,τ_s,SmoothingWindow);
            initGaussian!(this,μ_init);
            this.noise=noise
            return this;
    end
end
