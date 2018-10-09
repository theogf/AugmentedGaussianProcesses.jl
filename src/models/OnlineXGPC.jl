
#Online Gaussian Process Classifier

mutable struct OnlineXGPC <: OnlineGPModel
    @commonfields
    @functionfields
    @latentfields
    @stochasticfields
    @kernelfields
    @gaussianparametersfields
    @onlinefields

    function OnlineXGPC(X::AbstractArray,y::AbstractArray;kmeansalg::KMeansAlg=StreamOnline(),Sequential::Bool=false,AdaptiveLearningRate::Bool=false,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(α=0.1),OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,batchsize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100,
                                    kernel=0,noise::Real=1e-3,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,
                                    verbose::Integer=0)
            this = new();
            this.ModelType = XGPC;
            this.Name = "Online Sparse Gaussian Process Classification";
            initCommon!(this,X,y,noise,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
            initOnline!(this,kmeansalg,Sequential,m)
            initFunctions!(this);
            initStochastic!(this,AdaptiveLearningRate,batchsize,κ_s,τ_s,SmoothingWindow);
            initGaussian!(this,μ_init);
            initKernel!(this,kernel); this.nFeatures = this.m
            initLatentVariables!(this)
            return this;
    end
end
