
#Batch Xtreme Gaussian Process Classifier (no inducing points)

mutable struct MultiClass <: MultiClassGPModel
    @commonfields
    @functionfields
    @multiclassfields
    @multiclasskernelfields
    function MultiClass(X::AbstractArray,y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),nEpochs::Integer = 200, KStochastic::Bool = false, nClassesUsed::Int=0,
                                    kernel=0,noise::Float64=1e-3,AutotuningFrequency::Integer=2,IndependentGPs::Bool=false,
                                    ϵ::Float64=1e-5,μ_init::Array{Float64,1}=[0.0],VerboseLevel::Integer=0)
            Y,y_map,ind_map,y_class = one_of_K_mapping(y)
            this = new()
            this.ModelType = MultiClassGP
            this.Name = "MultiClass Gaussian Process Classifier"
            initCommon!(this,X,y,noise,ϵ,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            initMultiClass!(this,Y,y_class,y_map,ind_map,KStochastic,nClassesUsed);
            initMultiClassKernel!(this,kernel,IndependentGPs);
            this.Knn = [Matrix{Float64}(undef,this.nSamples,this.nSamples) for i in 1:this.K]
            this.invK = [Matrix{Float64}(undef,this.nSamples,this.nSamples) for i in 1:this.K]
            initMultiClassVariables!(this,μ_init)
            return this;
    end
end
