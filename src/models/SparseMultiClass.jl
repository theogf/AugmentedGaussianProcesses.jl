
#Batch Xtreme Gaussian Process Classifier (no inducing points)

mutable struct SparseMultiClass <: MultiClassGPModel
    @commonfields
    @functionfields
    @multiclassfields
    @multiclassstochasticfields
    @multiclasskernelfields
    @multiclass_sparsefields
    function SparseMultiClass(X::AbstractArray,y::AbstractArray;Stochastic::Bool=false,KStochastic::Bool=false,nClassesUsed::Int=0,AdaptiveLearningRate::Bool=true,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(α=0.1),OptimizeIndPoints::Bool=false, IndependentGPs::Bool=true,
                                    nEpochs::Integer = 10000,KSize::Int64=-1,batchsize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100,
                                    kernel=0,noise::Real=1e-3,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,
                                    VerboseLevel::Integer=0)
            Y,y_map,ind_map,y_class = one_of_K_mapping(y)
            this = new()
            this.ModelType = MultiClassGP
            this.Name = "Sparse MultiClass Gaussian Process Classifier"
            initCommon!(this,X,y,noise,ϵ,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            initMultiClass!(this,Y,y_class,y_map,ind_map,KStochastic,nClassesUsed);
            initMultiClassKernel!(this,kernel,IndependentGPs);
            if this.VerboseLevel > 2
                println("$(now()): Classes data treated")
            end
            if Stochastic
                initMultiClassStochastic!(this,AdaptiveLearningRate,batchsize,κ_s,τ_s,SmoothingWindow);
            else
                this.MBIndices = collect(1:this.nSamples); this.nSamplesUsed = this.nSamples; this.StochCoeff=1.0; this.ρ_s=ones(Float64,this.K)
            end
            initMultiClassSparse!(this,m,OptimizeIndPoints)
            initMultiClassVariables!(this,μ_init)
            return this;
    end
end
