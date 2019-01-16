
#Batch Xtreme Gaussian Process Classifier (no inducing points)

mutable struct SoftMaxMultiClass{T<:Real} <: MultiClassGPModel{T}
    @commonfields
    @functionfields
    @multiclassfields
    @multiclasskernelfields
    optimizer::Optimizer
    function MultiClass(X::AbstractArray{T},y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),nEpochs::Integer = 200, KStochastic::Bool = false, nClassesUsed::Int=0,
                                    kernel=0,AutotuningFrequency::Integer=2,IndependentGPs::Bool=false,
                                    ϵ::Real=T(1e-5),μ_init::Vector{T}=zeros(T,1),verbose::Integer=0) where {T<:Real}
            Y,y_map,ind_map,y_class = one_of_K_mapping(y)
            this = new{T}()
            this.ModelType = MultiClassGP
            this.Name = "SoftMax MultiClass Gaussian Process Classifier"
            initCommon!(this,X,y,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency);
            initFunctions!(this);
            initMultiClass!(this,Y,y_class,y_map,ind_map,KStochastic,nClassesUsed);
            initMultiClassKernel!(this,kernel,IndependentGPs);
            this.Knn = [Symmetric(Matrix{T}(undef,this.nSamples,this.nSamples)) for i in 1:this.K]
            this.invK = [Symmetric(Matrix{T}(undef,this.nSamples,this.nSamples)) for i in 1:this.K]
            initMultiClassVariables!(this,μ_init)
            this.optimizer = optimizer
            return this;
    end
end
