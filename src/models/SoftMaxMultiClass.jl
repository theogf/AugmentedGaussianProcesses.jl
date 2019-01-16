
#Batch Xtreme Gaussian Process Classifier (no inducing points)

mutable struct SoftMaxMultiClass{T} <: MultiClassGPModel{T}
    @commonfields
    @functionfields
    @multiclassfields
    @multiclasskernelfields
    μ_optimizer::Vector
    Σ_optimizer::Vector
    L::Vector{AbstractArray{T}}
    function SoftMaxMultiClass(X::AbstractArray{T},y::AbstractArray;Autotuning::Bool=false,optimizer=0.01,nEpochs::Integer = 200, KStochastic::Bool = false, nClassesUsed::Int=0,
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
            this.L = [LowerTriangular(Diagonal(one(T)*I,this.nFeatures)) for _ in 1:this.K]
            this.μ_optimizer = [Adam(α=optimizer) for _ in 1:this.K]
            this.Σ_optimizer = [Adam(α=optimizer) for _ in 1:this.K]
            return this;
    end
end
