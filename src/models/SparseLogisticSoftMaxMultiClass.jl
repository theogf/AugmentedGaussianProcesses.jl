
#Batch Xtreme Gaussian Process Classifier (no inducing points)

mutable struct SparseLogisticSoftMaxMultiClass{T<:Real} <: MultiClassGPModel{T}
    @commonfields
    @functionfields
    @multiclassfields
    @multiclassstochasticfields
    @multiclasskernelfields
    @multiclass_sparsefields
    μ_optimizer::Vector
    Σ_optimizer::Vector
    grad_μ::Vector{AbstractVector}
    grad_Σ::Vector{AbstractVector}
    L::Vector{AbstractArray{T}}
    varMCMC::Vector{AbstractMatrix}
    function SparseLogisticSoftMaxMultiClass(X::AbstractArray{T},y::AbstractArray;Stochastic::Bool=false,KStochastic::Bool=false,nClassesUsed::Int=0,AdaptiveLearningRate::Bool=true,
                                    Autotuning::Bool=false,OptimizeIndPoints::Bool=false, IndependentGPs::Bool=true,
                                    nEpochs::Integer = 10000,KSize::Int64=-1,batchsize::Integer=-1,κ_s::T=T(0.51),τ_s::Integer=1,optimizer::Real=0.01,
                                    kernel=0,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Vector{T}=zeros(T,1),SmoothingWindow::Integer=5,
                                    verbose::Integer=0) where {T<:Real}
            Y,y_map,ind_map,y_class = one_of_K_mapping(y)
            this = new{T}()
            this.ModelType = MultiClassGP
            this.Name = "Sparse MultiClass Gaussian Process Classifier"
            initCommon!(this,X,y,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency);
            initFunctions!(this);
            initMultiClass!(this,Y,y_class,y_map,ind_map,KStochastic,nClassesUsed);
            initMultiClassKernel!(this,kernel,IndependentGPs);
            if this.verbose > 2
                println("$(now()): Classes data treated")
            end
            if Stochastic
                initMultiClassStochastic!(this,AdaptiveLearningRate,batchsize,κ_s,τ_s,SmoothingWindow);
            else
                this.MBIndices = collect(1:this.nSamples); this.nSamplesUsed = this.nSamples; this.StochCoeff=1.0; this.ρ_s=ones(Float64,this.K)
            end
            initMultiClassSparse!(this,m,OptimizeIndPoints)
            initMultiClassVariables!(this,μ_init)
            if this.Stochastic && this.AdaptiveLearningRate
                MCInit!(this)
            end
            this.L = [LowerTriangular(Diagonal(one(T)*I,this.nFeatures)) for _ in 1:this.K]
            this.μ_optimizer = [Adam(α=optimizer) for _ in 1:this.K]
            this.Σ_optimizer = [Adam(α=optimizer) for _ in 1:this.K]
            this.grad_μ = [zeros(T,this.nSamplesUsed) for _ in 1:this.K]
            this.grad_Σ = [zeros(T,this.nSamplesUsed) for _ in 1:this.K]
            this.varMCMC = Vector{AbstractMatrix}()
            return this;
    end
end
