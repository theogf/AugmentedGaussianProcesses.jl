"""Sparse Xtreme Gaussian Process Classifier
Create a GP model taking the  training data and labels X & y as required arguments. Other optional arguments are:
- Stochastic::Bool : Is the method trained via mini batches
- AdaptiveLearningRate::Bool : Is the learning rate adapted via estimation of the gradient variance? see "Adaptive Learning Rate for Stochastic Variational inference" https://pdfs.semanticscholar.org/9903/e08557f328d58e4ba7fce68faee380d30b12.pdf, if not use simple exponential decay with parameters κ_s and τ_s seen under (1/(iter+τ_s))^-κ_s
- Autotuning::Bool : Are the hyperparameters trained as well
- optimizer::Optimizer : Type of optimizer for the hyperparameters
- OptimizeIndPoints::Bool : Is the location of inducing points optimized
- nEpochs::Integer : How many iteration steps
- batchsize::Integer : number of samples per minibatches
- κ_s::Real
- τ_s::Real
- kernel::Kernel : Kernel for the model
- noise::Float64 : noise added in the model
- m::Integer : Number of inducing points
- ϵ::Float64 : minimum value for convergence
- SmoothingWindow::Integer : Window size for averaging convergence in the stochastic case
- verbose::Integer : How much information is displayed (from 0 to 3)
"""
mutable struct SparseXGPC <: SparseModel
    @commonfields
    @functionfields
    @stochasticfields
    @kernelfields
    @sparsefields
    @gaussianparametersfields
    c::Vector{Float64}
    θ::Vector{Float64}
    "SparseXGPC Constructor"
    function SparseXGPC(X::AbstractArray,y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(α=0.1),OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,batchsize::Integer=-1,κ_s::Float64=1.0,τ_s::Integer=100,
                                    kernel=0,noise::Real=1e-3,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,
                                    verbose::Integer=0)
            this = new();
            this.ModelType = XGPC;
            this.Name = "Polya-Gamma Sparse Gaussian Process Classifier";
            initCommon!(this,X,y,noise,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            if Stochastic
                initStochastic!(this,AdaptiveLearningRate,batchsize,κ_s,τ_s,SmoothingWindow);
            else
                this.MBIndices = 1:this.nSamples; this.nSamplesUsed = this.nSamples; this.StochCoeff=1.0;
            end
            initKernel!(this,kernel);
            initSparse!(this,m,OptimizeIndPoints);
            initGaussian!(this,μ_init);
            this.c = abs.(rand(this.nSamplesUsed))*2;
            this.θ = zero(this.c)
            if this.Stochastic && this.AdaptiveLearningRate
                MCInit!(this)
            end
            return this;
    end
    function SparseXGPC()
        this = new()
        this.ModelType = XGPC
        this.Name = "Polya-Gamma Sparse Gaussian Process Classifier";
        initFunctions!(this)
        return this;
    end
end
