"Student T Gaussian Process Regression"
mutable struct BatchStudentT <: FullBatchModel
    @commonfields
    @functionfields
    @gaussianparametersfields
    @kernelfields
    ν::Float64
    α::Float64
    β::Vector{Float64}
    θ::Vector{Float64}
    "BatchStudentT Constructor"
    function BatchStudentT(X::AbstractArray,y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),
                                    nEpochs::Integer = 200,
                                    kernel=0,noise::Float64=1e-3,AutotuningFrequency::Integer=1,
                                    ϵ::Float64=1e-5,μ_init::Array{Float64,1}=[0.0],verbose::Integer=0,ν::Float64=5.0)
            this = new(X,y)
            this.ModelType = StudentT
            this.Name = "Student T Gaussian Process Regression"
            initCommon!(this,X,y,noise,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            initKernel!(this,kernel);
            initGaussian!(this,μ_init);
            this.ν = ν
            this.α = (this.ν+1.0)/2.0
            this.β = abs.(rand(this.nSamples))*2;
            this.θ = zero(this.β)
            return this;
    end
end
