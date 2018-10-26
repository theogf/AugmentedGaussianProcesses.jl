
#Batch Gaussian Process Regression (no inducing points)

mutable struct GPRegression <: FullBatchModel
    @commonfields
    @functionfields
    @kernelfields
    function GPRegression(X::AbstractArray,y::AbstractArray;Autotuning::Bool=false,optimizer::Optimizer=Adam(),nEpochs::Integer = 10,
                                    kernel=0,noise::Float64=1e-3,verbose::Integer=0)
            this = new(X,y)
            this.ModelType = Regression
            this.Name = "Gaussian Process Regression with Gaussian Likelihood"
            initCommon!(this,X,y,noise,1e-16,nEpochs,verbose,Autotuning,1,optimizer);
            initFunctions!(this);
            initKernel!(this,kernel);
            return this;
    end

end
