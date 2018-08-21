"Efficient Gaussian Process Classifier with Inference done via Gibbs Sampling"
mutable struct GibbsSamplerGPC <: FullBatchModel
    @commonfields
    @functionfields
    @latentfields
    @gaussianparametersfields

    @kernelfields
    @samplingfields
    "GibbsSamplerGPC Constructor"
    function GibbsSamplerGPC(X::AbstractArray,y::AbstractArray;burninsamples::Integer = 200, samplefrequency::Integer=100,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(),nEpochs::Integer = 200,
                                    kernel=0,noise::Float64=1e-3,AutotuningFrequency::Integer=10,
                                    ϵ::Float64=1e-5,μ_init::Array{Float64,1}=[0.0],VerboseLevel::Integer=0)
            this = new(X,y)
            this.ModelType = XGPC
            this.Name = "Polya-Gamma Gaussian Process Classifier by sampling"
            initCommon!(this,X,y,noise,ϵ,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            initKernel!(this,kernel);
            initGaussian!(this,μ_init);
            initLatentVariables!(this);
            initSampling!(this,burninsamples,samplefrequency)
            return this;
    end

end

"Compute one sample of all parameters via Gibbs Sampling"
function updateParameters!(model::GibbsSamplerGPC,iter::Integer)
    computeMatrices!(model)
    model.α = broadcast(model.pgsampler.draw,1.0,model.μ) #Sample from a polya-gamm distribution
    push!(model.samplehistory_α,model.α)
    C = Matrix(Symmetric(inv(diagm(model.α)+model.invK),:U))
    model.μ = rand(MvNormal(0.5*C*model.y,C))
    push!(model.samplehistory_f,model.μ)
    if iter > model.burninsamples && (iter-model.burninsamples)%model.samplefrequency==0
        push!(model.estimate,model.μ)
    end
end
