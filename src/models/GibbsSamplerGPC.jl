"Efficient Gaussian Process Classifier with Inference done via Gibbs Sampling"
mutable struct GibbsSamplerGPC{T<:Real} <: FullBatchModel{T}
    @commonfields
    @functionfields
    @latentfields
    @gaussianparametersfields

    @kernelfields
    @samplingfields
    "GibbsSamplerGPC Constructor"
    function GibbsSamplerGPC(X::AbstractArray{T},y::AbstractArray;burninsamples::Integer = 200, samplefrequency::Integer=100,
                                    Autotuning::Bool=false,nEpochs::Integer = 200,
                                    kernel=0,AutotuningFrequency::Integer=10,
                                    ϵ::T=1e-5,μ_init::Vector{T}=ones(T,1),verbose::Integer=0) where {T<:Real}
            this = new{T}(X,y)
            this.ModelType = XGPC
            this.Name = "Polya-Gamma Gaussian Process Classifier by sampling"
            initCommon!(this,X,y,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency);
            initFunctions!(this);
            initKernel!(this,kernel);
            initGaussian!(this,μ_init);
            initLatentVariables!(this);
            initSampling!(this,burninsamples,samplefrequency)
            return this;
    end

end

"Compute one sample of all parameters via Gibbs Sampling"
function updateParameters!(model::GibbsSamplerGPC{T},iter::Integer) where T
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
