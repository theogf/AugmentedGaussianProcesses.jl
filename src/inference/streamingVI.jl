mutable struct StreamingVI{T<:Real,N} <: VariationalInference{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    nMinibatch::Int64 #Size of mini-batches
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    vi_opt::Ntuple{N,AVIOptimizer}
    MBIndices::Vector{Int64} #Indices of the minibatch
    xview::SubArray{T,2,Matrix{T}}
    yview::AbstractVector

    function StreamingVI{T}(ϵ::T,nIter::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,flag::Bool) where T
        return new{T}(ϵ,nIter,nSamplesUsed,MBIndices,flag)
    end
    function StreamingVI{T,1}(ϵ::T,Stochastic::Bool,nFeatures::Int,nSamples::Int,nMinibatch::Int,nLatent::Int,optimizer::Optimizer) where {T}
        vi_opts = ntuple(_->AVIOptimizer{T}(nFeatures,optimizer),nLatent)
        new{T,nLatent}(ϵ,0,Stochastic,nSamples,nMinibatch,nSamples/nMinibatch,true,vi_opts,collect(1:nMinibatch))
    end
end

function StreamingVI(nSamplesUsed::Int64=10;ϵ::T=1e-5) where {T<:Real}
    StreamingVI{Float64}(ϵ,0,nSamplesUsed,[1],true)
end

function Base.show(io::IO,inference::StreamingVI{T}) where T
    print(io,"Streaming Variational Inference")
end


"""Initialize the final version of the inference object"""
function tuple_inference(inference::StreamingVI{T},nLatent::Integer,nFeatures::Integer,nSamples::Integer,nSamplesUsed::Integer) where {T<:Real}
    inference.nSamplesUsed = nSamplesUsed
    inference.MBIndices = 1:nSamplesUsed
    return inference
end

"""Generic method for variational updates using analytical formulas"""
function variational_updates!(model::AbstractGP{T,L,StreamingVI}) where {T,L}
    #Set as old values
    println(model.inference.nIter," ",model.nFeatures)
    Kₐ = copy(model.Kmm[1])
    invKₐ = copy(model.invKmm[1])
    Zₐ = copy(model.Zalg.centers)
    invDₐ = Symmetric(-2*model.η₂[1]-invKₐ)
    Dₐ = inv(invDₐ)
    mₐ = copy(model.nFeatures)
    ŷ = vcat(model.y[1][model.inference.MBIndices],Dₐ*model.η₁[1]);
    updateZ!(model)
    ### DO STUFF WITH HYPERPARAMETERS HERE
    computeMatrices!(model)
    L = cholesky(model.Kmm[1])
    invL = inv(L)
    Kab = kernelmatrix(Zₐ,model.Zalg.centers,model.kernel[1])
    Kab[1:mₐ,1:mₐ] = Kab[1:mₐ,1:mₐ] + convert(T,Jittering())*I
    # if model.m >= 2 && model.oldm >= 2
    # display(heatmap(Kab))
    # sleep(0.1)
    # end
    Kf̂b = vcat(model.Knm[1],Kab)
    invΣŷ = Matrix(Diagonal{T}(1.0/model.likelihood.ϵ[1]*I,(model.inference.nSamplesUsed+mₐ)))
    invΣŷ[model.inference.nSamplesUsed+1:end,model.inference.nSamplesUsed+1:end] = invDₐ
    # invΣŷ = inv(Σŷ)
    A = model.invKmm[1]*Kf̂b'*invΣŷ
    # invD = inv(I + invL*Kf̂b'*invΣŷ*Kf̂b*invL+convert(T,Jittering())*I)
    model.η₂[1] = -0.5*Symmetric(model.invKmm[1]+A*Kf̂b*model.invKmm[1])
    model.Σ[1] = -0.5*inv(model.η₂[1])
    model.η₁[1] = A*ŷ
    model.μ[1] = model.Σ[1]*model.η₁[1]
end

"""Generic method for variational updates using analytical formulas"""
function variational_updates_old!(model::AbstractGP{LType,StreamingVI{T}}) where {LType<:Likelihood,T}
    #Set as old values
    println(model.inference.nIter," ",model.nFeatures)
    Kₐ = copy(model.Kmm[1])
    invKₐ = copy(model.invKmm[1])
    Zₐ = copy(model.Zalg.centers)
    invDₐ = Symmetric(-2*model.η₂[1]-invKₐ)
    mₐ = copy(model.nFeatures)
    ŷ = vcat(model.y[1][model.inference.MBIndices],Dₐ*model.η₁[1]);
    updateZ!(model)
    ### DO STUFF WITH HYPERPARAMETERS HERE
    computeMatrices!(model)
    L = cholesky(model.Kmm[1])
    invL = inv(L)
    Kab = kernelmatrix(Zₐ,model.Zalg.centers,model.kernel[1])
    Kab[1:mₐ,1:mₐ] = Kab[1:mₐ,1:mₐ] + convert(T,Jittering())*I
    # if model.m >= 2 && model.oldm >= 2
    # display(heatmap(Kab))
    # sleep(0.1)
    # end
    Kf̂b = vcat(model.Knm[1],Kab)
    invΣŷ = Matrix(Diagonal{T}(1.0/model.likelihood.ϵ[1]*I,(model.inference.nSamplesUsed+mₐ)))
    invΣŷ[model.inference.nSamplesUsed+1:end,model.inference.nSamplesUsed+1:end] = invDₐ
    # invΣŷ = inv(Σŷ)
    A = model.invKmm[1]*Kf̂b'*invΣŷ
    # invD = inv(I + invL*Kf̂b'*invΣŷ*Kf̂b*invL+convert(T,Jittering())*I)
    model.η₂[1] = -0.5*Symmetric(model.invKmm[1]+A*Kf̂b*model.invKmm[1])
    model.Σ[1] = -0.5*inv(model.η₂[1])
    model.η₁[1] = A*ŷ
    model.μ[1] = model.Σ[1]*model.η₁[1]
end
