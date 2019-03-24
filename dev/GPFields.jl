@def onlinefields begin
    Sequential::Bool #Defines if we know how many point will be treated at the beginning
    alldataparsed::Bool #Check if all data has been treated
    lastindex::Int64
    kmeansalg::KMeansAlg # Online KMean algorithm
    indpoints_updated::Bool#Trigger for matrix computations
    m::Int64 #Number of wanted inducing points
    Kmm::Matrix{T} #Kernel matrix
    invKmm::Matrix{T} #Inverse Kernel matrix of inducing points
    Ktilde::Vector{T} #Diagonal of the covariance matrix between inducing points and generative points
    κ::Matrix{T} #Kmn*invKmm
end

"""
Function for initiating online parameters
"""
function initOnline!(model::GPModel{T},alg::KMeansAlg,Sequential::Bool,m::Int64) where T
    model.m = m
    model.kmeansalg = alg
    model.Sequential = Sequential
    model.alldataparsed = false
    model.lastindex=1
    if Sequential
        if typeof(alg) <: StreamOnline || typeof(alg) <: DataSelection
            # newbatchsize = min(max(15,floor(Int64,(model.m-15)/5.0))-1,model.nSamples-model.lastindex)
            newbatchsize = min(model.nSamplesUsed-1,model.nSamples-model.lastindex)
            model.MBIndices = model.lastindex:(model.lastindex+newbatchsize)
            init!(model.kmeansalg,model.X[model.MBIndices,:],model.y[model.MBIndices],model,model.m)
        else
            @assert model.nSamples >= model.m
            newbatchsize = min(model.m-1,model.nSamples-model.lastindex)
            model.MBIndices = model.lastindex:(model.lastindex+newbatchsize)
            init!(model.kmeansalg,model.X[model.MBIndices,:],model.y[model.MBIndices],model,model.m)
        end
    else
        model.MBIndices = StatsBase.sample(1:model.nSamples,model.m,replace=false) #Sample nSamplesUsed indices for the minibatches
        init!(model.kmeansalg,model.X,model.y,model,model.m)
    end
    model.m = model.kmeansalg.k
    model.nFeatures = model.m
    model.indpoints_updated = true
end
