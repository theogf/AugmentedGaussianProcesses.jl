#Module for either generating data or exporting from an existing dataset

module DataAccess

using Distributions

export generate_uniform_data, generate_normal_data, generate_two_multivariate_data
export get_Dataset


function get_Dataset(datasetname::String)
    data = readdlm("../data/"*datasetname)
    X = data[:,1:end-1]; y = data[:,end];
    return (X,y,datasetname)
end

function generate_uniform_data(nFeatures_::Int64,nSamples_::Int64; range::Float64 = 1.0, noise::Float64 = 0.3)
    X = rand(Uniform(-range,range),(nSamples,nFeatures))
    β_true = rand(Normal(0,1),nFeatures)
    y = sign(X*β_true+rand(Normal(0,noise),nSamples))
    accuracy= 1-countnz(y-sign(X*β_true))/nSamples
    return (X,y,"Linearly Separated Synthetic Uniform Data",β_true,accuracy)
end

function generate_two_multivariate_data(nFeatures_::Int64,nSamples_::Int64; sep_::Float64 = 2.0, y_neg_::Bool = true, noise_::Float64 = 1.0)
    X = randn(MersenneTwister(seed),(nSamples,nFeatures))
    #Separate the normal into two
    X[1:nSamples÷2,:] += sep
    X[nSamples÷2+1:end,:] -= sep
    y = zeros((nSamples,1))
    y[1:nSamples÷2] += 1
    β_true[:] = sep
    return (X,y,"Two synthetic normal distributions",β_true)
end

function generate_normal_data(nFeatures_::Int64,nSamples_::Int64; sep_::Float64 = 2.0, noise::Float64 = 0.3, σ::Float64 = 1.0)
    X = rand(IsoNormal(zeros(nFeatures),PDMats.ScalMat(nFeatures,σ)),nSamples)'
    β_true = rand(Normal(0,1),nFeatures)
    y = sign(X*β_true+rand(Normal(0,noise),nSamples))
    accuracy= 1-countnz(y-sign(X*β_true))/nSamples
    return (X,y,"Linearly Separated Synthetic Normal Data",β_true,accuracy)
end
end #end of module
