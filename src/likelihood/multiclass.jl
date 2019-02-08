abstract type MultiClassLikelihood{T<:Real} <: Likelihood{T} end

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractArray{T,N},likelihood::L) where {T,N,L<:MultiClassLikelihood}
    @assert N <= 1 "Target should be a vector of labels"
    likelihood = init_multiclass_likelihood(likelihood,y)
end

function init_multiclass_likelihood(likelihood::L,y::AbstractVector) where {L<:MultiClassLikelihood}
    L(one_of_K_mapping(y)...)
end

""" Given the labels, return one hot encoding, and the mapping of each class """
function one_of_K_mapping(y)
    y_values = unique(y)
    Y = [spzeros(length(y)) for i in 1:length(y_values)]
    y_class = zeros(Int64,length(y))
    for i in 1:length(y)
        for j in 1:length(y_values)
            if y[i]==y_values[j]
                Y[j][i] = 1;
                y_class[i] = j;
                break;
            end
        end
    end
    ind_values = Dict(value => key for (key,value) in enumerate(y_values))
    return Y,y_values,ind_values,y_class
end


include("softmax.jl")
include("logisticsoftmax.jl")
