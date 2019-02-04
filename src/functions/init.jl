""" Verify that the data is self-consistent and consistent with the likelihood """
function check_data!(X::AbstractArray{T1,N1},y::AbstractArray{T2,N2},likelihood::Likelihood) where {T1<:Real,T2,N1,N2}
    @assert (size(y,1)==size(X,1)) "There is not the same number of samples in X and y";
    @assert N1 <= 2 "The input matrix X can only be a vector or a matrix"
    if N1 == 1 #Reshape a X vector as a matrix
        X = reshape(X,length(X),1);
    end
    y,likelihood = treat_labels!(y,likelihood)
    return X,y,likelihood
end

""" Verify that the likelihood and inference are compatible (are implemented) """
function check_implementation(likelihood::L,inference::I) where {I<:Inference,L<:Likelihood}
    if isa(likelihood,GaussianLikelihood)
        if isa(inference,AnalyticInference)
            return true
        else
            return false
        end
    elseif isa(likelihood,LogisticLikelihood)
        if isa(inference,AnalyticInference)
            return true
        else
            return false
        end
    if isa(likelihood,SoftMaxLikelihood)
        if isa(inference,AnalyticInference)
            return true
        # elseif isa(inference,)
            # return false
        end
    end
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
