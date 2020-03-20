## Verify that the data is self-consistent and consistent with the likelihood ##
function check_data!(X::AbstractArray{T₁,N₁},y::AbstractArray{T₂,N₂},likelihood::Union{Distribution,Likelihood}) where {T₁<:Real,T₂,N₁,N₂}
    @assert (size(y,1)==size(X,1)) "There is not the same number of samples in X and y";
    @assert N₁ <= 2 "The input matrix X can only be a vector or a matrix"
    y,nLatent,likelihood = treat_labels!(y,likelihood)
    if N₁ == 1 #Reshape a X vector as a matrix
        return reshape(X,:,1),y,nLatent,likelihood
    else
        return X,y,nLatent,likelihood
    end
end
