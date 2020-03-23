## Verify that the data is self-consistent and consistent with the likelihood ##
function check_data!(X::AbstractMatrix{T₁},y::AbstractArray{T₂,N},likelihood::Union{Distribution,Likelihood}) where {T₁<:Real,T₂,N}
    @assert (size(y,1)==size(X,1)) "There is not the same number of samples in X and y";
    y,nLatent,likelihood = treat_labels!(y,likelihood)
    return y,nLatent,likelihood
end
