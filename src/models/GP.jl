abstract type GP{L<:Likelihood,I<:Inference,T<:Real,V<:AbstractArray{T}} end

function Random.rand!(model::GP,A::DenseArray{T},X::AbstractArray{T}) where {T<:Real}
    rand!(MvNormal(predict_f(model,X,covf=true,fullcov=true)...),A)
end

function Random.rand(model::GP,X::AbstractArray{T},n::Int=1) where {T<:Real}
    rand!(model,Array{T}(undef,size(X,1),n),X)
end
