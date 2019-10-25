### Compute the gradients using a gradient function and matrices Js ###

function compute_hyperparameter_gradient(k::Kernel,gradient_function::Function,J::Vector{<:AbstractMatrix}) where {T<:Real}
    return map(gradient_function,J)
end

function compute_hyperparameter_gradient(k::Kernel,gradient_function::Function,Jmm::Vector{<:AbstractMatrix},Jnm::Vector{<:AbstractMatrix}, Jnn::Vector{<:AbstractVector},l::Likelihood,y::AbstractVector) where T
    return map(gradient_function,Jmm,Jnm,Jnn)
end
