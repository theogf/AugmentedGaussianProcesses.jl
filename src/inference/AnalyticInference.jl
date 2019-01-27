""" Solve conjugate or conditionally conjugate likelihoods (especially valid for augmented likelihoods) """

struct AnalyticInference <: Inference
    ϵ::Real #Convergence criteria
    nIter::Integer #Number of steps performed
    λ::LearningRate #Learning rate for stochastic updates
    ∇η₁::AbstractVector{AbstractVector}
    ∇η₂::AbstractVector{AbstractArray}
    ρ::Real #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
end

function variational_updates!(model::VGP{L,AnalyticInference}) where {L<:Likelihood}
    local_updates!(model)
    model.η₁,model.η₂ = natural_gradient(model)
end

function variational_updates(model::SVGP{L,AnalyticInference}) where {L<:Likelihood}
    local_updates!(model)
    natural_gradient!(model)
    computeLearningRate(model)
    global_update!(model)
end
