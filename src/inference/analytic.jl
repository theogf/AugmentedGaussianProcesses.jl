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
