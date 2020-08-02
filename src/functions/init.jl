## Verify that the data is self-consistent and consistent with the likelihood ##
function check_data!(
    y::AbstractArray,
    likelihood::Union{Distribution,Likelihood},
)
    y, nLatent, likelihood = treat_labels!(y, likelihood)
    return y, nLatent, likelihood
end


##
function init_Z(Z::AbstractInducingPoints, Zoptimiser)
    # if nInducingPoints isa Int
    #     @assert nInducingPoints > 0 "The number of inducing points is incorrect (negative or bigger than number of samples)"
    #     if nInducingPoints > nSamples
    #         @warn "Number of inducing points bigger than the number of points : reducing it to the number of samples: $(nSamples)"
    #         nInducingPoints = nSamples
    #     else
    #         nInducingPoints = Kmeans(nInducingPoints, nMarkov = 10)
    #     end
    # end
    # if nInducingPoints isa Int && nInducingPoints == nSamples
    #     Z = X
    # else
    #     IPModule.init!(nInducingPoints, X, y, kernel)
    #     Z = nInducingPoints.Z
    # end
    #
    if Zoptimiser isa Bool
        Zoptimiser = Zoptimiser ? ADAM(0.001) : nothing
    end
    if Z isa InducingPoints.OptimIP
        return Z
    else
        return InducingPoints.OptimIP(Z, Zoptimiser)
    end
end
