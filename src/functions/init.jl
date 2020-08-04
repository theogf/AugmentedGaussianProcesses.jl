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
    if Zoptimiser isa Bool
        Zoptimiser = Zoptimiser ? ADAM(1e-3) : nothing
    end
    if Z isa OptimIP
        return Z
    else
        return OptimIP(Z, Zoptimiser)
    end
end
