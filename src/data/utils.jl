view_x(d::AbstractDataContainer, indices) = view(d.X, indices)
view_y(::AbstractLikelihood, y::AbstractVector, i::AbstractVector) = view(y, i)
view_y(l::AbstractLikelihood, d::AbstractDataContainer, i::AbstractVector) = view_y(l, output(d), i)
view_y(l::AbstractLikelihood, d::MODataContainer, i::AbstractVector) = view_y.(l, output(d), Ref(i))
view_y(l::AbstractVector{<:AbstractLikelihood}, d::MODataContainer, i::AbstractVector) = view_y.(l, output(d), Ref(i))


# Verify that the data is self-consistent and consistent with the likelihood ##
function check_data!(
    y::AbstractArray,
    likelihood::Union{Distribution,AbstractLikelihood},
)
    y, nLatent, likelihood = treat_labels!(y, likelihood)
    return y, nLatent, likelihood
end


# Transform Z as an OptimIP if it's not the case already
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