## Verify that the data is self-consistent and consistent with the likelihood ##
function check_data!(
    X::AbstractMatrix{T₁},
    y::AbstractArray{T₂,N},
    likelihood::Union{Distribution,Likelihood},
) where {T₁<:Real,T₂,N}
    @assert (size(y, 1) == size(X, 1)) "There is not the same number of samples in X and y"
    y, nLatent, likelihood = treat_labels!(y, likelihood)
    return y, nLatent, likelihood
end


##
function wrap_X_multi(X, nTask)
    X = if X isa AbstractArray{<:Real} # Do a better recognition of what X is
        if X isa AbstractVector
            [reshape(X, :, 1)]
        elseif X isa AbstractMatrix
            [X]
        else
            throw(ErrorException("X does not have the right dimensions ($(size(X)))"))
        end
    else
        @assert length(X) == nTask "There is not the same number of input matrices as output matrices"
        @assert all(isa.(X,AbstractMatrix)) "All X should be matrices"
        X
    end
end

##
function init_Z(nInducingPoints, nSamples, X, y, kernel, Zoptimiser)
    if nInducingPoints isa Int
        @assert nInducingPoints > 0 "The number of inducing points is incorrect (negative or bigger than number of samples)"
        if nInducingPoints > nSamples
            @warn "Number of inducing points bigger than the number of points : reducing it to the number of samples: $(nSamples)"
            nInducingPoints = nSamples
        else
            nInducingPoints = Kmeans(nInducingPoints, nMarkov = 10)
        end
    end
    if nInducingPoints isa Int && nInducingPoints == nSamples
        Z = X
    else
        IPModule.init!(nInducingPoints, X, y, kernel)
        Z = nInducingPoints.Z
    end

    if isa(Zoptimiser, Bool)
        Zoptimiser = Zoptimiser ? ADAM(0.001) : nothing
    end
    return Z = FixedInducingPoints(Z, Zoptimiser)
end
