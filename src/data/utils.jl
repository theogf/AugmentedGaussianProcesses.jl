view_x(d::AbstractDataContainer, indices) = view(d.X, indices)
view_y(::AbstractLikelihood, y::AbstractVector, i::AbstractVector) = view(y, i)
function view_y(l::AbstractLikelihood, d::AbstractDataContainer, i::AbstractVector)
    return view_y(l, output(d), i)
end
function view_y(l::AbstractLikelihood, d::MODataContainer, i::AbstractVector)
    return view_y.(l, output(d), Ref(i))
end
function view_y(
    l::Tuple{Vararg{<:AbstractLikelihood}}, d::MODataContainer, i::AbstractVector
)
    return view_y.(l, output(d), Ref(i))
end

# Verify that the data is self-consistent and consistent with the likelihood ##
function check_data!(y::AbstractArray, likelihood::Union{Distribution,AbstractLikelihood})
    return treat_labels!(y, likelihood)
end

function wrap_data(X, y, likelihood::AbstractLikelihood)
    y = check_data!(y, likelihood)
    return wrap_data(X, y)
end

function wrap_data(X, y, likelihoods::Tuple{Vararg{<:AbstractLikelihood}})
    ys = map(check_data!, y, likelihoods)
    return wrap_modata(X, ys)
end
