view_x(d::DataContainer, indices) = view(d.X, indices)
view_y(l::Likelihood, y::AbstractVector, i::AbstractVector) = view(y, i)
view_y(l::Likelihood, d::DataContainer, i::AbstractVector) = view_y(l, d.y, i)
