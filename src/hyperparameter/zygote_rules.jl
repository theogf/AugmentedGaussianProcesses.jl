Zygote.@adjoint function /(A::AbstractVecOrMat, B::Cholesky)
  Y, back = Zygote.pullback((A, U)->(A / U) / U', A, B.U)
  return Y, function(Ȳ)
    A̅, B̅_factors = back(Ȳ)
    return (A̅, (uplo=nothing, status=nothing, factors=B̅_factors))
  end
end