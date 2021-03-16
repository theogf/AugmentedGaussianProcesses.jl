Zygote.@adjoint function /(A::AbstractVecOrMat, B::Cholesky)
  Y, back = Zygote.pullback((A, U)->(A / U) / U', A, B.U)
  return Y, function(Ȳ)
    A̅, B̅_factors = back(Ȳ)
    return (A̅, (uplo=nothing, status=nothing, factors=B̅_factors))
  end
end

Zygote.@adjoint function StatsFuns.softmax(x)
    y = StatsFuns.softmax(x)
    function softmax_pullback(Δ)
      out = Δ .* y
      return (out .= out .- y .* sum(out), )
    end
    return y, softmax_pullback
end

# Zygote.@adjoint function binomial(n, k)
#   y = binomial(n, k)
#   return y, function(Δ) begin
#     (Zygote.NO_FIELDS, )    
#   end
# end