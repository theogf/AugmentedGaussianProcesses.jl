function ChainRulesCore.rrule(::typeof(Base.:(/)), A::AbstractVecOrMat, B::Cholesky)
  Y, back = Zygote.pullback((A, U)->(A / U) / U', A, B.U)
  function rdiv_callback(Ȳ)
    A̅, B̅_factor = back(Ȳ)
    return (NO_FIELDS, A̅, (uplo=DoesNotExist(), info=DoesNotExist(), factors=B̅_factor))
  end
  return Y, rdiv_callback
end

function ChainRulesCore.rrule(::typeof(StatsFuns.softmax), x)
    y = StatsFuns.softmax(x)
    function softmax_pullback(Δ)
      out = Δ .* y
      return (NO_FIELDS, out .-= y .* sum(out))
    end
    return y, softmax_pullback
end

# Zygote.@adjoint function binomial(n, k)
#   y = binomial(n, k)
#   return y, function(Δ) begin
#     (Zygote.NO_FIELDS, )    
#   end
# end