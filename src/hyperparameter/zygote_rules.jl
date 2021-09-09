function ChainRulesCore.rrule(::typeof(Base.:(/)), A::AbstractVecOrMat, B::Cholesky)
    Y, back = Zygote.pullback((A, U) -> (A / U) / U', A, B.U)
    function rdiv_callback(Ȳ)
        A̅, B̅_factor = back(Ȳ)
        return (NoTangent(), A̅, (uplo=NoTangent(), info=NoTangent(), factors=B̅_factor))
    end
    return Y, rdiv_callback
end

function ChainRulesCore.rrule(::typeof(StatsFuns.softmax), x)
    y = StatsFuns.softmax(x)
    function softmax_pullback(Δ)
        out = Δ .* y
        return (NoTangent(), out .-= y .* sum(out))
    end
    return y, softmax_pullback
end
