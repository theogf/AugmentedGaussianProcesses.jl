Zygote.refresh()

function ∇L_ρ(gp,X,f)
    K = kernelmatrix(gp.kernel,X,obsdim=1)
    Zygote.forwarddiff()
    Zygote.@showgrad K
    _∇L_ρ(K)
end

_∇L_ρ(K) = tr(K)

Zygote.@adjoint function _∇L_ρ(K)
    tr(K), J->(begin @show J;tr(J); end,nothing)
end
