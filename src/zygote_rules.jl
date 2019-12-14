Zygote.refresh()

function ∇L_ρ(gp,X,f)
    Zygote.gradient(()->_∇L_ρ(gp.kernel,X,f),Flux.params(gp.kernel))
end

_∇L_ρ(kernel,X,f) = f(kernelmatrix(kernel,X,obsdim=1))

# Zygote.@adjoint function _∇L_ρ(K,f)
#     f(K), J->(begin @show J;tr(J); end,nothing)
# end
