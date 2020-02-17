function ∇L_ρ(f,gp,X)
    Zygote.gradient(()->_∇L_ρ(f,gp.kernel,first(gp.σ_k),X),params(gp.kernel))
end

_∇L_ρ(f,kernel,σ,X) = f(σ*kernelmatrix(kernel,X,obsdim=1))

function ∇L_ρ(f,gp,X,∇E_μ,∇E_Σ,i,opt)
    Zygote.gradient(()->_∇L_ρ(f,gp.kernel,first(gp.σ_k),gp.Z.Z,X,∇E_μ,∇E_Σ,i,opt),params(gp.kernel))
end

function _∇L_ρ(f,kernel,σ,Z,X,∇E_μ,∇E_Σ,i,opt)
    Kmm = σ*kernelmatrix(kernel,Z,obsdim=1)
    Knm = σ*kernelmatrix(kernel,X,Z,obsdim=1)
    Knn = σ*diag(kernelmatrix(kernel,X,obsdim=1)) # TO FIX ONCE Zygote#429 is fixed.
    f(Kmm,Knm,Knn,∇E_μ,∇E_Σ,i,opt)
end
