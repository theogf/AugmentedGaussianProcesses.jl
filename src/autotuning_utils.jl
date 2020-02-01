
### Global constant allowing to chose between forward_diff and reverse_diff for hyperparameter optimization ###
const ADBACKEND = Ref(:forward_diff)

function setadbackend(backend_sym)
    @assert backend_sym == :forward_diff || backend_sym == :reverse_diff
    ADBACKEND[] = backend_sym
end

### To be replaced later by a self method of KernelFunctions ###
for k in (SqExponentialKernel,Matern32Kernel,LinearKernel,KernelSum,KernelProduct)
    Flux.@functor(k)
end

for t in (ARDTransform,ScaleTransform,LowRankTransform)
    Flux.@functor(t)
end

##
function apply_grads_kernel_params!(opt,k::Kernel,Δ::IdDict)
    ps = Flux.params(k)
    for p in ps
      Δ[p] == nothing && continue
      Δlogp = Flux.Optimise.apply!(opt, p, p.*vec(Δ[p]))
      p .= exp.(log.(p).+Δlogp)
    end
end

function apply_grads_kernel_variance!(opt,gp::Abstract_GP,grad::Real)
    Δlogσ = Flux.Optimise.apply!(opt,gp.σ_k,gp.σ_k.*[grad])
    gp.σ_k .= exp.(log.(gp.σ_k).+Δlogσ)
end

function apply_gradients_mean_prior!(opt,μ::PriorMean,g::AbstractVector,X::AbstractMatrix)
    update!(opt,μ,g,X)
end

# Kernel Product
recursive_hadamard(A::AbstractMatrix,V::AbstractVector) = recursive_hadamard.([A],V)
recursive_hadamard(A::AbstractMatrix,V::AbstractMatrix) = hadamard(A,V)
recursive_hadamard(A::AbstractVector,V::AbstractVector) = recursive_hadamard.([A],V)
recursive_hadamard(A::AbstractVector,V::AbstractVector{<:Real}) = hadamard(A,V)

function indpoint_derivative(kernel::Kernel,Z::InducingPoints)
    reshape(ForwardDiff.jacobian(x->kernelmatrix(kernel,x,obsdim=1),Z),size(Z,1),size(Z,1),size(Z,1),size(Z,2))
end

function indpoint_derivative(kernel::Kernel,X,Z::InducingPoints)
    reshape(ForwardDiff.jacobian(x->kernelmatrix(kernel,X,x,obsdim=1),Z),size(X,1),size(Z,1),size(Z,1),size(Z,2))
end


function ELBO_given_theta(model)
    model.inference.HyperParametersUpdated = true
    computeMatrices!(model)
    ELBO(model)
end
