using ForwardDiff
using AugmentedGaussianProcesses
using AugmentedGaussianProcesses.KernelModule
using LinearAlgebra
using Distributions

N_data = 1000;N_test = 20
N_indpoints = 80; N_dim = 2
noise = 2.0
minx=-5.0; maxx=5.0
function latent(x)
    return x[:,1].*sin.(1.0*x[:,2])
end
X = rand(N_data,N_dim)*(maxx-minx).+minx
x_test = range(minx,stop=maxx,length=N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
y = sign.(latent(X)+rand(Normal(0,noise),size(X,1)))
y_test = sign.(latent(X_test))

function tracekernel(sigma::Array)
     k= RBFKernel(sigma,dim=N_dim)
     K = kernelmatrix(X,k)
     tr(A*K)
end


model = SparseXGPC(X,y,Autotuning=true,m=10,kernel=RBFKernel(ones(N_dim)))

# function ELBO(model::SparseXGPC,blah::Bool)
function ADELBO(kernelparams_and_indpoints::AbstractArray)
    kernelparams=kernelparams_and_indpoints[1:model.nDim+1]
    model.inducingPoints=reshape(kernelparams_and_indpoints[model.nDim+2:end],(model.m,model.nDim))
    # println(kernelparams)
    model.kernel = RBFKernel(kernelparams[1:model.nDim],dim=model.nDim,variance=kernelparams[end])
    model.HyperParametersUpdated =true
    AugmentedGaussianProcesses.computeMatrices!(model)
    AugmentedGaussianProcesses.ELBO(model)
end
grads = ForwardDiff.gradient(ADELBO,vcat(getlengthscales(model.kernel),getvariance(model.kernel),model.inducingPoints[:]))
vcat(getlengthscales(model.kernel),getvariance(model.kernel),model.inducingPoints[:])
# end
ADELBO([[getlengthscales(model.kernel),getvariance(model.kernel)],model.inducingPoints])
AugmentedGaussianProcesses.computeMatrices!(model)
ELBO(model)
typeof(model.kernel)
model.Kmm = Symmetric(kernelmatrix(X,model.kernel)+1.0e-7*I)
