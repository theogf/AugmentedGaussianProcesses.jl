using AugmentedGaussianProcesses.KernelModule
using Profile, ProfileView, BenchmarkTools, Test
using Random: seed!
using MLKernels, Statistics
using ForwardDiff
seed!(42)
dims= 2
A = rand(1000,dims)
B = rand(100,dims)
#Compare kernel results with MLKernels package

#RBF Kernel
θ = 0.1
mlk = MLKernels.SquaredExponentialKernel(0.5/θ^2)
agpk = KernelModule.SEKernel(θ)
mlK = MLKernels.kernelmatrix(mlk,A)
agpK = KernelModule.kernelmatrix(A,agpk)
mlKab = MLKernels.kernelmatrix(mlk,A,B)
agpKab = KernelModule.kernelmatrix(A,B,agpk)

@test sum(abs.(mlK-agpK)) ≈ 0 atol = 1e-5
@test sum(abs.(mlKab-agpKab)) ≈ 0 atol = 1e-5

#Matern3_2Kernel

θ = 1.0
mlk = MLKernels.MaternKernel(1.5,θ)
agpk = KernelModule.Matern3_2Kernel(θ)
mlK = MLKernels.kernelmatrix(mlk,A)
agpK = KernelModule.kernelmatrix(A,agpk)
mlKab = MLKernels.kernelmatrix(mlk,A,B)
agpKab = KernelModule.kernelmatrix(A,B,agpk)

@test sum(abs.(mlK-agpK)) ≈ 0 atol = 1e-5
@test sum(abs.(mlKab-agpKab)) ≈ 0 atol = 1e-5

#Check for derivatives
θ = 1.0; ϵ=1e-7
for model in [RBFKernel,Matern3_2Kernel]
    k = model([θ,θ])
    keps = model([θ+ϵ,θ])
    diffK  = (KernelModule.kernelmatrix(A,keps) - KernelModule.kernelmatrix(A,k))./ϵ
    derivK = KernelModule.kernelderivativematrix(A,k)[1]
    diffKmn =  (KernelModule.kernelmatrix(A,B,keps) - KernelModule.kernelmatrix(A,B,k))./ϵ
    derivKmn = KernelModule.kernelderivativematrix(A,B,k)[1]
    display(@test sum(abs.(diffK-derivK)) ≈ 0 atol = 1e-1)
    display(@test sum(abs.(diffKmn-derivKmn)) ≈ 0 atol = 1e-1)
end

k = Matern3_2Kernel([1.0,1.0])

## Compute matrix variations via finite differences and compare with analytical
mlk = MLKernels.SquaredExponentialKernel(0.5/θ^2)
agpk = KernelModule.SEKernel([θ],dim=dims)
agpkepsilon = KernelModule.SEKernel([θ+ϵ,θ])

dagpK = (KernelModule.kernelmatrix(A,agpkepsilon) - KernelModule.kernelmatrix(A,agpk))./ϵ
dagpKana = KernelModule.kernelderivativematrix(A,agpk)


agpk = KernelModule.SEKernel(θ)
agpkepsilon = KernelModule.SEKernel(θ+ϵ)

dagpK = (KernelModule.kernelmatrix(A,agpkepsilon) - KernelModule.kernelmatrix(A,agpk))./ϵ
dagpKana = KernelModule.kernelderivativematrix(A,agpk)





using Plots; plotlyjs()

plot(heatmap(dagpK),heatmap(dagpKana),)
heatmap()
@btime mlK = MLKernels.kernelmatrix($mlk,$A);
@btime agpK = KernelModule.kernelmatrix($A,$agpk);
mlK = MLKernels.kernelmatrix(mlk,A);
agpK = KernelModule.kernelmatrix(A,agpk);

@test sum(agpK-mlK)≈0 atol=1e-3

mlK-agpK
ForwardDiff.gradient(make_matrix,[0.5])
