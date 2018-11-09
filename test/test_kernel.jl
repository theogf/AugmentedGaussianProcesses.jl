using AugmentedGaussianProcesses.KernelModule
using Profile, ProfileView, BenchmarkTools, Test
using Random: seed!
using MLKernels, Statistics, LinearAlgebra
using ForwardDiff
seed!(42)
dims= 2
A = sort(rand(1000,dims),dims=1)
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
mlk = MLKernels.MaternKernel(2.0,θ)
agpk = KernelModule.MaternKernel(θ,2.0)
mlK = MLKernels.kernelmatrix(mlk,A)
agpK = KernelModule.kernelmatrix(A,agpk)
mlKab = MLKernels.kernelmatrix(mlk,A,B)
agpKab = KernelModule.kernelmatrix(A,B,agpk)

@test sum(abs.(mlK-agpK)) ≈ 0 atol = 1e-5
@test sum(abs.(mlKab-agpKab)) ≈ 0 atol = 1e-5

global derivK = KernelModule.kernelderivativematrix(A,agpk)


#Check for derivatives
θ = 1.0; ϵ=1e-7; ν=2.0; Aeps = copy(A); Aeps[1] = Aeps[1]+ϵ
for params in [([θ,θ],[θ+ϵ,θ]),(θ,θ+ϵ)]
    println("Testing params $params")
    for model in [KernelModule.MaternKernel,RBFKernel]
        println("Testing model $model")
        if model == KernelModule.MaternKernel
            k = model(params[1],ν)
            keps = model(params[2],ν)
        elseif model == KernelModule.RBFKernel
            k = model(params[1])
            keps = model(params[2])
        end
        global diffK  = (KernelModule.kernelmatrix(A,keps) - KernelModule.kernelmatrix(A,k))./ϵ
        global derivK = typeof(params[1]) <:AbstractArray ? KernelModule.kernelderivativematrix(A,k)[1] : KernelModule.kernelderivativematrix(A,k)
        global diffKmn =  (KernelModule.kernelmatrix(A,B,keps) - KernelModule.kernelmatrix(A,B,k))./ϵ
        global derivKmn = typeof(params[1]) <: AbstractArray ?  KernelModule.kernelderivativematrix(A,B,k)[1] : KernelModule.kernelderivativematrix(A,B,k)
        global diffInd = ((KernelModule.kernelmatrix(Aeps,k) - KernelModule.kernelmatrix(A,k))./ϵ)[:,1]
        K = Symmetric(KernelModule.kernelmatrix(A,k))
        global derivInd = KernelModule.computeIndPointsJmm(k,A,1,K)[:,1]
        display(@test sum(abs.(diffK-derivK)) ≈ 0 atol = 1e-1)
        display(@test sum(abs.(diffKmn-derivKmn)) ≈ 0 atol = 1e-1)
        display(@test sum(abs.(diffInd-derivInd)) ≈ 0 atol = 1e-1)
    end
end
