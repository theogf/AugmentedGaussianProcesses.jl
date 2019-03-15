using AugmentedGaussianProcesses
using Random: seed!
using MLKernels, Statistics, LinearAlgebra
using ForwardDiff
const AGP = AugmentedGaussianProcesses
seed!(42)
dims= 2; nA = 10; nB = 5
A = rand(nA,dims)
B = rand(nB,dims)

function adapt_AD(kernel::AGP.Kernel{T},X::AbstractMatrix{T2}) where {T,T2}
    T.(X)
end

function create_kernel(k,θ,name;ν=0)
    if name == "RBFKernel"
        return k(θ)
    elseif name == "MaternKernel"
        return k(θ,ν)
    end
end
#Compare kernel results with MLKernels package
@testset "Kernels" begin

    @testset "Kernel computation" begin
        θ = 0.5; ν = 2.0
        @testset "Iso Kernels" begin
            kernels_GP = [AGP.RBFKernel(θ),AGP.MaternKernel(θ,ν)]
            kernels_ML = [MLKernels.GaussianKernel(0.5/θ^2),MLKernels.MaternKernel(ν,θ)]
            test_name = ["RBFKernel","MaternKernel"]
            for (k_GP,k_ML,t_name) in zip(kernels_GP,kernels_ML,test_name)
                @testset "$t_name" begin
                    @test sum(abs.(MLKernels.kernelmatrix(k_ML,A)- AGP.kernelmatrix(A,k_GP))) ≈ 0 atol=1e-6
                    @test sum(abs.(MLKernels.kernelmatrix(k_ML,A,B)- AGP.kernelmatrix(A,B,k_GP))) ≈ 0 atol=1e-6
                    @test sum(abs.(diag(AGP.kernelmatrix(A,k_GP))-AGP.kerneldiagmatrix(A,k_GP))) ≈ 0 atol=1e-6
                end
            end
        end
        @testset "ARD Kernels" begin
            kernels_Iso = [AGP.RBFKernel(θ),AGP.MaternKernel(θ,ν)]
            kernels_ARD = [AGP.RBFKernel(fill(θ,dims)),AGP.MaternKernel(fill(θ,dims),ν)]
            test_name = ["RBFKernel","MaternKernel"]
            for (k_iso,k_ard,t_name) in zip(kernels_Iso,kernels_ARD,test_name)
                @testset "$t_name" begin
                    @test sum(abs.(AGP.kernelmatrix(A,k_iso)- AGP.kernelmatrix(A,k_ard))) ≈ 0 atol=1e-6
                    @test sum(abs.(AGP.kernelmatrix(A,B,k_iso)- AGP.kernelmatrix(A,B,k_ard))) ≈ 0 atol=1e-6
                    @test sum(abs.(diag(AGP.kernelmatrix(A,k_ard))-AGP.kerneldiagmatrix(A,k_ard))) ≈ 0 atol=1e-6
                end
            end
        end
    end

    @testset "Kernel derivatives" begin
        θ = 0.5; ν = 2.0
        @testset "Iso Kernels" begin
            kernels = [AGP.RBFKernel(θ)]#,AGP.MaternKernel(θ,ν)]
            kernels_AD = [AGP.RBFKernel]#,AGP.MaternKernel]
            test_name = ["RBFKernel"]#,"MaternKernel"]
            for (k,k_AD,t_name) in zip(kernels,kernels_AD,test_name)
                @testset "$t_name" begin
                    dA_ad = reshape(ForwardDiff.jacobian(x->begin;k_ = create_kernel(k_AD,x[1],t_name,ν=ν); AGP.kernelmatrix(adapt_AD(k_,A),k_); end,[θ]),nA,nA)
                    dAB_ad = reshape(ForwardDiff.jacobian(x->begin;k_ = create_kernel(k_AD,x[1],t_name,ν=ν); AGP.kernelmatrix(adapt_AD(k_,A),adapt_AD(k_,B),k_); end,[θ]),nA,nB)
                    @test sum(abs.(AGP.kernelderivativematrix(A,k).- dA_ad)) ≈ 0 atol=1e-6
                    @test sum(abs.(AGP.kernelderivativematrix(A,B,k)- dAB_ad)) ≈ 0 atol=1e-6
                    @test sum(abs.(AGP.kernelderivativediagmatrix(A,k)-diag(dA_ad))) ≈ 0 atol=1e-6
                end
            end
        end
        @testset "ARD Kernels" begin
            θ = rand(dims)
            kernels = [AGP.RBFKernel(θ)]#,AGP.MaternKernel(θ,ν)]
            kernels_AD = [AGP.RBFKernel]#,AGP.MaternKernel]
            test_name = ["RBFKernel"]#,"MaternKernel"]
            for (k,k_AD,t_name) in zip(kernels,kernels_AD,test_name)
                @testset "$t_name" begin
                    dA_ad = ForwardDiff.jacobian(x->begin;k_ = create_kernel(k_AD,x,t_name,ν=ν); AGP.kernelmatrix(adapt_AD(k_,A),k_); end,θ)
                    dAB_ad = ForwardDiff.jacobian(x->begin;k_ = create_kernel(k_AD,x,t_name,ν=ν); AGP.kernelmatrix(adapt_AD(k_,A),adapt_AD(k_,B),k_); end,θ)
                    @test sum(abs.(hcat(vec.(AGP.kernelderivativematrix(A,k))...).- dA_ad)) ≈ 0 atol=1e-6
                    @test sum(abs.(hcat(vec.(AGP.kernelderivativematrix(A,B,k))...)- dAB_ad)) ≈ 0 atol=1e-6
                    @test sum(abs.(sum(AGP.kernelderivativediagmatrix(A,k).-diag.(reshape.(eachcol(dA_ad),nA,nA))))) ≈ 0 atol=1e-6
                end
            end
        end

    end
end
