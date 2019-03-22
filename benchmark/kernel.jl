using AugmentedGaussianProcesses.KernelModule
using Distances, LinearAlgebra
using BenchmarkTools
using Random
if !@isdefined(SUITE)
    const SUITE = BenchmarkGroup(["Kernel"])
end

paramfile = "params/kernel.json"
dim = 50
Random.seed!(1234)
N1 = 1000; N2 = 500;
X = rand(Float64,N1,dim)
Y = rand(Float64,N2,dim)
KXY = rand(Float64,N1,N2)
KX = rand(Float64,N1,N1)
sKX = Symmetric(rand(Float64,N1,N1))
kX = rand(Float64,N1)

kernelnames = ["Matern","RBF"]
kerneltypes = ["ARD","ISO"]
kernels=Dict{String,Dict{String,Kernel}}()
for k in kernelnames
    kernels[k] = Dict{String,Kernel}()
    SUITE["Kernel"][k] = BenchmarkGroup(kerneltypes)
    for kt in kerneltypes
        SUITE["Kernel"][k][kt] = BenchmarkGroup(["XY","XYinplace","X","Xinplace","diagX","diagXinplace","dXY","dX","ddiagX"])
        kernels[k][kt] = eval(Meta.parse(k*"Kernel("*(kt == "ARD" ? "[2.0]" : "2.0" )*",variance=10.0,dim=dim)"))
    end
end

for k in kernelnames
    for kt in kerneltypes
        SUITE["Kernel"][k][kt]["XY"] = @benchmarkable kernelmatrix($X,$Y,$(kernels[k][kt]))
        SUITE["Kernel"][k][kt]["XYinplace"] = @benchmarkable kernelmatrix!(KXY,$X,$Y,$(kernels[k][kt])) setup=(KXY=copy($KXY))
        SUITE["Kernel"][k][kt]["X"] = @benchmarkable kernelmatrix($X,$(kernels[k][kt]))
        SUITE["Kernel"][k][kt]["Xinplace"] = @benchmarkable kernelmatrix!(KX,$X,$(kernels[k][kt])) setup=(KX=copy($KX))
        SUITE["Kernel"][k][kt]["diagX"] = @benchmarkable kerneldiagmatrix($X,$(kernels[k][kt]))
        SUITE["Kernel"][k][kt]["diagXinplace"] = @benchmarkable kerneldiagmatrix!(kX,$X,$(kernels[k][kt])) setup=(kX=copy($kX))
        SUITE["Kernel"][k][kt]["dXY"] = @benchmarkable kernelderivativematrix($X,$Y,$(kernels[k][kt]))
        # SUITE["Kernel"][k][kt]["dXY_K"] = @benchmarkable kernelderivativematrix_K($X,$Y,KXY,$(kernels[k][kt])) setup=(KXY=copy($KXY))
        SUITE["Kernel"][k][kt]["dX"] = @benchmarkable kernelderivativematrix($X,$(kernels[k][kt]))
        # SUITE["Kernel"][k][kt]["dX_K"] = @benchmarkable kernelderivativematrix_K($X,sKX,$(kernels[k][kt])) setup=(sKX=copy($sKX))
        SUITE["Kernel"][k][kt]["ddiagX"] = @benchmarkable kernelderivativediagmatrix($X,$(kernels[k][kt]))
    end
end

# if isfile(paramfile)
#     loadparams!(suite,BenchmarkTools.load(paramfile)[1])
# else
#     tune!(suite,verbose=true)
#     BenchmarkTools.save(paramfile,params(suite))
# end
#
# results = run(suite,verbose=true,seconds=10)
# save_target = "results/kernel_"*("$(now())"[1:10])
# i = 1
# while isfile(save_target*"_$(i).json")
#     global i += 1
# end
# BenchmarkTools.save(save_target*"_$(i).json",results)
