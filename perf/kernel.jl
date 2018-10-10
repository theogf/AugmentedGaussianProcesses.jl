cd(dirname(@__FILE__))
using BenchmarkTools,Traceur,Profile,ProfileView
using OMGP.KernelModule
using Distances, LinearAlgebra, Dates
suite = BenchmarkGroup()
suite["ardmatrices"] = BenchmarkGroup(["XY","XYinplace","X","Xinplace","diagX","diagXinplace"])
suite["plainmatrices"] = BenchmarkGroup(["XY","XYinplace","X","Xinplace","diagX","diagXinplace"])
paramfile = "params/kernel.json"
dim = 50
N1 = 1000; N2 = 500;
X = rand(N1,dim)
Y = rand(N2,dim)
KXY = rand(N1,N2)
KX = rand(N1,N1)
sKX = Symmetric(rand(N1,N1))
kX = rand(N1)
kernels=Dict{String,Kernel}()
kernels["ard"] = RBFKernel([2.0],variance=10.0,dim=dim)
kernels["plain"] = RBFKernel(2.0,variance=10.0)
export kernelderivativematrix_K,kernelderivativediagmatrix_K
export kernelderivativematrix,kernelderivativediagmatrix
for KT in ["ard","plain"]
    suite[KT*"matrices"]["XY"] = @benchmarkable kernelmatrix($X,$Y,$(kernels[KT]))
    suite[KT*"matrices"]["XYinplace"] = @benchmarkable kernelmatrix!($KXY,$X,$Y,$(kernels[KT]))
    suite[KT*"matrices"]["X"] = @benchmarkable kernelmatrix($X,$(kernels[KT]))
    suite[KT*"matrices"]["Xinplace"] = @benchmarkable kernelmatrix!($KX,$X,$(kernels[KT]))
    suite[KT*"matrices"]["diagX"] = @benchmarkable kerneldiagmatrix($X,$(kernels[KT]))
    suite[KT*"matrices"]["diagXinplace"] = @benchmarkable kerneldiagmatrix!($kX,$X,$(kernels[KT]))
    suite[KT*"matrices"]["dXY"] = @benchmarkable kernelderivativematrix($X,$Y,$(kernels[KT]))
    suite[KT*"matrices"]["dXY_K"] = @benchmarkable kernelderivativematrix_K($X,$Y,$KXY,$(kernels[KT]))
    suite[KT*"matrices"]["dX"] = @benchmarkable kernelderivativematrix($X,$(kernels[KT]))
    suite[KT*"matrices"]["dX_K"] = @benchmarkable kernelderivativematrix_K($X,$sKX,$(kernels[KT]))
    suite[KT*"matrices"]["ddiagX"] = @benchmarkable kernelderivativediagmatrix($X,$(kernels[KT]))
end

if isfile(paramfile)
    loadparams!(suite,BenchmarkTools.load(paramfile)[1])
else
    tune!(suite,verbose=true)
    BenchmarkTools.save(paramfile,params(suite))
end

results = run(suite,verbose=true)
save_target = "results/kernel_"*("$(now())"[1:10])
i = 1
while isfile(save_target*"_$(i).json")
    global i += 1
end
BenchmarkTools.save(save_target*"_$(i).json",results)
