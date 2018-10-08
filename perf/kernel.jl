using BenchmarkTools,Traceur,Profile,ProfileView
using OMGP.KernelModule
using Distances
suite = BenchmarkGroup()
suite["matrices"] = BenchmarkGroup(["XY","XYinplace","X","Xinplace","diagX","diagXinplace"])

X = rand(1000,100)
Y = rand(500,100)
KXY = zeros(1000,500)
KX = zeros(1000,1000)
kX = zeros(1000)
kernel = RBFKernel([2.0],variance=10.0,dim=100)


suite["matrices"]["XY"] = @benchmarkable kernelmatrix($X,$Y,$kernel)
suite["matrices"]["XYinplace"] = @benchmarkable kernelmatrix!($KXY,$X,$Y,$kernel)
suite["matrices"]["X"] = @benchmarkable kernelmatrix($X,$kernel)
suite["matrices"]["Xinplace"] = @benchmarkable kernelmatrix!($KX,$X,$kernel)
suite["matrices"]["diagX"] = @benchmarkable kerneldiagmatrix($X,$kernel)
suite["matrices"]["diagXinplace"] = @benchmarkable kerneldiagmatrix!($kX,$X,$kernel)

tune!(suite)

results = run(suite)

Profile.clear()
@profile kernelmatrix(X,Y,kernel)
ProfileView.view()
Profile.clear()
@profile KernelModule.kernelderivativematrix(X,Y,kernel)
