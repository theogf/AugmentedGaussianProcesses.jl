using BenchmarkTools
using OMGP
using Distances, LinearAlgebra, Dates, MLDataUtils, DelimitedFiles
suite = BenchmarkGroup()
suite["Full"] = BenchmarkGroup(["init","elbo","computematrices","updatevariational","updatehyperparam","predic","predicproba"])
suite["Sparse"] = BenchmarkGroup(["init","elbo","computematrices","updatevariational","updatehyperparam","predic","predicproba"])
suite["SparseStoch"] = BenchmarkGroup(["init","elbo","computematrices","updatevariational","updatehyperparam","predic","predicproba"])
paramfile = "params/xgpc.json"
data = readdlm("banana_dataset")
train,test=splitobs(data',at=0.7)
X_train = train'[:,1:2]; y_train = train'[:,3]
X_test = test'[:,1:2]; y_test = test'[:,3]
m = 50; batchsize = 50
kernel = RBFKernel([2.0],variance=1.0,dim=2)
models = Dict{String,GPModel}()
models["Full"] = BatchXGPC(X_train,y_train,kernel=kernel,Autotuning=true,μ_init=ones(Float64,size(X_train,1)))

models["Sparse"] = SparseXGPC(X_train,y_train,kernel=kernel,Autotuning=true,μ_init=ones(Float64,m),Stochastic=false,m=m)
models["SparseStoch"] = SparseXGPC(X_train,y_train,kernel=kernel,Autotuning=true,μ_init=ones(Float64,m),Stochastic=true,m=m,batchsize=batchsize)

suite["Full"]["init"] = @benchmarkable BatchXGPC($X_train,$y_train,kernel=$kernel,Autotuning=true,μ_init=ones(Float64,size($X_train,1)))
suite["Sparse"]["init"] = @benchmarkable SparseXGPC($X_train,$y_train,kernel=$kernel,Autotuning=true,μ_init=ones(Float64,$m),Stochastic=false,m=$m)
suite["SparseStoch"]["init"] = @benchmarkable SparseXGPC($X_train,$y_train,kernel=$kernel,Autotuning=true,μ_init=ones(Float64,$m),Stochastic=true,m=$m,batchsize=$batchsize)
for KT in ["Full","Sparse","SparseStoch"]
    models[KT].train(iterations=1)
    suite[KT]["elbo"] = @benchmarkable OMGP.ELBO($(models[KT]))
    suite[KT]["computematrices"] = @benchmarkable OMGP.computeMatrices!($(models[KT]))
    suite[KT]["updatevariational"] = @benchmarkable OMGP.updatevariational!($(models[KT]))
    suite[KT]["updatehyperparam"] = @benchmarkable OMGP.updatehyperparam!($(models[KT]))
    suite[KT]["predic"] = @benchmarkable OMGP.probitpredict($(models[KT]),$X_test)
    suite[KT]["predicproba"] = @benchmarkable OMGP.probitpredictproba($(models[KT]),$X_test)
end

if isfile(paramfile)
    loadparams!(suite,BenchmarkTools.load(paramfile))
else
    println("Tuning parameters")
    tune!(suite)
    BenchmarkTools.save(paramfile,params(suite))
end
println("Running benchmarks")
results = run(suite)
save_target = "results/xpgc_"*("$(now())"[1:10])
i = 1
while isfile(save_target*"_$(i).json")
    global i += 1
end
BenchmarkTools.save(save_target*"_$(i).json",results)
