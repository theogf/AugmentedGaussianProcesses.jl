cd(dirname(@__FILE__))
using BenchmarkTools
using OMGP
using Distances, LinearAlgebra, Dates, MLDataUtils, DelimitedFiles
suite = BenchmarkGroup()
suite["Full"] = BenchmarkGroup(["init","elbo","computematrices","updatevariational","updatehyperparam","predic","predicproba"])
suite["Sparse"] = BenchmarkGroup(["init","elbo","computematrices","updatevariational","updatehyperparam","predic","predicproba"])
suite["SparseStoch"] = BenchmarkGroup(["init","elbo","computematrices","updatevariational","updatehyperparam","predic","predicproba"])
paramfile = "params/bsvm.json"
data = readdlm("data/banana_dataset")
train,test=splitobs(data',at=0.7)
X_train = train'[:,1:2]; y_train = train'[:,3]
X_test = test'[:,1:2]; y_test = test'[:,3]
m = 50; batchsize = 50
kernel = RBFKernel([2.0],variance=1.0,dim=2)
models = Dict{String,GPModel}()
models["Full"] = BatchBSVM(X_train,y_train,kernel=kernel,Autotuning=true,μ_init=ones(Float64,size(X_train,1)))

models["Sparse"] = SparseBSVM(X_train,y_train,kernel=kernel,Autotuning=true,μ_init=ones(Float64,m),Stochastic=false,m=m)
models["SparseStoch"] = SparseBSVM(X_train,y_train,kernel=kernel,Autotuning=true,μ_init=ones(Float64,m),Stochastic=true,m=m,batchsize=batchsize)

suite["Full"]["init"] = @benchmarkable BatchBSVM($X_train,$y_train,kernel=$kernel,Autotuning=true,μ_init=ones(Float64,size($X_train,1)))
suite["Sparse"]["init"] = @benchmarkable SparseBSVM($X_train,$y_train,kernel=$kernel,Autotuning=true,μ_init=ones(Float64,$m),Stochastic=false,m=$m)
suite["SparseStoch"]["init"] = @benchmarkable SparseBSVM($X_train,$y_train,kernel=$kernel,Autotuning=true,μ_init=ones(Float64,$m),Stochastic=true,m=$m,batchsize=$batchsize)
for KT in ["Full","Sparse","SparseStoch"]
    models[KT].train(iterations=1)
    suite[KT]["elbo"] = @benchmarkable OMGP.ELBO($(models[KT]))
    suite[KT]["computematrices"] = @benchmarkable OMGP.computeMatrices!(model) setup=(model=deepcopy($(models[KT])))
    suite[KT]["updatevariational"] = @benchmarkable OMGP.variational_updates!(model,1) setup=(model=deepcopy($(models[KT])))
    suite[KT]["updatehyperparam"] = @benchmarkable OMGP.updateHyperParameters!(model) setup=(model=deepcopy($(models[KT])))
    suite[KT]["predic"] = @benchmarkable OMGP.probitpredict(model,$X_test) setup=(model=deepcopy($(models[KT])))
    suite[KT]["predicproba"] = @benchmarkable OMGP.probitpredictproba(model,X_test) setup=(model=deepcopy($(models[KT])))
end

if isfile(paramfile)
    loadparams!(suite,BenchmarkTools.load(paramfile)[1])
else
    println("Tuning parameters")
    tune!(suite,verbose=true)
    BenchmarkTools.save(paramfile,params(suite))
end
println("Running benchmarks")
results = run(suite,verbose=true,seconds=30)
save_target = "results/bsvm_"*("$(now())"[1:10])
i = 1
while isfile(save_target*"_$(i).json")
    global i += 1
end
BenchmarkTools.save(save_target*"_$(i).json",results)
