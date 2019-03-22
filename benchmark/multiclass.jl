cd(dirname(@__FILE__))
using BenchmarkTools
using AugmentedGaussianProcesses
using Distances, LinearAlgebra, Dates,  DelimitedFiles
using MLDataUtils: splitobs
suite = BenchmarkGroup()
suite["Full"] = BenchmarkGroup(["init","elbo","computematrices","updatevariational","updatehyperparam","predic","predicproba"])
suite["FullKStoch"] = BenchmarkGroup(["init","elbo","computematrices","updatevariational","updatehyperparam","predic","predicproba"])
suite["Sparse"] = BenchmarkGroup(["init","elbo","computematrices","updatevariational","updatehyperparam","predic","predicproba"])
suite["SparseStoch"] = BenchmarkGroup(["init","elbo","computematrices","updatevariational","updatehyperparam","predic","predicproba"])
suite["SparseStochKStoch"] = BenchmarkGroup(["init","elbo","computematrices","updatevariational","updatehyperparam","predic","predicproba"])

paramfile = "params/multiclass.json"
data = readdlm("data/vehicle.csv",',')
train,test=splitobs(data',at=0.7)
X_train = train'[:,1:end-1]; y_train = train'[:,end]
X_test = test'[:,1:end-1]; y_test = test'[:,end]
m = 50; batchsize = 50
kernel = RBFKernel([2.0],variance=1.0,dim=size(X_train,2))
models = Dict{String,GPModel}()

models["Full"] = MultiClass(X_train,y_train,kernel=kernel,Autotuning=true,μ_init=ones(Float64,size(X_train,1)))
models["FullKStoch"] = MultiClass(X_train,y_train,kernel=kernel,Autotuning=true,μ_init=ones(Float64,size(X_train,1)),KStochastic=true,nClassesUsed=1)
models["Sparse"] = SparseMultiClass(X_train,y_train,kernel=kernel,Autotuning=true,μ_init=ones(Float64,m),Stochastic=false,m=m)
models["SparseStoch"] = SparseMultiClass(X_train,y_train,kernel=kernel,Autotuning=true,μ_init=ones(Float64,m),Stochastic=true,m=m,batchsize=batchsize)
models["SparseStochKStoch"] = SparseMultiClass(X_train,y_train,kernel=kernel,Autotuning=true,μ_init=ones(Float64,m),Stochastic=true,m=m,batchsize=batchsize,KStochastic=true,nClassesUsed=1)

suite["Full"]["init"] = @benchmarkable MultiClass($X_train,$y_train,kernel=$kernel,Autotuning=true,μ_init=ones(Float64,size($X_train,1)))
suite["FullKStoch"]["init"] = @benchmarkable MultiClass($X_train,$y_train,kernel=$kernel,Autotuning=true,μ_init=ones(Float64,size($X_train,1)),KStochastic=true,nClassesUsed=1)
suite["Sparse"]["init"] = @benchmarkable SparseMultiClass($X_train,$y_train,kernel=$kernel,Autotuning=true,μ_init=ones(Float64,$m),Stochastic=false,m=$m)
suite["SparseStoch"]["init"] = @benchmarkable SparseMultiClass($X_train,$y_train,kernel=$kernel,Autotuning=true,μ_init=ones(Float64,$m),Stochastic=true,m=$m,batchsize=$batchsize)
suite["SparseStochKStoch"]["init"] = @benchmarkable SparseMultiClass($X_train,$y_train,kernel=$kernel,Autotuning=true,μ_init=ones(Float64,$m),Stochastic=true,m=$m,batchsize=$batchsize,KStochastic=true,nClassesUsed=1)
for KT in ["Full","FullKStoch","Sparse","SparseStoch","SparseStochKStoch"]
    println(KT)
    models[KT].train(iterations=1)
    suite[KT]["elbo"] = @benchmarkable AugmentedGaussianProcesses.ELBO(model) setup=(model=deepcopy($(models[KT])))
    suite[KT]["computematrices"] = @benchmarkable AugmentedGaussianProcesses.computeMatrices!(model) setup=(model=deepcopy($(models[KT])))
    suite[KT]["updatevariational"] = @benchmarkable AugmentedGaussianProcesses.variational_updates!(model,1) setup=(model=deepcopy($(models[KT])))
    suite[KT]["updatehyperparam"] = @benchmarkable AugmentedGaussianProcesses.updateHyperParameters!(model) setup=(model=deepcopy($(models[KT])))
    suite[KT]["predic"] = @benchmarkable AugmentedGaussianProcesses.multiclasspredict(model,$X_test) setup=(model=deepcopy($(models[KT])))
    suite[KT]["predicproba"] = @benchmarkable AugmentedGaussianProcesses.multiclasspredictproba(model,$X_test) setup=(model=deepcopy($(models[KT])))
end

if isfile(paramfile)
    loadparams!(suite,BenchmarkTools.load(paramfile))
else
    println("Tuning parameters")
    tune!(suite,verbose=true)
    BenchmarkTools.save(paramfile,params(suite))
end
println("Running benchmarks")
results = run(suite,verbose=true,seconds=30)
save_target = "results/multiclass_"*("$(now())"[1:10])
i = 1
while isfile(save_target*"_$(i).json")
    global i += 1
end
BenchmarkTools.save(save_target*"_$(i).json",results)
