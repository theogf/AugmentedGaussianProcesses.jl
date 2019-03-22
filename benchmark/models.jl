using BenchmarkTools
using AugmentedGaussianProcesses
using Distances, LinearAlgebra, DelimitedFiles
using MLDataUtils: splitobs
using Random
const AGP = AugmentedGaussianProcesses

compat = Dict{String,Dict{String,Vector{String}}}()
likelihoodnames = ["Gaussian","AugmentedStudentT","AugmentedLogistic","BayesianSVM","AugmentedLogisticSoftMax"]
inferencenames = ["AnalyticInference","StochasticAnalyticInference"]
modelnames = ["GP","VGP","SVGP"]
funcs = ["init","elbo","computematrices","updatevariational","updatehyperparam","predic","predicproba"]
compat["GP"] = Dict{String,Vector{String}}("Gaussian"=>["AnalyticInference"])
compat["VGP"] = Dict{String,Vector{String}}()
compat["VGP"]["AugmentedStudentT"] = ["AnalyticInference"]
compat["VGP"]["AugmentedLogistic"] = ["AnalyticInference"]
compat["VGP"]["BayesianSVM"] = ["AnalyticInference"]
compat["VGP"]["AugmentedLogisticSoftMax"] = ["AnalyticInference"]
compat["SVGP"] = Dict{String,Vector{String}}()
compat["SVGP"]["Gaussian"] = ["AnalyticInference","StochasticAnalyticInference"]
compat["SVGP"]["AugmentedStudentT"] = ["AnalyticInference","StochasticAnalyticInference"]
compat["SVGP"]["AugmentedLogistic"] = ["AnalyticInference","StochasticAnalyticInference"]
compat["SVGP"]["BayesianSVM"] = ["AnalyticInference","StochasticAnalyticInference"]
compat["SVGP"]["AugmentedLogisticSoftMax"] = ["AnalyticInference","StochasticAnalyticInference"]

# const SUITE = BenchmarkGroup(["Models"])
Random.seed!(1234)
nData = 200; nDim = 3;
X = rand(nData,nDim)
y = Dict("Gaussian"=>norm.(eachrow(X)),"AugmentedStudentT"=>norm.(eachrow(X)),"BayesianSVM"=>sign.(norm.(eachrow(X)).-1.0),"AugmentedLogistic"=>sign.(norm.(eachrow(X)).-1.0),"AugmentedLogisticSoftMax"=>floor.(norm.(eachrow(X.*2))))
n_ind = 50; batchsize = 50; ν = 10.0
convertl(lname::String) = lname*(lname != "BayesianSVM" ? "Likelihood" : "")*"("*(lname == "AugmentedStudentT" ? "ν" : "")*")"
converti(iname::String) = iname*"("*(iname[1:10] == "Stochastic" ? "batchsize" : "")*")"
add_ind(mname::String) = mname == "SVGP" ? ",n_ind" : ""
kernel = RBFKernel([2.0],variance=1.0,dim=nDim)
models = Dict{String,Dict{String,Dict{String,AbstractGP}}}()
SUITE["Models"] = BenchmarkGroup(modelnames)
for model in String.(keys(compat))
    SUITE["Models"][model] = BenchmarkGroup(String.(keys(compat[model])))
    models[model] = Dict{String,Dict{String,AbstractGP}}()
    for likelihood in String.(keys(compat[model]))
        SUITE["Models"][model][likelihood] = BenchmarkGroup(compat[model][likelihood])
        models[model][likelihood] = Dict{String,AbstractGP}()
        for i in compat[model][likelihood]
            SUITE["Models"][model][likelihood][i] = BenchmarkGroup(funcs)
            if model == "GP"
                println(Meta.parse(model*"(\$X,\$y[\"$likelihood\"],\$kernel,Autotuning=true,atfrequency=1)"))
                models[model][likelihood][i] = eval(Meta.parse(model*"(X,y[\"$likelihood\"],kernel,Autotuning=true,atfrequency=1)"))
                SUITE["Models"][model][likelihood][i]["init"] = @benchmarkable eval(Meta.parse(modelname*"(\$X,y_train,\$kernel,Autotuning=true,atfrequency=1)")) setup = (y_train = $y[$likelihood], modelname=$model)
            else
                println(Meta.parse(model*"(X,y[\"$likelihood\"],kernel,$(convertl(likelihood)) ,$(converti(i))$(add_ind(model)),Autotuning=true,atfrequency=1)"))
                models[model][likelihood][i] = eval(Meta.parse(model*"(X,y[\"$likelihood\"],kernel,$(convertl(likelihood)) ,$(converti(i))$(add_ind(model)),Autotuning=true,atfrequency=1)"))
                SUITE["Models"][model][likelihood][i]["init"] = @benchmarkable eval(Meta.parse($model*"(\$X,y_train,$(convertl($likelihood)),$(converti(i))$(add_ind),Autotuning=true,atfrequency=1)")) setup = (y_train = $y[$likelihood])
            end
            SUITE["Models"][model][likelihood][i]["elbo"] = @benchmarkable ELBO(gpmodel) setup=(gpmodel=deepcopy($(models[model][likelihood][i])))
            SUITE["Models"][model][likelihood][i]["computematrices"] = @benchmarkable AGP.computeMatrices!(gpmodel) setup=(gpmodel=deepcopy($(models[model][likelihood][i])))
            SUITE["Models"][model][likelihood][i]["updatevariational"] = @benchmarkable variational_updates!(gpmodel,1) setup=(gpmodel=deepcopy($(models[model][likelihood][i])))
            SUITE["Models"][model][likelihood][i]["updatehyperparam"] = @benchmarkable updateHyperParameters!(gpmodel) setup=(gpmodel=deepcopy($(models[model][likelihood][i])))
            SUITE["Models"][model][likelihood][i]["predic"] = @benchmarkable predict_y(gpmodel,$X) setup=(gpmodel=deepcopy($(models[model][likelihood][i])))
            SUITE["Models"][model][likelihood][i]["predicproba"] = @benchmarkable proba_y(gpmodel,$X) setup=(gpmodel=deepcopy($(models[model][likelihood][i])))
        end
    end
end
#
# if isfile(paramfile)
#     loadparams!(suite,BenchmarkTools.load(paramfile))
# else
#     println("Tuning parameters")
#     tune!(suite,verbose=true)
#     BenchmarkTools.save(paramfile,params(suite))
# end
# println("Running benchmarks")
# results = run(suite,verbose=true,seconds=30)
# save_target = "results/multiclass_"*("$(now())"[1:10])
# i = 1
# while isfile(save_target*"_$(i).json")
#     global i += 1
# end
# BenchmarkTools.save(save_target*"_$(i).json",results)
