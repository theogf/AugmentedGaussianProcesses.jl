cd(dirname(@__FILE__))
using BenchmarkTools
using AugmentedGaussianProcesses
using Distances, LinearAlgebra, Dates,  DelimitedFiles
using MLDataUtils: splitobs
using Random


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

const SUITE = BenchmarkGroup(["Models"])
paramfile = "params/multiclass.json"
Random.seed!(1234)
nData = 200; nDim = 3;
X = rand(nData,nDim)
y = Dict("Gaussian"=>norm.(eachrow(X)),"AugmentedStudentT"=>norm.(eachrow(X)),"BayesianSVM"=>sign.(norm.(eachrow(X)).-1.0),"AugmentedLogistic"=>sign.(norm.(eachrow(X)).-1.0),"AugmentedLogisticSoftMax"=>floor.(norm.(eachrow(X.*2))))
n_ind = 50; batchsize = 50; ν = 10.0
convertl(l::String) = l*(l!= "BayesianSVM" ? "Likelihood" : "")*"("*(l == "AugmentedStudentT" ? "ν" : "")*")"
converti(i::String) = i*"("*(i[1:10] == "Stochastic" ? "batchsize" : "")*")"
add_ind(mname::String) = mname == "SVGP" ? ",n_ind" : ""
kernel = RBFKernel([2.0],variance=1.0,dim=nDim)
models = Dict{String,Dict{String,Dict{String,AbstractGP}}}()
SUITE["Models"] = BenchmarkGroup(modelnames)
for m in modelnames
    SUITE["Models"][m] = BenchmarkGroup(String.(keys(compat[m])))
    models[m] = Dict{String,Dict{String,AbstractGP}}()
    for l in String.(keys(compat[m]))
        SUITE["Models"][m][l] = BenchmarkGroup(compat[m][l])
        models[m][l] = Dict{String,AbstractGP}()
        for i in compat[m][l]
            SUITE["Models"][m][l][i] = BenchmarkGroup(funcs)
            println(m,convertl(l),converti(i))
            if m == "GP"
                models[m][l][i] = eval(Meta.parse(m*"(X,y[l],kernel,Autotuning=true,atfrequency=1)"))
            else
                models[m][l][i] = eval(Meta.parse(m*"(X,y[l],kernel,$(convertl(l)) ,$(converti(i))$(add_ind(m)),Autotuning=true,atfrequency=1)"))
            end
            SUITE["Models"][m][l][i]["init"] = @benchmarkable eval(Meta.parse(m*"(X,y[l],kernel,$(convertl(l)),$(converti(i))$(add_ind),Autotuning=true,atfrequency=1)"))
            SUITE["Models"][m][l][i]["elbo"] = @benchmarkable ELBO(model) setup=(model=deepcopy($(models[m][l][i])))
            SUITE["Models"][m][l][i]["computematrices"] = @benchmarkable computeMatrices!(model) setup=(model=deepcopy($(models[m][l][i])))
            SUITE["Models"][m][l][i]["updatevariational"] = @benchmarkable variational_updates!(model,1) setup=(model=deepcopy($(models[m][l][i])))
            SUITE["Models"][m][l][i]["updatehyperparam"] = @benchmarkable updateHyperParameters!(model) setup=(model=deepcopy($(models[m][l][i])))
            SUITE["Models"][m][l][i]["predic"] = @benchmarkable predict_y(model,$X) setup=(model=deepcopy($(models[m][l][i])))
            SUITE["Models"][m][l][i]["predicproba"] = @benchmarkable proba_y(model,$X) setup=(model=deepcopy($(models[m][l][i])))
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
