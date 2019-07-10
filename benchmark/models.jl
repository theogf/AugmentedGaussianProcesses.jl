using BenchmarkTools
using AugmentedGaussianProcesses
using Distances, LinearAlgebra, CSV
using Random
const AGP = AugmentedGaussianProcesses

## Benchmark variational parameters update

## Benchmark hyperparameter optimization

## Benchmark full update

## Benchmark fit performance

## Benchmark



compat = Dict{String,Dict{String,Vector{String}}}()
likelihoodnames = ["Gaussian","StudentT","Logistic","BayesianSVM","LogisticSoftMax"]
inferencenames = ["AnalyticVI","AnalyticSVI"]
modelnames = ["GP","VGP","SVGP"]
funcs = ["init","elbo","computematrices","updatevariational","updatehyperparam","predic","predicproba"]
compat["GP"] = Dict{String,Vector{String}}("Gaussian"=>["AnalyticVI"])
compat["VGP"] = Dict{String,Vector{String}}()
compat["VGP"]["StudentT"] = ["AnalyticVI"]
compat["VGP"]["Logistic"] = ["AnalyticVI"]
compat["VGP"]["BayesianSVM"] = ["AnalyticVI"]
compat["VGP"]["LogisticSoftMax"] = ["AnalyticVI"]
compat["SVGP"] = Dict{String,Vector{String}}()
compat["SVGP"]["Gaussian"] = ["AnalyticVI","AnalyticSVI"]
compat["SVGP"]["StudentT"] = ["AnalyticVI","AnalyticSVI"]
compat["SVGP"]["Logistic"] = ["AnalyticVI","AnalyticSVI"]
compat["SVGP"]["BayesianSVM"] = ["AnalyticVI","AnalyticSVI"]
compat["SVGP"]["LogisticSoftMax"] = ["AnalyticVI","AnalyticSVI"]

const SUITE = BenchmarkGroup(["Models"])
Random.seed!(1234)
D = 20; N = 3000
data = CSV.read("benchmarkdata.csv")
X = Matrix(data[:,1:D])
y_key = Dict("Gaussian"=>:y_reg,"StudentT"=>:y_reg,"BayesianSVM"=>:y_class,"Logistic"=>:y_class,"LogisticSoftMax"=>:y_multi)
n_ind = 50; batchsize = 50; ν = 5.0
convertl(lname::String) = lname*(lname != "BayesianSVM" ? "Likelihood" : "")*"("*(lname == "StudentT" ? "ν" : "")*")"
converti(iname::String) = iname*"("*(iname == "AnalyticSVI" ? "batchsize" : "")*")"
add_ind(mname::String) = mname == "SVGP" ? ",n_ind" : ""
kernel = RBFKernel([2.0],variance=1.0,dim=D)
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
                models[model][likelihood][i] = eval(Meta.parse(model*"(X,Vector(data[:$((y_key[likelihood]))]),kernel,atfrequency=1)"))
                SUITE["Models"][model][likelihood][i]["init"] = eval(Meta.parse("@benchmarkable $model(\$X,y_train,\$kernel,atfrequency=1) setup=(y_train = Vector(\$D[:\$((y_key[likelihood]))])"))
            else
                # println(Meta.parse(model*"(X,y[\"$likelihood\"],kernel,$(convertl(likelihood)) ,$(converti(i))$(add_ind(model)),atfrequency=1)"))
                models[model][likelihood][i] = eval(Meta.parse(model*"(X,Vector(data[:$((y_key[likelihood]))]),kernel,$(convertl(likelihood)) ,$(converti(i))$(add_ind(model)),atfrequency=1)"))
                SUITE["Models"][model][likelihood][i]["init"] = eval(Meta.parse("@benchmarkable $model(\$X,y_train,\$kernel,$(convertl(likelihood)),$(converti(i)) $(add_ind(model)),atfrequency=1) setup=(y_train = Vector(\$data[:\$((y_key[likelihood]))]))"))
            end
            SUITE["Models"][model][likelihood][i]["elbo"] = @benchmarkable ELBO(gpmodel) setup=(gpmodel=deepcopy($(models[model][likelihood][i])))
            SUITE["Models"][model][likelihood][i]["computematrices"] = @benchmarkable AGP.computeMatrices!(gpmodel) setup=(gpmodel=deepcopy($(models[model][likelihood][i])))
            SUITE["Models"][model][likelihood][i]["updatevariational"] = @benchmarkable AGP.variational_updates!(gpmodel) setup=(gpmodel=deepcopy($(models[model][likelihood][i])))
            SUITE["Models"][model][likelihood][i]["updatehyperparam"] = @benchmarkable AGP.update_hyperparameters!(gpmodel) setup=(gpmodel=deepcopy($(models[model][likelihood][i])))
            SUITE["Models"][model][likelihood][i]["predic"] = @benchmarkable predict_y(gpmodel,$X) setup=(gpmodel=deepcopy($(models[model][likelihood][i])))
            SUITE["Models"][model][likelihood][i]["predicproba"] = @benchmarkable proba_y(gpmodel,$X) setup=(gpmodel=deepcopy($(models[model][likelihood][i])))
        end
    end
end
