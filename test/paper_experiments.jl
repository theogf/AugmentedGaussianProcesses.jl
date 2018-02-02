#### Paper_Experiments ####
# Run on a file and compute accuracy on a nFold cross validation
# Compute also the brier score and the logscore
PWD = pwd()
if isdir(PWD*"/src")
    SRC_PATH = pwd()*"/src"
else
    SRC_PATH = pwd()*"/../src"
end
if !in(LOAD_PATH,SRC_PATH); push!(LOAD_PATH,SRC_PATH); end;
include("functions_paper_experiment.jl")
using PyPlot
using DataAccess
using ArgParse
function parse_commandline()
    s = ArgParseSettings(exc_handler=ArgParse.debug_handler)
    @add_arg_table s begin
        "dataset"
            help = "Dataset to train on"
            required = true
        "-M", "--IndPoints"
            help = "Number of inducing points"
            arg_type = Int
            default = 0
        "--autotuning", "-A"
            help = "Autotuning activated or not"
            action = :store_true
        "--maxiter", "-I"
            help = "Maximum number of iterations"
            arg_type = Int
            default = 1000
        "--noXGPC"
            help = "Run XGPC"
            action = :store_false
        "--SVGPC"
            help = "Run SVGPC"
            action = :store_true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

#Compare XGPC, BSVM, SVGPC and Logistic Regression

#Methods and scores to test
doBXGPC = false #Batch XGPC (no sparsity)
doSXGPC = true #Sparse XGPC (sparsity)
doLBSVM = false #Linear BSVM
doBBSVM = false #Batch BSVM
doSBSVM = false #Sparse BSVM
doSVGPC = true #Sparse Variational GPC (Hensmann)
doLogReg = false #Logistic Regression
doAutotuning = true
doPointOptimization = false
# ExperimentName = "Prediction"
ExperimentName = "ConvergenceExperiment"
@enum ExperimentType PredictionExp=0 AccuracyExp=1 ConvergenceExp=2
ExperimentTypes = Dict("Prediction"=>PredictionExp, "ConvergenceExperiment"=>ConvergenceExp,"Accuracy"=>AccuracyExp)
Experiment = ExperimentTypes[ExperimentName]
doTime = true #Return time needed for training
doAccuracy = true #Return Accuracy
doBrierScore = true # Return BrierScore
doLogScore = true #Return LogScore
doAUCScore = true
doLikelihoodScore = true
doSaveLastState = true
doPlot = true
doWrite = false #Write results in approprate folder
ShowIntResults = true #Show intermediate time, and results for each fold

iFold = 1
#Testing Parameters
#= Datasets available are X :
aXa, Bank_marketing, Click_Prediction, Cod-rna, Diabetis, Electricity, German, Shuttle
=#
dataset = length(ARGS)>0 ? String(ARGS[1]) : "Diabetis"
(X_data,y_data,DatasetName) = get_Dataset(dataset)
MaxIter = length(ARGS)>2 > Int64(ARGS[3]) : 10 #Maximum number of iterations for every algorithm
iter_points= vcat(1:99,100:10:999,1000:1000:9999)
(nSamples,nFeatures) = size(X_data);
nFold = 10; #Chose the number of folds
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold

println("Dataset $dataset loaded")

#Main Parameters
main_param = DefaultParameters()
main_param["nFeatures"] = nFeatures
main_param["nSamples"] = nSamples
main_param["ϵ"] = 1e-10 #Convergence criterium
main_param["M"] = length(ARGS)>3 ? Int64(ARGS[4]) : min(100,floor(Int64,0.2*nSamples)) #Number of inducing points
main_param["Kernel"] = "rbf"
main_param["Θ"] = 5.0 #Hyperparameter of the kernel
main_param["BatchSize"] = 100
main_param["Verbose"] = 0
main_param["Window"] = 10
main_param["Autotuning"] = length(ARGS)>1 ? Bool(ARGS[2]) : doAutotuning
main_param["PointOptimization"] = doPointOptimization
#BSVM and SVGPC Parameters
BXGPCParam = XGPCParameters(main_param=main_param)
SXGPCParam = XGPCParameters(Stochastic=true,Sparse=true,ALR=true,main_param=main_param)
LBSVMParam = BSVMParameters(Stochastic=false,NonLinear=true)
BBSVMParam = BSVMParameters(Stochastic=false,Sparse=false,ALR=false,main_param=main_param)
SBSVMParam = BSVMParameters(Stochastic=true,Sparse=true,ALR=true,main_param=main_param)
LogRegParam = LogRegParameters(main_param=main_param)
SVGPCParam = SVGPCParameters(Stochastic=true,Sparse=true,main_param=main_param)


#Set of all models
TestModels = Dict{String,TestingModel}()

if doBXGPC; TestModels["BXGPC"] = TestingModel("BXGPC",DatasetName,ExperimentName,"BXGPC",BXGPCParam); end;
if doSXGPC; TestModels["SXGPC"] = TestingModel("SXGPC",DatasetName,ExperimentName,"SXGPC",SXGPCParam); end;
if doLBSVM; TestModels["LBSVM"] = TestingModel("LBSVM",DatasetName,ExperimentName,"LBSVM",BBSVMParam); end;
if doBBSVM; TestModels["BBSVM"] = TestingModel("BBSVM",DatasetName,ExperimentName,"BBSVM",BBSVMParam); end;
if doSBSVM; TestModels["SBSVM"] = TestingModel("SBSVM",DatasetName,ExperimentName,"SBSVM",SBSVMParam); end;
if doLogReg; TestModels["LogReg"] = TestingModel("LogReg",DatasetName,ExperimentName,"LogReg",LogRegParam); end;
if doSVGPC;   TestModels["SVGPC"]   = TestingModel("SVGPC",DatasetName,ExperimentName,"SVGPC",SVGPCParam);      end;

writing_order = Array{String,1}();                    if doTime; push!(writing_order,"time"); end;
if doAccuracy; push!(writing_order,"accuracy"); end;  if doBrierScore; push!(writing_order,"brierscore"); end;
if doLogScore; push!(writing_order,"-logscore"); end;  if doAUCScore; push!(writing_order,"AUCscore"); end;
if doLikelihoodScore; push!(writing_order,"medianlikelihoodscore"); push!(writing_order,"meanlikelihoodscore"); end;
for (name,testmodel) in TestModels
  println("Running $(testmodel.MethodName) on $(testmodel.DatasetName) dataset")
  #Initialize the results storage
  testmodel.Model = Array{Any}(nFold)
  if Experiment == ConvergenceExp
      testmodel.Results["Time"] = Array{Any}(nFold);
      testmodel.Results["Accuracy"] = Array{Any}(nFold);
      testmodel.Results["MeanL"] = Array{Any}(nFold);
      testmodel.Results["MedianL"] = Array{Any}(nFold);
      testmodel.Results["ELBO"] = Array{Any}(nFold);
      testmodel.Results["Param"] = Array{Any}(nFold);
      testmodel.Results["Coeff"] = Array{Any}(nFold);
  else
      if doTime;        testmodel.Results["time"]       = Array{Float64,1}(nFold);end;
      if doAccuracy;    testmodel.Results["accuracy"]   = Array{Float64,1}(nFold);end;
      if doBrierScore;  testmodel.Results["brierscore"] = Array{Float64,1}(nFold);end;
      if doLogScore;    testmodel.Results["-logscore"]   = Array{Float64,1}(nFold);end;
      if doAUCScore;    testmodel.Results["AUCscore"]   = Array{Float64,1}(nFold);end;
      if doLikelihoodScore;  testmodel.Results["medianlikelihoodscore"] = Array{Float64,1}(nFold);
                        testmodel.Results["meanlikelihoodscore"] = Array{Float64,1}(nFold);end;
  end
 for i in 1:iFold #Run over all folds of the data
    if ShowIntResults
      println("#### Fold number $i/$nFold###")
    end
    X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
    y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
    if (length(y_test) > 100000 )
        subset = StatsBase.sample(1:length(y_test),100000,replace=false)
        X_test = X_test[subset,:];
        y_test = y_test[subset];
    end
    X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
    y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
    init_t = time_ns()
    CreateModel!(testmodel,i,X,y)
    if Experiment == PredictionExp
        TrainModel!(testmodel,i,X,y,X_test,y_test,MaxIter)
        tot_time = (time_ns()-init_t)*1e-9
        if doTime; testmodel.Results["time"][i] = tot_time; end;
        RunTests(testmodel,i,X,X_test,y_test,accuracy=doAccuracy,brierscore=doBrierScore,logscore=doLogScore,AUCscore=doAUCScore,likelihoodscore=doLikelihoodScore)
    elseif Experiment == ConvergenceExp
        LogArrays= hcat(TrainModelwithTime!(testmodel,i,X,y,X_test,y_test,MaxIter,iter_points)...)
        testmodel.Results["Time"][i] = TreatTime(init_t,LogArrays[1,:],LogArrays[6,:])
        testmodel.Results["Accuracy"][i] = LogArrays[2,:]
        testmodel.Results["MeanL"][i] = LogArrays[3,:]
        testmodel.Results["MedianL"][i] = LogArrays[4,:]
        testmodel.Results["ELBO"][i] = LogArrays[5,:]
        testmodel.Results["Param"][i] = LogArrays[7,:]
        testmodel.Results["Coeff"][i] = LogArrays[8,:]
    end
    if ShowIntResults
        println("$(testmodel.MethodName) : Time  = $((time_ns()-init_t)*1e-9)s")
    end
    if doSaveLastState
        WriteLastStateParameters(testmodel,X_test,y_test,i)
    end
    #Reset the kernel
    if testmodel.MethodName == "SVGPC"
        rbf = testmodel.Model[i][:kern][:rbf]
        println("SVGPC : params : $(rbf[:lengthscales][:value]) and coeffs : $(rbf[:variance][:value])")
        testmodel.Param["Kernel"] = gpflow.kernels[:Add]([gpflow.kernels[:RBF](main_param["nFeatures"],lengthscales=main_param["Θ"],ARD=false),gpflow.kernels[:White](input_dim=main_param["nFeatures"],variance=main_param["γ"])])
    elseif testmodel.MethodName == "SXGPC"
        println("SXGPC : params : $(broadcast(getfield,testmodel.Model[i].Kernels,:param)), and coeffs $(broadcast(getfield,testmodel.Model[i].Kernels,:coeff))")
    end
  end
  if Experiment != ConvergenceExp
      ProcessResults(testmodel,writing_order) #Compute mean and std deviation
      PrintResults(testmodel.Results["allresults"],testmodel.MethodName,writing_order) #Print the Results in the end
  else
      ProcessResultsConvergence(testmodel,iFold)
      println(size(testmodel.Results["Processed"]))
  end
  if doWrite
    top_fold = "data";
    if !isdir(top_fold); mkdir(top_fold); end;
    WriteResults(testmodel,top_fold,writing_order) #Write the results in an adapted format into a folder
  end
end
if doPlot
    if Experiment != ConvergenceExp
        PlotResults(TestModels,writing_order)
    else
        PlotResultsConvergence(TestModels)
    end
end
# end
