#### Paper_Experiment_Predictions ####
# Run on a file and compute accuracy on a nFold cross validation
# Compute also the brier score and the logscore

push!(LOAD_PATH,".")
# if !isdefined(:DataAccess); include("DataAccess.jl"); end;
if !isdefined(:TestFunctions); include("paper_experiment_functions.jl");end;
# using TestFunctions
using PyPlot
using DataAccess
#Compare XGPC,BSVM, SVGPC and Logistic regressions

#Methods and scores to test
doBXGPC = true #Batch XGPC (no sparsity)
doSXGPC = true #Sparse XGPC (sparsity)
doLBSVM = false #Linear BSVM
doBBSVM = false #Batch BSVM
doSBSVM = false #Sparse BSVM
doSVGPC = false #Sparse Variational GPC (Hensmann)
doLogReg = false #Logistic Regression

# ExperimentName = "Prediction"
ExperimentName = "ConvergenceExperiment"
@enum ExperimentType FrameworkExp=0 PredictionExp=1 ConvergenceExp=2
ExperimentTypes = Dict("Frameworktest"=>FrameworkExp, "ConvergenceExperiment"=>ConvergenceExp,"Prediction"=>PredictionExp)
Experiment = ExperimentTypes[ExperimentName]
doTime = true #Return time needed for training
doAccuracy = true #Return Accuracy
doBrierScore = true # Return BrierScore
doLogScore = true #Return LogScore
doAUCScore = true
doLikelihoodScore = true
doWrite = true #Write results in approprate folder
ShowIntResults = true #Show intermediate time, and results for each fold

#Testing Parameters
#= Datasets available are X :
aXa, Bank_marketing, Click_Prediction, Cod-rna, Diabetis, Electricity, German, Shuttle
=#
dataset = "German"
 # (X_data,y_data,DatasetName) = get_BreastCancer()
(X_data,y_data,DatasetName) = get_Dataset(dataset)
MaxIter = 50000 #Maximum number of iterations for every algorithm
iter_points = [1:1:99;100:10:999;1000:100:9999;10000:1000:100000] #Iteration points where measures are taken
(nSamples,nFeatures) = size(X_data);
nFold = 10; #Chose the number of folds
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold

#Main Parameters
main_param = DefaultParameters()
main_param["nFeatures"] = nFeatures
main_param["nSamples"] = nSamples
main_param["ϵ"] = 1e-6 #Convergence criterium
main_param["M"] = 64 #Number of inducing points
main_param["Kernel"] = "rbf"
main_param["Θ"] = 5.0 #Hyperparameter of the kernel
main_param["BatchSize"] = 100
main_param["Verbose"] = 0
main_param["Window"] = 10
#BSVM and SVGPC Parameters
BXGPCParam = XGPCParameters(Stochastic=false,Sparse=false,ALR=true,main_param=main_param)
SXGPCParam = XGPCParameters(Stochastic=true,Sparse=true,ALR=true,main_param=main_param)
LBSVMParam = BSVMParameters(Stochastic=false,NonLinear=true)
BBSVMParam = BSVMParameters(Stochastic=false,Sparse=false,ALR=false,main_param=main_param)
SBSVMParam = BSVMParameters(Stochastic=true,Sparse=true,ALR=true,main_param=main_param)
LogRegParam = LogRegParameters(main_param=main_param)
SVGPCParam = SVGPCParameters(Stochastic=true,Sparse=true,main_param=main_param)

#Global variables for debugging
X = []; y = []; X_test = []; y_test = [];

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
  if Experiment == ConvergenceExp
      testmodel.Results["Time"] = Array{Any,1}();
      testmodel.Results["Accuracy"] = Array{Any,1}();
      testmodel.Results["MeanL"] = Array{Any,1}();
      testmodel.Results["MedianL"] = Array{Any,1}();
      testmodel.Results["ELBO"] = Array{Any,1}();
  else
      if doTime;        testmodel.Results["time"]       = Array{Float64,1}();end;
      if doAccuracy;    testmodel.Results["accuracy"]   = Array{Float64,1}();end;
      if doBrierScore;  testmodel.Results["brierscore"] = Array{Float64,1}();end;
      if doLogScore;    testmodel.Results["-logscore"]   = Array{Float64,1}();end;
      if doAUCScore;    testmodel.Results["AUCscore"]   = Array{Float64,1}();end;
      if doLikelihoodScore;  testmodel.Results["medianlikelihoodscore"] = Array{Float64,1}();
                        testmodel.Results["meanlikelihoodscore"] = Array{Float64,1}();end;
  end
  for i in 1:nFold #Run over all folds of the data
    if ShowIntResults
      println("#### Fold number $i/$nFold ###")
    end

    X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
    y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
    if (length(y_test) > 100000 )
        X_test = X_test[StatsBase.sample(1:length(y_test),100000,replace=false),:];
        y_test = y_test[StatsBase.sample(1:length(y_test),100000,replace=false)];
    end
    X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
    y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
    init_t = time_ns()
    CreateModel!(testmodel,X,y)
    if Experiment == PredictionExp || Experiment == FrameworkExp
        TrainModel!(testmodel,X,y,MaxIter)
        tot_time = (time_ns()-init_t)*1e-9
        if doTime; push!(testmodel.Results["time"],tot_time); end;
        RunTests(testmodel,X,X_test,y_test,accuracy=doAccuracy,brierscore=doBrierScore,logscore=doLogScore,AUCscore=doAUCScore,likelihoodscore=doLikelihoodScore)
    elseif Experiment == ConvergenceExp
        LogArrays= hcat(TrainModelwithTime!(testmodel,X,y,X_test,y_test,MaxIter,iter_points)...)
        push!(testmodel.Results["Time"],TreatTime(init_t,LogArrays[1,:],LogArrays[6,:]))
        push!(testmodel.Results["Accuracy"],LogArrays[2,:])
        push!(testmodel.Results["MeanL"],LogArrays[3,:])
        push!(testmodel.Results["MedianL"],LogArrays[4,:])
        push!(testmodel.Results["ELBO"],LogArrays[5,:])
    end
    if ShowIntResults
       println("$(testmodel.MethodName) : Time  = $(logt[end])")
    end
  end
  if Experiment != ConvergenceExp
      ProcessResults(testmodel,writing_order) #Compute mean and std deviation
      PrintResults(testmodel.Results["allresults"],testmodel.MethodName,writing_order) #Print the Results in the end
  else
      ProcessResultsConvergence(testmodel)
  end
  if doWrite
    top_fold = "data_M$(main_param["M"])";
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
