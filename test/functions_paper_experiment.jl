# Paper_Experiment_Functions.jl
#= ---------------- #
Set of datatype and functions for efficient testing.
# ---------------- =#




# module TestFunctions

include("../src/DataAugmentedModels.jl")
include("../src/ECM.jl")
import DAM
using KMeansModule
using ScikitLearn;
if !isdefined(:SGDClassifier); @sk_import linear_model: SGDClassifier; end;
using PyCall
@pyimport gpflow as gpflow
@pyimport tensorflow
using Distributions
using KernelFunctions
using ECM



# export TestingModel
# export DefaultParameters, XGPCParameters, BSVMParameters, SVGPCParameters, LogRegParameters, ECMParameters, SVMParameters
# export CreateModel, TrainModel, TrainModelwithTime, RunTests, ProcessResults, PrintResults, WriteResults
# export ComputePrediction, ComputePredictionAccuracy

#Datatype for containing the model, its results and its parameters
type TestingModel
  MethodName::String #Name of the method
  DatasetName::String #Name of the dataset
  ExperimentType::String #Type of experiment
  MethodType::String #Type of method used ("SVM","BSVM","ECM","SVGPC")
  Param::Dict{String,Any} #Some paramters to run the method
  Results::Dict{String,Any} #Saved results
  Model::Any
  TestingModel(methname,dataset,exp,methtype) = new(methname,dataset,exp,methtype,Dict{String,Any}(),Dict{String,Any}())
  TestingModel(methname,dataset,exp,methtype,params) = new(methname,dataset,exp,methtype,params,Dict{String,Any}())
end

#Create a default dictionary
function DefaultParameters()
  param = Dict{String,Any}()
  param["ϵ"]= 1e-8 #Convergence criteria
  param["BatchSize"] = 10 #Number of points used for stochasticity
  param["Kernel"] = "rbf" # Kernel function
  param["Θ"] = 1.0 # Hyperparameter for the kernel function
  param["γ"] = 1.0 #Variance of introduced noise
  param["M"] = 32 #Number of inducing points
  param["Window"] = 5 #Number of points used to check convergence (smoothing effect)
  param["Verbose"] = 0 #Verbose
  param["Autotuning"] = false
  param["ConvCriter"] = "HOML"
  param["PointOptimization"] = false
  param["FixedInitialization"] = true
  return param
end

#Create a default parameters dictionary for XGPC
function XGPCParameters(;Stochastic=true,Sparse=true,ALR=true,Autotuning=false,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["Stochastic"] = Stochastic #Is the method stochastic
  param["Sparse"] = Sparse #Is the method using inducing points
  param["ALR"] = ALR #Is the method using adpative learning rate (in case of the stochastic case)
  param["Autotuning"] = main_param["Autotuning"] #Is hyperoptimization performed
  param["PointOptimization"] = main_param["PointOptimization"] #Is hyperoptimization on inducing points performed
  param["ATFrequency"] = 1 #Number of iterations between every autotuning
  param["κ_s"] = 1.0;  param["τ_s"] = 100; #Parameters for learning rate of Stochastic gradient descent when ALR is not used
  param["ϵ"] = main_param["ϵ"]; param["Window"] = main_param["Window"]; #Convergence criteria (checking parameters norm variation on a window)
  param["ConvCriter"] = main_param["ConvCriter"]
  param["Kernels"] = [Kernel(main_param["Kernel"],1.0,params=main_param["Θ"])] #Kernel creation (standardized for now)
  param["Verbose"] = if typeof(main_param["Verbose"]) == Bool; main_param["Verbose"] ? 2 : 0; else; param["Verbose"] = main_param["Verbose"]; end; #Verbose
  param["BatchSize"] = main_param["BatchSize"] #Number of points used for stochasticity
  param["FixedInitialization"] = main_param["FixedInitialization"]
  param["M"] = main_param["M"] #Number of inducing points
  param["γ"] = main_param["γ"] #Variance of introduced noise
  return param
end

#Create a default parameters dictionary for BSVM
function BSVMParameters(;Stochastic=true,NonLinear=true,Sparse=true,ALR=true,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["Stochastic"] = Stochastic #Is the method stochastic
  param["Sparse"] = Sparse #Is the method using inducing points
  param["NonLinear"] = NonLinear #Is the method using kernels
  param["ALR"] = ALR #Is the method using adpative learning rate (in case of the stochastic case)
  param["Autotuning"] = main_param["Autotuning"] #Is hyperoptimization performed
  param["PointOptimization"] = main_param["PointOptimization"] #Is hyperoptimization on inducing points performed
  param["ATFrequency"] = 1 #Number of iterations between every autotuning
  param["κ_s"] = 1.0;  param["τ_s"] = 100; #Parameters for learning rate of Stochastic gradient descent when ALR is not used
  param["ϵ"] = main_param["ϵ"]; param["Window"] = main_param["Window"]; #Convergence criteria (checking parameters norm variation on a window)
  param["ConvCriter"] = main_param["ConvCriter"]
  param["Kernels"] = [Kernel(main_param["Kernel"],1.0,params=main_param["Θ"])] #Kernel creation (standardized for now)
  param["Verbose"] = if typeof(main_param["Verbose"]) == Bool; main_param["Verbose"] ? 2 : 0; else; param["Verbose"] = main_param["Verbose"]; end; #Verbose
  param["BatchSize"] = main_param["BatchSize"] #Number of points used for stochasticity
  param["FixedInitialization"] = main_param["FixedInitialization"]
  param["M"] = main_param["M"] #Number of inducing points
  param["γ"] = main_param["γ"] #Variance of introduced noise
  return param
end

#Create a default parameters dictionary for SVGPC (similar to BSVM)
function SVGPCParameters(;Sparse=true,Stochastic=false,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["Sparse"] = Sparse
  if Sparse
    param["Stochastic"] = Stochastic
  else
    param["Stochastic"] = false
  end
  param["Autotuning"] = main_param["Autotuning"] #Is hyperoptimization performed
  param["PointOptimization"] = main_param["PointOptimization"] #Is hyperoptimization on inducing points performed
  param["ϵ"] = main_param["ϵ"]
  param["Kernel"] = gpflow.kernels[:Add]([gpflow.kernels[:RBF](main_param["nFeatures"],lengthscales=main_param["Θ"],ARD=false),gpflow.kernels[:White](input_dim=main_param["nFeatures"],variance=main_param["γ"])])
  # param["Kernel"] = gpflow.kernels[:Add]([gpflow.kernels[:RBF](main_param["nFeatures"],lengthscales=main_param["Θ"]*ones(main_param["nFeatures"])),gpflow.kernels[:White](input_dim=main_param["nFeatures"],variance=main_param["γ"])])
  param["BatchSize"] = main_param["BatchSize"]
  param["M"] = main_param["M"]
  param["SmoothingWindow"] = main_param["Window"]
  param["ConvCriter"] = main_param["ConvCriter"]
  return param
end

function LogRegParameters(;main_param=DefaultParameters())
    param = Dict{String,Any}()
    param["Penalty"]="l2"
    param["γ"] =main_param["γ"]
    param["ϵ"]=main_param["ϵ"]
    param["ConvCriter"]=main_param["ConvCriter"]
    return param
end


#Create a model given the parameters passed in p
function CreateModel!(tm::TestingModel,i,X,y) #tm testing_model, p parameters
    if tm.MethodType == "BXGPC"
        tm.Model[i] = DAM.BatchXGPC(X,y;Kernels=tm.Param["Kernels"],Autotuning=tm.Param["Autotuning"],AutotuningFrequency=tm.Param["ATFrequency"],ϵ=tm.Param["ϵ"],γ=tm.Param["γ"],
            VerboseLevel=tm.Param["Verbose"],μ_init=tm.Param["FixedInitialization"] ? zeros(y) : [0.0])
    elseif tm.MethodType == "SXGPC"
        tm.Model[i] = DAM.SparseXGPC(X,y;Stochastic=tm.Param["Stochastic"],BatchSize=tm.Param["BatchSize"],m=tm.Param["M"],
            Kernels=tm.Param["Kernels"],Autotuning=tm.Param["Autotuning"],OptimizeIndPoints=tm.Param["PointOptimization"],AutotuningFrequency=tm.Param["ATFrequency"],AdaptiveLearningRate=tm.Param["ALR"],κ_s=tm.Param["κ_s"],τ_s = tm.Param["τ_s"],ϵ=tm.Param["ϵ"],γ=tm.Param["γ"],
            SmoothingWindow=tm.Param["Window"],VerboseLevel=tm.Param["Verbose"],μ_init=tm.Param["FixedInitialization"] ? zeros(tm.Param["M"]) : [0.0])
    elseif tm.MethodType == "LBSVM"
        tm.Model[i] = DAM.LinearBSVM(X,y;Stochastic=tm.Param["Stochastic"],BatchSize=tm.Param["BatchSize"],
            Autotuning=tm.Param["Autotuning"],AutotuningFrequency=tm.Param["ATFrequency"],AdaptiveLearningRate=tm.Param["ALR"],κ_s=tm.Param["κ_s"],τ_s = tm.Param["τ_s"],ϵ=tm.Param["ϵ"],γ=tm.Param["γ"],
            SmoothingWindow=tm.Param["Window"],VerboseLevel=tm.Param["Verbose"],μ_init=tm.Param["FixedInitialization"] ? zeros(y) : [0.0])
    elseif tm.MethodType == "BBSVM"
        tm.Model[i] = DAM.BatchBSVM(X,y;Kernels=tm.Param["Kernels"],Autotuning=tm.Param["Autotuning"],AutotuningFrequency=tm.Param["ATFrequency"],ϵ=tm.Param["ϵ"],γ=tm.Param["γ"],
            VerboseLevel=tm.Param["Verbose"],μ_init=tm.Param["FixedInitialization"] ? zeros(y) : [0.0])
    elseif tm.MethodType == "SBSVM"
        tm.Model[i] = DAM.SparseBSVM(X,y;Stochastic=tm.Param["Stochastic"],BatchSize=tm.Param["BatchSize"],m=tm.Param["M"],
            Kernels=tm.Param["Kernels"],Autotuning=tm.Param["Autotuning"],OptimizeIndPoints=tm.Param["PointOptimization"],AutotuningFrequency=tm.Param["ATFrequency"],
            AdaptiveLearningRate=tm.Param["ALR"],κ_s=tm.Param["κ_s"],τ_s = tm.Param["τ_s"],ϵ=tm.Param["ϵ"],γ=tm.Param["γ"],
            SmoothingWindow=tm.Param["Window"],VerboseLevel=tm.Param["Verbose"],μ_init=tm.Param["FixedInitialization"] ? zeros(tm.Param["M"]) : [0.0])
    elseif tm.MethodType == "SVGPC"
        if tm.Param["Sparse"]
            if tm.Param["Stochastic"]
                #Stochastic Sparse SVGPC model
                tm.Model[i] = gpflow.svgp[:SVGP](X, reshape((y+1)./2,(length(y),1)),kern=gpflow.kernels[:Add]([gpflow.kernels[:RBF](main_param["nFeatures"],lengthscales=main_param["Θ"],ARD=false),gpflow.kernels[:White](input_dim=main_param["nFeatures"],variance=main_param["γ"])]), likelihood=gpflow.likelihoods[:Bernoulli](), Z=KMeansInducingPoints(X,tm.Param["M"],10), minibatch_size=tm.Param["BatchSize"])
            else
                #Sparse SVGPC model
                tm.Model[i] = gpflow.svgp[:SVGP](X, reshape((y+1)./2,(size(y,1),1)),kern=gpflow.kernels[:Add]([gpflow.kernels[:RBF](main_param["nFeatures"],lengthscales=main_param["Θ"],ARD=false),gpflow.kernels[:White](input_dim=main_param["nFeatures"],variance=main_param["γ"])]), likelihood=gpflow.likelihoods[:Bernoulli](), Z=KMeansInducingPoints(X,tm.Param["M"],10))
            end
            if !tm.Param["PointOptimization"]
                tm.Model[i][:Z][:fixed]=true;
            end
            if !tm.Param["Autotuning"]
                tm.Model[i][:kern][:fixed]=true;
            end
        else
            #Basic SVGPC model
            tm.Model[i] = gpflow.vgp[:VGP](X, reshape((y+1)./2,(size(y,1),1)),kern=deepcopy(tm.Param["Kernel"]),likelihood=gpflow.likelihoods[:Bernoulli]())
            if !tm.Param["Autotuning"]
                tm.Model[i][:kern][:fixed]=true;
            end
        end
    elseif tm.MethodType == "LogReg"
        tm.Results["Time"][i]=[time_ns()]
        tm.Model[i] = SGDClassifier(loss="log", penalty="l2", alpha=tm.Param["γ"],
         fit_intercept=true, tol=tm.Param["ϵ"], shuffle=true,
          n_jobs=1, learning_rate="optimal" ,warm_start=false)
        push!(tm.Results["Time"][i],time_ns())
    end
end

function LogLikeConvergence(model::DAM.AugmentedModel,iter::Integer,X_test,y_test)
    if iter==1
        push!(model.evol_conv,Inf)
        y_p = model.predictproba(X_test)
        loglike = zeros(y_p)
        loglike[y_test.==1] = log.(y_p[y_test.==1])
        loglike[y_test.==-1] = log.(1-y_p[y_test.==-1])
        new_params = mea2n(loglike)
        model.prev_params = new_params
        return Inf
    end
    if !model.Stochastic || iter%10 == 0
        y_p = model.predictproba(X_test)
        loglike = zeros(y_p)
        loglike[y_test.==1] = log.(y_p[y_test.==1])
        loglike[y_test.==-1] = log.(1-y_p[y_test.==-1])
        new_params = mean(loglike)
        push!(model.evol_conv,abs(new_params-model.prev_params)/((abs(model.prev_params)+abs(new_params))/2.0))
        model.prev_params = new_params
    elseif model.Stochastic
        return 1
    end
    if model.Stochastic
        return mean(model.evol_conv[max(1,length(model.evol_conv)-model.SmoothingWindow+1):end])
    else
        return model.evol_conv[end]
    end
end


#train the model on trainin set (X,y) for #iterations
function TrainModel!(tm::TestingModel,i,X,y,X_test,y_test,iterations)
  if typeof(tm.Model[i]) <: DAM.AugmentedModel
      # tm.Model[i].train(iterations=iterations,convergence=function (model::AugmentedModel,iter) return LogLikeConvergence(model,iter,X_test,y_test);end;)
    tm.Model[i].train(iterations=iterations)
  elseif tm.MethodType == "SVGPC"
      prev= tm.Param["ConvCriter"] == "HOML" ? -Inf : Inf*ones(2*tm.Param["M"])
      logconv = Array{Float64,1}()
      convfrequency = tm.Param["ConvCriter"] == "HOML" ? 10 : 1
      @pydef type Logger
          __init__(self) = (self[:i] = 1)
          convcriter(self,x) =  begin
              if self[:i]%convfrequency == 0
                  if tm.Param["ConvCriter"] == "HOML"
                    y_p = tm.Model[i].predictproba(X_test)
                    loglike = zeros(y_p)
                    loglike[y_test.==1] = log.(y_p[y_test.==1])
                    loglike[y_test.==-1] = log.(1-y_p[y_test.==-1])
                    new_mean = mean(loglike)
                    conv = abs(new_mean-prev)/((abs(prev)+abs(new_mean))/2.0)
                    push!(logconv,conv)
                else
                    q_new = [tm.Model[i][:q_mu][:value][:,1];diag(tm.Model[i][:q_sqrt][:value][:,:,1])]
                    conv = mean(abs.(q_new-q_prev)./((abs.(prev)+abs.(q_new))/2.0))
                    prev[:] = q_new
                    push!(logconv,conv)
                end
                  averaged_conv = mean(logconv[max(1,self[:i]-tm.Param["SmoothingWindow"]+1):self[:i]])
                  if averaged_conv < tm.Param["ϵ"]
                      return false
                  end
              end #endif
              self[:i]+=1
              return true
          end
      end
      a = Logger()
    tm.Model[i][:optimize](maxiter=iterations,callback=a[:convcriter],tensorflow.train[:AdamOptimizer]())
    println("Training ended after $(tm.Model[:num_fevals]) iterations")
  elseif tm.MethodType == "LogReg"
    tm.Model[i][:max_iter] = iterations
    tm.Model[i][:fit](X,y)
  elseif tm.MethodType == "SVM"
    tm.Model[i][:max_iter] = iterations
    tm.Model[i][:fit](X,y)
  elseif tm.MethodType == "ECM"
    tm.Model[i] = ECMTraining(X,y,γ=tm.Param["γ"],nepochs=iterations,ϵ=tm.Param["ϵ"],kernel=tm.Param["Kernel"],verbose=tm.Param["Verbose"])
  end
end

function InitConvergence(tm,i)
    tm.Results["Accuracy"][i] = Array{Float64,1}()
    tm.Results["MeanL"][i] = Array{Float64,1}()
    tm.Results["MedianL"][i] = Array{Float64,1}()
    tm.Results["ELBO"][i] = Array{Float64,1}()
    tm.Results["Param"][i] = Array{Float64,1}()
    tm.Results["Coeff"][i] = Array{Float64,1}()
end

#Compute interesting value for non GP models
function ConvergenceTest(tm,i,X_test,y_test;X=0)
    y_p = ComputePredictionAccuracy(tm,i,X,X_test)
    loglike = zeros(y_p)
    for i in 1:length(y_p)
        if y_test[i] == 1
            loglike[i] = y_p[i] <= 0 ? -Inf : log.(y_p[i])
        elseif y_test[i] == -1
            loglike[i] = y_p[i] >= 1 ? -Inf : log.(1-y_p[i])
        end
    end
    push!(tm.Results["Accuracy"][i],TestAccuracy(y_test,sign.(y_p-0.5)))
    push!(tm.Results["MeanL"][i],TestMeanHoldOutLikelihood(loglike))
    push!(tm.Results["MedianL"][i],TestMedianHoldOutLikelihood(loglike))
    push!(tm.Results["ELBO"][i],1.0)
    push!(tm.Results["Param"][i],1.0)
    push!(tm.Results["Coeff"][i],1.0)
end

function TrainModelwithTime!(tm::TestingModel,i,X,y,X_test,y_test,iterations,iter_points)
    LogArrays = Array{Any,1}()
    if typeof(tm.Model[i]) <: DAM.AugmentedModel
        function LogIt(model::DAM.AugmentedModel,iter)
            if in(iter,iter_points)
                a = zeros(8)
                a[1] = time_ns()
                y_p = model.predictproba(X_test)
                loglike = zeros(y_p)
                for i in 1:length(y_p)
                    if y_test[i] == 1
                        loglike[i] = y_p[i] <= 0 ? -Inf : log.(y_p[i])
                    elseif y_test[i] == -1
                        loglike[i] = y_p[i] >= 1 ? -Inf : log.(1-y_p[i])
                    end
                end
                # loglike[y_test.==1] = log.(y_p[y_test.==1])
                # loglike[y_test.==-1] = log.(1-y_p[y_test.==-1])
                a[2] = TestAccuracy(y_test,sign.(y_p-0.5))
                a[3] = TestMeanHoldOutLikelihood(loglike)
                a[4] = TestMedianHoldOutLikelihood(loglike)
                a[5] = DAM.ELBO(model)
    #            println("Iteration $iter : Acc is $(a[2]), MedianL is $(a[4]), ELBO is $(a[5]) θ is $(model.Kernels[1].param)")
                a[6] = time_ns()
                a[7] = model.Kernels[1].param
                a[8] = model.Kernels[1].coeff
                push!(LogArrays,a)
            end
        end
      tm.Model[i].train(iterations=iterations,callback=LogIt)
    elseif tm.MethodType == "SVGPC"
      @pydef type Logger
          __init__(self) = (self[:i] = 1)
          getlog(self,x) =  begin
              if in(self[:i],iter_points)
                  a = zeros(8)
                  a[1] = time_ns()
                  y_p = tm.Model[i][:predict_y](X_test)[1]
                  loglike = zeros(y_p)
                  loglike[y_test.==1] = log.(y_p[y_test.==1])
                  loglike[y_test.==-1] = log.(1-y_p[y_test.==-1])
                  a[2] = TestAccuracy(y_test,sign.(y_p-0.5))
                  a[3] = TestMeanHoldOutLikelihood(loglike)
                  a[4] = TestMedianHoldOutLikelihood(loglike)
                  a[5] = tm.Model[i][:_objective](x)[1]
                  # println("Iteration $(self[:i]) : Acc is $(a[2]), MedianL is $(a[4]), ELBO is $(a[5]) mean(θ) is $(mean(tm.Model[i][:kern][:rbf][:lengthscales][:value]))")
                  a[6] = time_ns()
                  a[7] = tm.Model[i][:kern][:rbf][:lengthscales][:value][1]
                  a[8] = tm.Model[i][:kern][:rbf][:variance][:value][1]
                  push!(LogArrays,a)
                  # println((a[1]-LogArrays[1][1])*1e-9)
                  if (a[1]-LogArrays[1][1])*1e-9 > 4000
                      return false
                  end
              end
              self[:i]+=1
              return true
          end
          gethyper(self,x) = begin
              if self[:i] == 1
                  self[:time] = time_ns()
                  self[:i] += 1
              end
              if floor(Int64,(time_ns()-self[:time])*1e-9)÷60==(self[:i]-1)
                  y_p = tm.Model[i][:predict_y](X_test)[1]
                  acc = TestAccuracy(y_test,sign.(y_p-0.5))
                  theta = tm.Model[i][:kern][:rbf][:lengthscales][:value]
                  writedlm("KernelParams"*tm.DatasetName,theta)
                  println("Time : $(floor(Int64,(time_ns()-self[:time])*1e-9)÷60), ledua accuracy is $acc and kernel params are $theta")
                  self[:i]+=1
              end
              return true
          end
      end

      loggerobject = Logger();
    #   tm.Model[i][:optimize](maxiter=iterations,callback=loggerobject[:gethyper],method=tensorflow.train[:AdamOptimizer]())
      tm.Model[i][:optimize](maxiter=iterations,callback=loggerobject[:getlog],method=tensorflow.train[:AdamOptimizer]())
    elseif tm.MethodType == "LogReg"
        InitConvergence(tm,i)
        ConvergenceTest(tm,i,X_test,y_test)
        tm.Model[i][:max_iter] = iterations
        tm.Model[i][:fit](X,y)
        push!(tm.Results["Time"][i],time_ns())
        ConvergenceTest(tm,i,X_test,y_test)
        tm.Results["Time"][i] = (tm.Results["Time"][i][2:3]-tm.Results["Time"][i][1])*1e-9
    elseif tm.MethodType == "SVM"
        warn("Not available for libSVM")
    elseif tm.MethodType == "ECM"
        warn("Not available for ECM")
    end
    return LogArrays
end

function TreatTime(init_time,before_comp,after_comp)
    before_comp = before_comp - init_time; after_comp = after_comp - init_time;
    diffs = after_comp-before_comp;
    for i in 2:length(before_comp)
        before_comp[i:end] -= diffs[i-1]
    end
    return before_comp*1e-9
end

#Run tests accordingly to the arguments and save them
function RunTests(tm::TestingModel,i,X,X_test,y_test;accuracy::Bool=true,brierscore::Bool=true,logscore::Bool=true,AUCscore::Bool=true,likelihoodscore::Bool=true,npoints::Integer=500)
  if accuracy
    tm.Results["Accuracy"][i]=TestAccuracy(y_test,ComputePrediction(tm,i,X,X_test))
  end
  y_predic_acc = 0
  if brierscore
    y_predic_acc = ComputePredictionAccuracy(tm,i, X, X_test)
    tm.Results["Brierscore"][i] = TestBrierScore(y_test,y_predic_acc)
  end
  if logscore
    if y_predic_acc == 0
      y_predic_acc = ComputePredictionAccuracy(tm,i, X, X_test)
    end
    tm.Results["-Logscore"][i]=TestLogScore(y_test,y_predic_acc)
  end
  if AUCscore
    if y_predic_acc == 0
        y_predic_acc = ComputePredictionAccuracy(tm,i,X,X_test)
    end
    tm.Results["AUCscore"][i] = TestAUCScore(ROC(y_test,y_predic_acc,npoints))
  end
  if likelihoodscore
      if y_predic_acc == 0
          y_predic_acc = ComputePredictionAccuracy(tm,i,X,X_test)
      end
      tm.Results["-MedianL"][i] = -TestMedianHoldOutLikelihood(HoldOutLikelihood(y_test,y_predic_acc))
      tm.Results["-MeanL"][i] = -TestMeanHoldOutLikelihood(HoldOutLikelihood(y_test,y_predic_acc))
  end
end


#Compute the mean and the standard deviation and assemble in one result
function ProcessResults(tm::TestingModel,writing_order)
  all_results = Array{Float64,1}()
  names = Array{String,1}()
  for name in writing_order
    result = [mean(tm.Results[name]), std(tm.Results[name])]
    all_results = vcat(all_results,result)
    names = vcat(names,name)
  end
  if haskey(tm.Results,"allresults")
    tm.Results["allresults"] = vcat(tm.Results["allresults"],all_results')
  else
    tm.Results["allresults"] = all_results'
  end
  if !haskey(tm.Results,"names")
    tm.Results["names"] = names
  end
end

function ProcessResultsConvergence(tm::TestingModel,iFold)
    #Find maximum length
    NMax = maximum(length.(tm.Results["Time"][1:iFold]))
    NFolds = length(tm.Results["Time"][1:iFold])
    Mtime = zeros(NMax); time= []
    Macc = zeros(NMax); acc= []
    Mmeanl = zeros(NMax); meanl= []
    Mmedianl = zeros(NMax); medianl= []
    Melbo = zeros(NMax); elbo = []
    Mparam = zeros(NMax); param = []
    Mcoeff = zeros(NMax); coeff = []
    for i in 1:iFold
        DiffN = NMax - length(tm.Results["Time"][i])
        if DiffN != 0
            time = [tm.Results["Time"][i];tm.Results["Time"][i][end]*ones(DiffN)]
            acc = [tm.Results["Accuracy"][i];tm.Results["Accuracy"][i][end]*ones(DiffN)]
            meanl = [tm.Results["MeanL"][i];tm.Results["MeanL"][i][end]*ones(DiffN)]
            medianl = [tm.Results["MedianL"][i];tm.Results["MedianL"][i][end]*ones(DiffN)]
            elbo = [tm.Results["ELBO"][i];tm.Results["ELBO"][i][end]*ones(DiffN)]
            param = [tm.Results["Param"][i];tm.Results["Param"][i][end]*ones(DiffN)]
            coeff = [tm.Results["Coeff"][i];tm.Results["Coeff"][i][end]*ones(DiffN)]
        else
            time = tm.Results["Time"][i];
            acc = tm.Results["Accuracy"][i];
            meanl = tm.Results["MeanL"][i];
            medianl = tm.Results["MedianL"][i];
            elbo = tm.Results["ELBO"][i];
            param = tm.Results["Param"][i];
            coeff = tm.Results["Coeff"][i];
        end
        Mtime = hcat(Mtime,time)
        Macc = hcat(Macc,acc)
        Mmeanl = hcat(Mmeanl,meanl)
        Mmedianl = hcat(Mmedianl,medianl)
        Melbo = hcat(Melbo,elbo)
        Mparam = hcat(Mparam,param)
        Mcoeff = hcat(Mcoeff,coeff)
    end
    Mtime = Mtime[:,2:end]; Macc = Macc[:,2:end]
    Mmeanl = Mmeanl[:,2:end]; Mmedianl = Mmedianl[:,2:end]
    Melbo = Melbo[:,2:end];
    Mparam = Mparam[:,2:end]; Mcoeff = Mcoeff[:,2:end]
    tm.Results["Time"] = Mtime;
    tm.Results["Accuracy"] = Macc;
    tm.Results["MeanL"] = Mmeanl
    tm.Results["MedianL"] = Mmedianl
    tm.Results["ELBO"] = Melbo
    tm.Results["Param"] = Mparam
    tm.Results["Coeff"] = Mcoeff
    tm.Results["Processed"]= [vec(mean(Mtime,2)) vec(std(Mtime,2)) vec(mean(Macc,2)) vec(std(Macc,2)) vec(mean(Mmeanl,2)) vec(std(Mmeanl,2)) vec(mean(Mmedianl,2)) vec(std(Mmedianl,2)) vec(mean(Melbo,2)) vec(std(Melbo,2)) vec(mean(Mparam,2)) vec(std(Mparam,2)) vec(mean(Mcoeff,2)) vec(std(Mcoeff,2))]
end

function PrintResults(results,method_name,writing_order)
  println("Model $(method_name) : ")
  i = 1
  for category in writing_order
    println("$category : $(results[i*2-1]) ± $(results[i*2])")
    i+=1
  end
end

function WriteResults(tm::TestingModel,location,writing_order)
  fold = String(location*"/"*tm.ExperimentType*"Experiment"*(Experiment==AccuracyExp ?"eps$(tm.Param["ϵ"])":""))
  if !isdir(fold); mkdir(fold); end;
  fold = fold*"/"*tm.DatasetName*"Dataset"
  labels=Array{String,1}(length(writing_order)*2)
  labels[1:2:end-1,:] = writing_order.*"_mean"
  labels[2:2:end,:] =  writing_order.*"_std"
  if !isdir(fold); mkdir(fold); end;
  if Experiment == AccuracyExp
      writedlm(String(fold*"/Results_"*tm.MethodName*".txt"),tm.Results["allresults"])
  elseif Experiment == IndPointsExp
      fold = fold*"/$(tm.Param["M"])Points"
      if !isdir(fold); mkdir(fold);end;
      writedlm(String(fold*"/Results_"*tm.MethodName*".txt"),tm.Results["Processed"])
  else
      writedlm(String(fold*"/Results_"*tm.MethodName*".txt"),tm.Results["Processed"])
  end
end

#Return predicted labels (-1,1) for test set X_test
function ComputePrediction(tm::TestingModel, i,X, X_test)
  y_predic = []
  if typeof(tm.Model[i]) <: DAM.AugmentedModel
    y_predic = sign.(tm.Model[i].predict(X_test))
  elseif tm.MethodType == "SVGPC"
    y_predic = sign.(tm.Model[i][:predict_y](X_test)[1]*2-1)
  elseif tm.MethodType == "LogReg"
    y_predic = sign.(tm.Model[i][:predict](X_test))
  elseif tm.MethodType == "SVM"
    y_predic = sign.(tm.Model[i][:predict](X_test))
  elseif tm.MethodType == "ECM"
    y_predic = sign.(PredicECM(X,tm.Model[i][4],X_test,tm.Model[i][1],tm.Model[i][2],tm.Param["γ"],tm.Model[i][3]))
  end
  return y_predic
end

#Return prediction certainty for class 1 on test set X_test
function ComputePredictionAccuracy(tm::TestingModel,i, X, X_test)
  y_predic = []
  if typeof(tm.Model[i]) <: DAM.AugmentedModel
    y_predic = tm.Model[i].predictproba(X_test)
  elseif tm.MethodType == "SVGPC"
    y_predic = tm.Model[i][:predict_y](X_test)[1]
  elseif tm.MethodType == "LogReg"
    y_predic = tm.Model[i][:predict_proba](X_test)[:,2]
  elseif tm.MethodType == "SVM"
    y_predic = tm.Model[i][:predict_proba](X_test)[:,2]
  elseif tm.MethodType == "ECM"
    y_predic = PredictProbaECM(X,tm.Model[i][4],X_test,tm.Model[i][1],tm.Model[i][2],tm.Param["γ"],tm.Model[i][3])
  end
  return y_predic
end

#Return Accuracy on test set
function TestAccuracy(y_test, y_predic)
  return 1-sum(1-y_test.*y_predic)/(2*length(y_test))
end
#Return Brier Score
function TestBrierScore(y_test, y_predic)
  return sum(((y_test+1)./2 - y_predic).^2)/length(y_test)
end
#Return Log Score
function TestLogScore(y_test, y_predic)
  return -sum((y_test+1)./2.*log.(y_predic)+(1-(y_test+1)./2).*log.(1-y_predic))/length(y_test)
end

function TestMeanHoldOutLikelihood(loglike)
    return mean(loglike)
end

function TestMedianHoldOutLikelihood(loglike)
    return median(loglike)
end

function HoldOutLikelihood(y_test,y_predic)
    loglike = zeros(y_predic)
    loglike[y_test.==1] = log.(y_predic[y_test.==1])
    loglike[y_test.==-1] = log.(1-y_predic[y_test.==-1])
    return loglike
end


#Return ROC
function ROC(y_test,y_predic,npoints)
    nt = length(y_test)
    truepositive = zeros(npoints); falsepositive = zeros(npoints)
    truenegative = zeros(npoints); falsenegative = zeros(npoints)
    thresh = collect(linspace(0,1,npoints))
    for i in 1:npoints
      for j in 1:nt
        truepositive[i] += (y_predic[j]>=thresh[i] && y_test[j]>=0.9) ? 1 : 0;
        truenegative[i] += (y_predic[j]<=thresh[i] && y_test[j]<=-0.9) ? 1 : 0;
        falsepositive[i] += (y_predic[j]>=thresh[i] && y_test[j]<=-0.9) ? 1 : 0;
        falsenegative[i] += (y_predic[j]<=thresh[i] && y_test[j]>=0.9) ? 1 : 0;
      end
    end
    return (truepositive./(truepositive+falsenegative),falsepositive./(truenegative+falsepositive))
end

function TestAUCScore(ROC)
    (sensibility,specificity) = ROC
    h = specificity[1:end-1]-specificity[2:end]
    AUC = sum(sensibility[1:end-1].*h)
    return AUC
end

function WriteLastStateParameters(testmodel,top_fold,X_test,y_test,i)
    if isa(testmodel.Model[i],DAM.AugmentedModel)
        if !isdir(top_fold); mkdir(top_fold); end;
        top_fold = top_fold*"/"*testmodel.DatasetName*"_SavedParams"
        if !isdir(top_fold); mkdir(top_fold); end;
        top_fold = top_fold*"/"*testmodel.MethodName
        if !isdir(top_fold); mkdir(top_fold); end;
        writedlm(top_fold*"/mu"*"_$i",testmodel.Model[i].μ)
        writedlm(top_fold*"/sigma"*"_$i",testmodel.Model[i].ζ)
        writedlm(top_fold*"/c"*"_$i",testmodel.Model[i].α)
        writedlm(top_fold*"/X_test"*"_$i",X_test)
        writedlm(top_fold*"/y_test"*"_$i",y_test)
        if isa(testmodel.Model[i],DAM.SparseModel)
            writedlm(top_fold*"/ind_points"*"_$i",testmodel.Model[i].inducingPoints)
        end
        if isa(testmodel.Model[i],DAM.NonLinearModel)
            writedlm(top_fold*"/kernel_param"*"_$i",broadcast(getfield,testmodel.Model[i].Kernels,:param))
            writedlm(top_fold*"/kernel_coeff"*"_$i",broadcast(getfield,testmodel.Model[i].Kernels,:coeff))
            writedlm(top_fold*"/kernel_name"*"_$i",broadcast(getfield,testmodel.Model[i].Kernels,:name))
        end
    end
end

function PlotResults(TestModels,tests)
    nModels=length(TestModels)
    if nModels == 0; return; end;
    bigwidth = 2
    nResults = length(tests)
    ind = 0.1:bigwidth:(nResults-1)*bigwidth+0.1
    width = bigwidth/(nModels+1)
    figure("Final Results");clf();
    colors=["b", "g", "r", "c", "m", "y", "k", "w"]
    iter=0
    for (name,testmodel) in TestModels
        results = testmodel.Results["allresults"]
        bar(ind+iter*width,results[1:2:end]',width,color=colors[iter+1],yerr=results[2:2:end]',label=name)#
        iter+=1
    end
    xticks(ind+nModels/2*width,tests)
    legend()
end

function PlotResultsConvergence(TestModels)
    nModels=length(TestModels)
    if nModels == 0; return; end;
    figure("Convergence Results");clf();
    colors=["b", "r"]
    subplot(3,2,1); #Accuracy
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot(results[1:step:end,1],results[1:step:end,3],color=colors[iter],label=name)
            fill_between(results[1:step:end,1],results[1:step:end,3]-results[1:step:end,4]/sqrt(10),results[1:step:end,3]+results[1:step:end,4]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
    legend()
    xlabel("Time [s]")
    ylabel("Accuracy")
    subplot(3,2,2); #Accuracy
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot(results[1:step:end,1],results[1:step:end,5],color=colors[iter],label=name)
            fill_between(results[1:step:end,1],results[1:step:end,5]-results[1:step:end,6]/sqrt(10),results[1:step:end,5]+results[1:step:end,6]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
    legend()
    xlabel("Time [s]")
    ylabel("Mean Log L")
    subplot(3,2,3); #Accuracy
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot(results[1:step:end,1],results[1:step:end,7],color=colors[iter],label=name)
            fill_between(results[1:step:end,1],results[1:step:end,7]-results[1:step:end,8]/sqrt(10),results[1:step:end,7]+results[1:step:end,8]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
    legend()
    xlabel("Time [s]")
    ylabel("Median Log L")
    subplot(3,2,4); #Accuracy
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot(results[1:step:end,1],results[1:step:end,9],color=colors[iter],label=name)
            fill_between(results[1:step:end,1],results[1:step:end,9]-results[1:step:end,10]/sqrt(10),results[1:step:end,9]+results[1:step:end,10]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
    legend()
    xlabel("Time [s]")
    ylabel("neg. ELBO")
    subplot(3,2,5); #Accuracy
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot(results[1:step:end,1],results[1:step:end,11],color=colors[iter],label=name)
            fill_between(results[1:step:end,1],results[1:step:end,11]-results[1:step:end,12]/sqrt(10),results[1:step:end,11]+results[1:step:end,12]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
    legend()
    xlabel("Time [s]")
    ylabel("Param")
    subplot(3,2,6); #Accuracy
        iter=1
        step =1
        for (name,testmodel) in TestModels
            results = testmodel.Results["Processed"]
            plot(results[1:step:end,1],results[1:step:end,13],color=colors[iter],label=name)
            fill_between(results[1:step:end,1],results[1:step:end,13]-results[1:step:end,14]/sqrt(10),results[1:step:end,13]+results[1:step:end,14]/sqrt(10),alpha=0.2,facecolor=colors[iter])
            iter+=1
        end
    legend()
    xlabel("Time [s]")
    display(ylabel("Coeff"))
end


# end #end of module
