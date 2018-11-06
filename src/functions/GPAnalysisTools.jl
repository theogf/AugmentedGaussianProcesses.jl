module GPAnalysisTools
using ValueHistories
using LinearAlgebra
using Statistics
export getLog, getMultiClassLog

"""
    Return a a MVHistory object and callback function for the training of binary classification problems
    The callback will store the ELBO and the variational parameters at every iterations included in iter_points
    If X_test and y_test are provided it will also store the test accuracy and the mean and median test loglikelihood
"""
function (model;X_test=0,y_test=0,iter_points=vcat(1:1:9,10:5:99,100:50:999,1000:100:9999))
    metrics = MVHistory()
    function measuremetrics(model,iter)
        if in(iter,iter_points)
                if X_test!=0
                    y_p = model.predictproba(X_test)
                    loglike = fill!(similar(y_p),0)
                    loglike[y_test.==1] = log.(y_p[y_test.==1])
                    loglike[y_test.==-1] = log.(1.0 .-y_p[y_test.==-1])
                    push!(metrics,:test_error,iter,mean(y_test.==sign.(y_p.-0.5)))
                    push!(metrics,:mean_neg_loglikelihood,iter,mean(-loglike))
                    push!(metrics,:median_neg_log_likelihood,iter,median(-loglike))
                end
                push!(metrics,:ELBO,iter,model.elbo())
                push!(metrics,:mu,iter,model.μ)
                push!(metrics,:sigma,iter,diag(model.Σ))
                push!(metrics,:kernel_variance,iter,getvariance(model.kernel))
                push!(metrics,:kernel_param,iter,getlengthscales(model.kernel))
        end
    end #end SaveLog
    return metrics,SaveLog
end

"""
    Return a a MVHistory object and callback function for the training of multiclass classification problems
    The callback will store the ELBO and the variational parameters at every iterations included in iter_points
    If X_test and y_test are provided it will also store the test accuracy
"""
function getMultiClassLog(model;X_test=0,y_test=0,iter_points=vcat(1:1:9,10:5:99,100:50:999,1000:100:9999))
    metrics = MVHistory()
    function SaveLog(model,iter;hyper=false)
        if in(iter,iter_points)
            y_p = model.predictproba(X_test)
            accuracy = TestAccuracy(model,y_test,y_p)
            loglike = LogLikelihood(model,y_test,y_p)
            push!(metrics,:ELBO,model.elbo())
            push!(metrics,:test_error,1.0-accuracy)
            push!(metrics,:mean_neg_loglikelihood,iter,mean(-loglike))
            push!(metrics,:median_neg_loglikelihood,iter,median(-loglike))
            println("Iteration $iter : acc = $(accuracy)")
        end
    end #end SaveLog
    return metrics,SaveLog
end

"Return the Kullback Leibler divergence for a series of points given the true GPs and predicted ones"
function KLGP(mu,sig,f,sig_f)
    N = length(f)
    tot = 0.5*N*(-log.(sig_f)-1)
    tot += 0.5*sum(log.(sig)+(sig_f+(mu-f).^2)./sig)
    return tot
end

"Return the Jensen Shannon divergence for a series of points given the true GPs and predicted ones"
function JSGP(mu,sig,f,sig_f)
    N = length(f)
    tot = -N*0.25
    tot += 0.125*sum(sig./(sig_f)+(sig_f)./(sig) + (1.0./sig_f+1.0./sig).*((mu-f).^2))
end

"Return Accuracy on test set"
function TestAccuracy(model, y_test, y_predic)
    score = 0
    for i in 1:length(y_test)
        if (model.class_mapping[argmax(y_predic[i])]) == y_test[i]
            score += 1
        end
    end
    return score/length(y_test)
end
"Return the loglikelihood of the test set"
function LogLikelihood(model,y_test,y_predic)
    return [log(y_predic[i][model.ind_mapping[y_t]]) for (i,y_t) in enumerate(y_test)]
end
end
