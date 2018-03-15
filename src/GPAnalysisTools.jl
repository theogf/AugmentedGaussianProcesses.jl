module GPAnalysisTools
using ValueHistories



function getLog(model,iter_points)
    metrics = MVHistory()
    function SaveLog(model::GPModel,iter;hyper=false)
        if in(iter,iter_points)
                y_p = model.predictproba(X_test)
                loglike = zeros(y_p)
                loglike[y_test.==1] = log.(y_p[y_test.==1])
                loglike[y_test.==-1] = log.(1-y_p[y_test.==-1])
                push!(metrics,:test_error,iter,sum(1-y_test.*sign.(y_p-0.5))/(2*length(y_test)))
                push!(metrics,:mean_neg_loglikelihood,iter,mean(-loglike))
                push!(metrics,:median_neg_log_likelihood,iter,median(-loglike))
                push!(metrics,:ELBO,iter,GPM.ELBO(model))
        end
    end
    return metrics,SaveLog
end
