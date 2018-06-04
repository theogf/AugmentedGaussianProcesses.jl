module GPAnalysisTools
using ValueHistories
export getLog
function getLog(model,X_test=0,y_test=0,iter_points=vcat(1:99,100:10:999,1000:100:9999))
    metrics = MVHistory()
    function SaveLog(model,iter;hyper=false)
        if in(iter,iter_points)
                if X_test!=0
                    y_p = model.predictproba(X_test)
                    loglike = zeros(y_p)
                    loglike[y_test.==1] = log.(y_p[y_test.==1])
                    loglike[y_test.==-1] = log.(1-y_p[y_test.==-1])
                    push!(metrics,:test_error,iter,sum(1-y_test.*sign.(y_p-0.5))/(2*length(y_test)))
                    push!(metrics,:mean_neg_loglikelihood,iter,mean(-loglike))
                    push!(metrics,:median_neg_log_likelihood,iter,median(-loglike))
                end
                push!(metrics,:ELBO,iter,model.elbo())
                push!(metrics,:mu,iter,model.μ)
                push!(metrics,:sigma,iter,diag(model.ζ))
                push!(metrics,:kernel_weight,iter,getindex(model.kernel.weight.value))
                push!(metrics,:kernel_param,iter,getindex(model.kernel.param[1].value))
        end
    end
    return metrics,SaveLog
end

end
