module GPAnalysisTools
using ValueHistories
using Plots
export getLog, getMultiClassLog
function getLog(model;X_test=0,y_test=0,iter_points=vcat(1:99,100:10:999,1000:100:9999))
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
                    # N_test = Int64(sqrt(size(X_test,1)))
                    # # x_test = linspace(-5.0,5.0,N_test)
                    # maxs = [3.65,3.4]
                    # mins = [-3.25,-2.85]
                    # x1_test = linspace(mins[1],maxs[1],N_test)
                    # x2_test = linspace(mins[2],maxs[2],N_test)
                    # p3=plot(x1_test,x2_test,reshape(y_p,N_test,N_test),t=:contour,fill=true,cbar=false,clims=(0,1),lab="",title="Sparse XGPC")
                    # plot!(model.X[(model.y.==1)[:],1],model.X[(model.y.==1)[:],2],t=:scatter,alpha=0.3,color=:red,markerstrokewidth=0.0,lab="y=1")
                    # plot!(model.X[(model.y.==-1)[:],1],model.X[(model.y.==-1)[:],2],t=:scatter,alpha=0.3,color=:blue,markerstrokewidth=0.0,lab="y=-1")
                    # plot!(model.inducingPoints[:,1],model.inducingPoints[:,2],t=:scatter,lab="inducing points")
                    # display(p3)
                end
                push!(metrics,:ELBO,iter,model.elbo())
                push!(metrics,:mu,iter,model.μ)
                push!(metrics,:sigma,iter,diag(model.ζ))
                push!(metrics,:kernel_weight,iter,getindex(model.kernel.weight.value))
                push!(metrics,:kernel_param,iter,getindex(model.kernel.param[1].value))
        end
    end #end SaveLog
    return metrics,SaveLog
end

function getMultiClassLog(model,X_test=0,y_test=0,iter_points=vcat(1:99,100:10:999,1000:100:9999))
    metrics = MVHistory()
    function SaveLog(model,iter;hyper=false)
        if in(iter,iter_points)
                if X_test!=0
                        y_sparse, = model.predict(X_test)
                        sparse_score=0
                        for (i,pred) in enumerate(y_sparse)
                            if pred == y_test[i]
                                sparse_score += 1
                            end
                        end
                    push!(metrics,:test_error,iter,sparse_score/length(y_test))
                    # push!(metrics,:mean_neg_loglikelihood,iter,mean(-loglike)) #TODO
                    # push!(metrics,:median_neg_log_likelihood,iter,median(-loglike)) #TODO
                end
                push!(metrics,:ELBO,iter,model.elbo())
                push!(metrics,:mu,iter,model.μ)
                push!(metrics,:sigma,iter,diag.(model.ζ))
                push!(metrics,:kernel_param,iter,getindex(model.kernel.param[1].value))
        end
    end #end SaveLog
    return metrics,SaveLog
end

end
