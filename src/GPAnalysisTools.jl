module GPAnalysisTools
using ValueHistories
using Plots
export getLog, getMultiClassLog, IntermediatePlotting
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

function KLGP(mu,sig,f,sig_f)
    N = length(f)
    tot = 0.5*N*(-log.(sig_f)-1)
    tot += 0.5*sum(log.(sig)+(sig_f+(mu-f).^2)./sig)
    return tot
end

function JSGP(mu,sig,f,sig_f)
    N = length(f)
    tot = -N*0.25
    tot += 0.125*sum(sig./(sig_f)+(sig_f)./(sig) + (1./(sig_f)+1./(sig)).*((mu-f).^2))
end

function plotting1D(iter,indices,X,f,sig_f,ind_points,pred_ind,X_test,pred,sig_pred,y_train,sig_train,title,sequential=false)
    if sequential
        p = plot!(X[indices,1],f[indices],t=:scatter,lab="",alpha=0.6,color=:red,markerstrokewidth=0)
        p = plot(X[1:(indices[1]-1),1],f[1:(indices[1]-1)],t=:scatter,lab="",color=:blue,alpha=0.4,markerstrokewidth=0)
        p = plot!(X[(indices[1]+1):end,1],f[(indices[1]+1):end],t=:scatter,lab="",color=:blue,alpha=0.1,markerstrokewidth=0)
    else
        p = plot(X,f,t=:scatter,lab="",color=:blue,alpha=0.1,markerstrokewidth=0)
        p = plot!(X[indices,1],f[indices],t=:scatter,lab="",alpha=1.0,color=:red,markerstrokewidth=0)
    end
    p = plot!(ind_points[:,1],pred_ind,t=:scatter,lab="",color=:green)
    p = plot!(X_test,pred+3*sqrt.(sig_pred),fill=(pred-3*sqrt.(sig_pred)),alpha=0.3,linewidth=0,lab="")
    display(plot!(X_test,pred,lab="",title="Iteration $iter, k: $(size(ind_points,1))"))
    # p = plot!(X_test,pred,lab="",title="Iteration $iter, k: $(size(ind_points,1))")
    # KL = KLGP.(y_train,sig_train,f,sig_f)
    # JS = JSGP.(y_train,sig_train,f,sig_f)
    # display(plot!(twinx(),X,[KL JS],lab=["KL" "JS"]))
    return p
end

function plotting2D(iter,indices,X,f,ind_points,pred_ind,x1_test,x2_test,pred,minf,maxf,title;full=false)
    N_test = size(x1_test,1)
    p = plot(x1_test,x2_test,reshape(pred,N_test,N_test),t=:contour,clim=(minf,maxf),fill=true,lab="",title="Iteration $iter")
    p = plot!(X[:,1],X[:,2],zcolor=f,t=:scatter,lab="",alpha=0.8,markerstrokewidth=0)
    # if !full
    #     p = plot!(ind_points[:,1],ind_points[:,2],zcolor=pred_ind,t=:scatter,lab="",color=:red)
    # end
    return p
end

function IntermediatePlotting(X_test,x1_test,x2_test,y_test)
    return function plotevolution(model,iter)
        y_ind = model.predict(model.kmeansalg.centers)
        y_pred,sig_pred = model.predictproba(X_test)
        y_train,sig_train = model.predictproba(X_test)
        if size(X_test,2) == 1
            display(plotting1D(iter,model.MBIndices,model.X,model.y,model.noise,model.kmeansalg.centers,y_ind,X_test,y_pred,sig_pred,y_train,sig_train,model.Name))
        else
            display(plotting2D(iter,model.MBIndices,model.X,model.y,model.kmeansalg.centers,y_ind,x1_test,x2_test,y_pred,minimum(model.y),maximum(model.y),model.Name))
        end
        sleep(0.05)
    end
end
end
