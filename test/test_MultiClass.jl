include("../src/OMGP.jl")
import OMGP
using Distributions
using StatsBase
using Gallium
N_data = 1000
N_class = 3
N_test = 50
minx=-5.0
maxx=5.0
noise = 0.1


function latent(X)
    return sqrt.(X[:,1].^2+X[:,2].^2)
end

X = (rand(N_data,2)*(maxx-minx))+minx
trunc_d = Truncated(Normal(0,3),minx,maxx)
X = rand(trunc_d,N_data,2)
x_test = linspace(minx,maxx,N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
y = min.(max.(1,floor.(Int64,latent(X)+rand(Normal(0,noise),size(X,1)))),N_class)
y_test =  min.(max.(1,floor.(Int64,latent(X_test))),N_class)

# data = readdlm("data/Iris")
#Dataset has been already randomized
# X = data[1:100,1:(end-1)]; y=data[1:100,end]
# X_test = data[101:end,1:(end-1)]; y_test=data[101:end,end]

#
# data = readdlm("data/Glass")
# # Dataset has been already randomized
# X = data[1:150,1:(end-1)]; y=data[1:150,end]
# X_test = data[151:end,1:(end-1)]; y_test=data[151:end,end]

#### Test on the mnist dataset

# X = readdlm("data/mnist_train")
# y=  X[:,1]; X= X[:,2:end]
# X_test = readdlm("data/mnist_test")
# y_test= X_test[:,1]; X_test=X_test[:,2:end]
# println("MNIST data loaded")


kernel = OMGP.Matern5_2Kernel(1.0)
# OMGP.setvalue!(kernel.weight,10.0)
# kernel= OMGP.PolynomialKernel([1.0,0.0,1.0])
# full_model = OMGP.MultiClass(X,y,VerboseLevel=3,kernel=kernel)
sparse_model = OMGP.SparseMultiClass(X,y,VerboseLevel=3,kernel=kernel,m=50,Stochastic=true,BatchSize=150)
# t_full = @elapsed full_model.train(iterations=200)
t_sparse = @elapsed sparse_model.train(iterations=200)
# y_full, = full_model.predict(X_test)
# m_base = OMGP.multiclasspredict(full_model,X_test,true)
m_base = OMGP.multiclasspredict(sparse_model,X_test)
# m_pred,sig_pred = OMGP.multiclasspredictproba(full_model,X_test)
println("Full predictions computed")
y_sparse, = sparse_model.predict(X_test)
println("Sparse predictions computed")

# m_pred_mc,sig_pred_mc = OMGP.multiclasspredictprobamcmc(full_model,X_test,1000)
# println("Sampling done")
# full_score = 0
# for (i,pred) in enumerate(y_full)
#     if pred == y_test[i]
#         full_score += 1
#     end
# end
sparse_score=0
for (i,pred) in enumerate(y_sparse)
    if pred == y_test[i]
        sparse_score += 1
    end
end
# println("Full model Accuracy is $(full_score/length(y_test)) in $t_full s")
println("Sparse model Accuracy is $(sparse_score/length(y_test)) in $t_sparse s")
# full_f_star,full_cov_f_star = OMGP.fstar(full_model,X_test)
function logit(x)
    return 1./(1+exp.(-x))
end
# logit_f = logit.(full_f_star)


#
# if false
if size(X,2)==2
    using Plots
    # pyplot()
    plotlyjs()
    # p1=plot(x_test,x_test,reshape(y_test,N_test,N_test),t=:contour,clims=(1,N_class),cbar=false,fill=:true)
    # [plot!(X[y.==i,1],X[y.==i,2],t=:scatter,lab="y=$i",title="Truth",xlims=(-5,5),ylims=(-5,5)) for i in 1:N_class]
    # p2=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,clims=(1,N_class),fill=true,cbar=false,lab="",title="MultiClass")
    # p3=plot(x_test,x_test,reshape(y_sparse,N_test,N_test),t=:contour,clims=(1,N_class),fill=true,cbar=false,lab="",title="Sparse MultiClass ($(sparse_model.m) points)")
    # [plot!(sparse_model.inducingPoints[k][:,1],sparse_model.inducingPoints[k][:,2],t=:scatter,lab="y=$(sparse_model.class_mapping[k])",xlims=(-5,5),y_lims=(-5,5)) for k in 1:N_class]
    # display(plot(p1,p2,p3));
    #
    # sparse_f_star = OMGP.fstar(sparse_model,X_test,covf=false)
    p_full = [plot(x_test,x_test,reshape(full_f_star[i],N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="f_$(full_model.class_mapping[i])") for i in 1:N_class]
    println("Latent plots ready")
    p_logit_full = [plot(x_test,x_test,reshape(logit_f[i],N_test,N_test),t=:contour,fill=true,clims=[0,1],cbar=true,lab="",title="σ(f)_$(full_model.class_mapping[i])") for i in 1:N_class]
    println("Logit Latent plots ready")
    p_val = [plot(x_test,x_test,reshape(broadcast(x->x[i],m_pred),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="mu_approx_$(full_model.class_mapping[i])") for i in 1:N_class]
    println("Mu plots ready")
    p_val_simple = [plot(x_test,x_test,reshape(broadcast(x->x[i],m_base),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="mu_simple_$(full_model.class_mapping[i])") for i in 1:N_class]
    println("Mu base plots ready")
    p_val_mc = [plot(x_test,x_test,reshape(broadcast(x->x[i],m_pred_mc),N_test,N_test),t=:contour,clims=[0,1],fill=true,cbar=true,lab="",title="MC mu_$(full_model.class_mapping[i])") for i in 1:N_class]
    println("MC Mu plots ready")
    p_sig = [plot(x_test,x_test,reshape(broadcast(x->x[i],sig_pred),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="sig_$(full_model.class_mapping[i])") for i in 1:N_class]
    println("Sig plots ready")
    p_sig_mc = [plot(x_test,x_test,reshape(broadcast(x->x[i],sig_pred_mc),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="MC sig_$(full_model.class_mapping[i])") for i in 1:N_class]
    println("MC sig plots ready")
    # p_sparse = [begin plot(x_test,x_test,reshape(sparse_f_star[i],N_test,N_test),t=:contour,fill=true,cbar=false,lab="",title="f_$(sparse_model.class_mapping[i])");plot!(sparse_model.inducingPoints[i][:,1],sparse_model.inducingPoints[i][:,2],t=:scatter,lab="") end for i in 1:N_class]
    display(plot(p_full...,p_logit_full...,p_val...,p_val_mc...,p_val_simple...,p_sig...,p_sig_mc...,layout=(7,N_class)))
end
