# include("../src/OMGP.jl")

import OMGP
using Distributions
using StatsBase
# using Gallium
using Dates
using PyCall

@pyimport sklearn.datasets as sk
@pyimport sklearn.model_selection as sp
N_data = 300
N_class = 3
N_test = 50
minx=-5.0
maxx=5.0
noise = 1.0
truthknown = false
doMCCompare = false
dolikelihood = false
println("$(now()): Starting testing multiclass")

function latent(X)
    return sqrt.(X[:,1].^2+X[:,2].^2)
end
dim=2
X = (rand(N_data,dim)*(maxx-minx)).+minx
trunc_d = Truncated(Normal(0,3),minx,maxx)
X = rand(trunc_d,N_data,dim)
x_test = range(minx,stop=maxx,length=N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
# X_test = rand(trunc_d,N_test^dim,dim)
y = min.(max.(1,floor.(Int64,latent(X)+rand(Normal(0,noise),size(X,1)))),N_class)
y_test =  min.(max.(1,floor.(Int64,latent(X_test))),N_class)

X,y = sk.make_classification(n_samples=N_data,n_features=dim,n_classes=N_class,n_clusters_per_class=1,n_informative=dim,n_redundant=0)
y.+=1
X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)

# X,y = MNIST.traindata()
# X=Float64.(reshape(X,28*28,60000)')
# X_test,y_test = MNIST.testdata()
# X_test=Float64.(reshape(X_test,28*28,10000)')

#Test on the Iris dataet
# train = readdlm("data/iris-X")
# X = train[1:100,:]; X_test=train[101:end,:]
# test = readdlm("data/iris-y")
# y = test[1:100,:]; y_test=test[101:end,:]
# truthknown = false
#
# data = readdlm("data/Glass")
# # Dataset has been already randomized
# X = data[1:150,1:(end-1)]; y=data[1:150,end]
# X_test = data[151:end,1:(end-1)]; y_test=data[151:end,end]

# function norm_data(input_df)
#     for i in 1:size(input_df,2)
#         input_df[:,i] = (input_df[:,i] - mean(input_df[:,i])) ./
#             (var(input_df[:,i])==0?1:sqrt(var(input_df[:,i])))
#     end
#     return input_df
# end

#### Test on the mnist dataset
# X = readdlm("data/mnist_train")
# y=  X[:,1]; X= X[:,2:end]
# X_test = readdlm("data/mnist_test")
# y_test= X_test[:,1]; X_test=X_test[:,2:end]
# println("$(now()): MNIST data loaded")
#
# ### Test on the artificial character dataset
# X = readdlm("data/artificial-characters-train")
# y=  X[:,1]; X= X[:,2:end]
# X_test = readdlm("data/artificial-characters-test")
# y_test= X_test[:,1]; X_test=X_test[:,2:end]
# println("$(now()): Artificial Characters data loaded")



##Which algorithm are tested
fullm = false
sparsem = true
sharedInd = true
# for l in [0.001,0.005,0.01,0.05,0.1,0.5,1.0]
# for l in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
 l = 3.5

 kernel = OMGP.RBFKernel(l)
# kernel = OMGP.ARDKernel(l*ones(size(X,2)))
OMGP.setvalue!(kernel.variance,10.0)
# kernel= OMGP.PolynomialKernel([1.0,0.0,1.0])
if fullm
    fmodel = OMGP.MultiClass(X,y,VerboseLevel=3,noise=1e-3,ϵ=1e-20,kernel=kernel,Autotuning=false,AutotuningFrequency=5,IndependentGPs=true)
    metrics, callback = OMGP.getMultiClassLog(fmodel,X_test,y_test)
    # full_model.AutotuningFrequency=1
    t_full = @elapsed fmodel.train(iterations=30)#,callback=callback)

    y_full,sig_full = fmodel.predict(X_test)
    y_fall = OMGP.multiclasspredict(fmodel,X_test,true)
    y_ftrain, = fmodel.predict(X)
    println("Full predictions computed")
    full_score = 0
    for (i,pred) in enumerate(y_full)
        if pred == y_test[i]
            global full_score += 1
        end
    end
    println("Full model Accuracy is $(full_score/length(y_test)) in $t_full s for l = $l")
end
# end #End for loop on kernel lengthscale
if sparsem
    smodel = OMGP.SparseMultiClass(X,y,VerboseLevel=3,kernel=kernel,m=200,Autotuning=false,AutotuningFrequency=5,Stochastic=true,batchsize=200,IndependentGPs=true)
    # smodel.AutotuningFrequency=5
    metrics, callback = OMGP.getMultiClassLog(smodel)#,X_test,y_test)
    # smodel = OMGP.SparseMultiClass(X,y,VerboseLevel=3,kernel=kernel,m=100,Stochastic=false)
    t_sparse = @elapsed smodel.train(iterations=1000,callback=callback)
    y_sparse, = smodel.predict(X_test)
    y_strain, = smodel.predict(X)
    y_sall = OMGP.multiclasspredict(smodel,X_test,true)

    println("Sparse predictions computed")
    sparse_score=0
    if sharedInd
        #writedlm("test/results/sharedIndPoints_acc",metrics[:test_error].values)
        # writedlm("test/results/sharedIndPoints_ELBO",metrics[:ELBO].values)
    else
        # writedlm("test/results/uniqueIndPoints_acc",metrics[:test_error].values)
        # writedlm("test/results/uniqueIndPoints_ELBO",metrics[:ELBO].values)
    end
    for (i,pred) in enumerate(y_sparse)
        if pred == y_test[i]
            global sparse_score += 1
        end
    end
    println("Sparse model Accuracy is $(sparse_score/length(y_test)) in $t_sparse s")
end

if doMCCompare
    full_f_star,full_cov_f_star = OMGP.fstar(fmodel,X_test)
    logit_f = logit.(full_f_star)
    m_base = OMGP.multiclasspredict(fmodel,X_test,true)
    # m_base = OMGP.multiclasspredict(smodel,X_test)
    m_pred,sig_pred = OMGP.multiclasspredictproba(fmodel,X_test)
    m_pred_mc,sig_pred_mc = OMGP.multiclasspredictprobamcmc(fmodel,X_test,1000)
end
println("Sampling done")
function logit(x)
    return 1.0./(1.0.+exp.(-x))
end
#
if true
if size(X,2)==2
    using Plots
    pyplot()
    if truthknown
        p1=plot(x_test,x_test,reshape(y_test,N_test,N_test),t=:contour,clims=(1,N_class),fill=true,cbar=false,lab="",title="Truth")
        [plot!(X[y.==i,1],X[y.==i,2],t=:scatter,lab="y=$i",title="Training Truth") for i in 1:N_class]
        p2=plot()
        if isdefined(:y_full)
            p2=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,clims=(1,N_class),fill=true,cbar=false,lab="",title="Fullbatch")
        end
        p3=plot()
        if isdefined(:y_sparse)
            p3=plot(x_test,x_test,reshape(y_sparse,N_test,N_test),t=:contour,clims=(1,N_class),fill=true,cbar=false,lab="",title="Sparse MultiClass ($(smodel.m) points)")
            [plot!(smodel.inducingPoints[k][:,1],smodel.inducingPoints[k][:,2],t=:scatter,lab="y=$(smodel.class_mapping[k])",xlims=(-5,5),y_lims=(-5,5)) for k in 1:N_class]
        end
        display(plot(p1,p2,p3));
    else
        p1=plot()
        [plot!(X[y.==i,1],X[y.==i,2],t=:scatter,lab="y=$i",title="Training Truth") for i in 1:N_class]
        p2=plot()
        [plot!(X_test[y_test.==i,1],X_test[y_test.==i,2],t=:scatter,lab="y=$i",title="Test Truth") for i in 1:N_class]
        p3=plot()
        if fullm
            [plot!(X[y_ftrain.==i,1],X[y_ftrain.==i,2],t=:scatter,lab="y=$(fmodel.class_mapping[i])",title="Training Prediction") for i in 1:N_class]
        else
            [plot!(X[y_strain.==i,1],X[y_strain.==i,2],t=:scatter,lab="y=$(smodel.class_mapping[i])",title="Training Prediction") for i in 1:N_class]
        end
        p4=plot()
        if fullm
            [plot!(X_test[y_full.==i,1],X_test[y_full.==i,2],t=:scatter,lab="y=$(fmodel.class_mapping[i])",title="Test Prediction") for i in 1:N_class]
        else
            [plot!(X_test[y_sparse.==i,1],X_test[y_sparse.==i,2],t=:scatter,lab="y=$(smodel.class_mapping[i])",title="Test Prediction") for i in 1:N_class]
        end
        display(plot(p1,p2,p3,p4))
    end
    if doMCCompare
        p_full = [plot(x_test,x_test,reshape(full_f_star[i],N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="f_$(fmodel.class_mapping[i])") for i in 1:N_class]
        println("Latent plots ready")
        p_logit_full = [plot(x_test,x_test,reshape(logit_f[i],N_test,N_test),t=:contour,fill=true,clims=[0,1],cbar=true,lab="",title="σ(f)_$(fmodel.class_mapping[i])") for i in 1:N_class]
        println("Logit Latent plots ready")
        p_val = [plot(x_test,x_test,reshape(broadcast(x->x[i],m_pred),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="mu_approx_$(fmodel.class_mapping[i])") for i in 1:N_class]
        println("Mu plots ready")
        p_val_simple = [plot(x_test,x_test,reshape(broadcast(x->x[i],m_base),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="mu_simple_$(fmodel.class_mapping[i])") for i in 1:N_class]
        println("Mu base plots ready")
        p_val_mc = [plot(x_test,x_test,reshape(broadcast(x->x[i],m_pred_mc),N_test,N_test),t=:contour,clims=[0,1],fill=true,cbar=true,lab="",title="MC mu_$(fmodel.class_mapping[i])") for i in 1:N_class]
        println("MC Mu plots ready")
        p_sig = [plot(x_test,x_test,reshape(broadcast(x->x[i],sig_pred),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="sig_$(fmodel.class_mapping[i])") for i in 1:N_class]
        println("Sig plots ready")
        p_sig_mc = [plot(x_test,x_test,reshape(broadcast(x->x[i],sig_pred_mc),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="MC sig_$(fmodel.class_mapping[i])") for i in 1:N_class]
        println("MC sig plots ready")
        # p_sparse = [begin plot(x_test,x_test,reshape(sparse_f_star[i],N_test,N_test),t=:contour,fill=true,cbar=false,lab="",title="f_$(smodel.class_mapping[i])");plot!(smodel.inducingPoints[i][:,1],smodel.inducingPoints[i][:,2],t=:scatter,lab="") end for i in 1:N_class]
        display(plot(p_full...,p_logit_full...,p_val...,p_val_mc...,p_val_simple...,p_sig...,p_sig_mc...,layout=(7,N_class)))
    end
    if dolikelihood
        p1 = [plot()]
        if isdefined(:fmodel)
            p1 = [plot(x_test,x_test,reshape(broadcast(x->x[i],y_fall),N_test,N_test),t=:contour,fill=true,clim=(0,1),cbar=true,lab="",title="Full_likelihood_$(fmodel.class_mapping[i])") for i in 1:N_class]
        end
        p2 = [plot()]
        if isdefined(:smodel)
            p2 = [plot(x_test,x_test,reshape(broadcast(x->x[i],y_sall),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="Sparse_likelihood_$(smodel.class_mapping[i])") for i in 1:N_class]
        end
        display(plot(p1...,p2...))
    end

end
end
