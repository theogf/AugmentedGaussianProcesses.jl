using AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using BenchmarkTools
using Dates
using PyCall
using ValueHistories
using Plots
pyplot()
clibrary(:cmocean)
seed!(42)
@pyimport sklearn.datasets as sk
@pyimport sklearn.model_selection as sp
N_data = 1000
N_class = 3
N_test = 50
N_grid = 50
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
N_dim=2
# X = (rand(N_data,N_dim)*(maxx-minx)).+minx
# trunc_d = Truncated(Normal(0,3),minx,maxx)
# X = rand(trunc_d,N_data,N_dim)
# x_test = range(minx,stop=maxx,length=N_test)
# X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
# # X_test = rand(trunc_d,N_test^dim,dim)
# y = min.(max.(1,floor.(Int64,latent(X)+rand(Normal(0,noise),size(X,1)))),N_class)
# y_test =  min.(max.(1,floor.(Int64,latent(X_test))),N_class)

# X,y = sk.make_classification(n_samples=N_data,n_features=N_dim,n_classes=N_class,n_clusters_per_class=1,n_informative=N_dim,n_redundant=0)
# y.+=1
# X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)

art_noise = 0.1

X_clean = (rand(N_data,N_dim)*2.0).-1.0
y = zeros(Int64,N_data); y_noise = similar(y)
function classify(X,y)
    for i in 1:size(X,1)
        if X[i,2] < min(0,-X[i,1])
            y[i] = 1
        elseif X[i,2] > max(0,X[i,1])
            y[i] = 2
        else
            y[i] = 3
        end
    end
    return y
end
X= X_clean+rand(Normal(0,art_noise),N_data,N_dim)
classify(X_clean,y);classify(X,y_noise);
bayes_error = count(y.!=y_noise)/length(y)

xmin = minimum(X); xmax = maximum(X)
X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)
x_grid = range(xmin,length=N_grid,stop=xmax)
X_grid = hcat([j for i in x_grid, j in x_grid][:],[i for i in x_grid, j in x_grid][:])

x₁slice = 0.5
X_slice = hcat(fill(x₁slice,N_grid),x_grid)


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




# for l in [0.001,0.005,0.01,0.05,0.1,0.5,1.0]
# for l in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
function initial_lengthscale(X)
    D = pairwise(SqEuclidean(),X')
    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end
function acc(y_test,y_pred)
    count(y_test.==y_pred)/length(y_pred)
end
function loglike(y_test,y_pred)
    ll = 0.0
    for i in 1:length(y_test)
        ll += log(y_pred[Symbol(y_test[i])][i])
    end
    ll /= length(y_test)
    return ll
end
 l = sqrt(initial_lengthscale(X))

kernel = AugmentedGaussianProcesses.RBFKernel([l],dim=N_dim)
# kernel = AugmentedGaussianProcesses.RBFKernel(l)
AugmentedGaussianProcesses.setvalue!(kernel.fields.variance,1.0)
# kernel= AugmentedGaussianProcesses.PolynomialKernel([1.0,0.0,1.0])
metrics = MVHistory()
elbos = MVHistory()
ρparams = MVHistory()
lparams = MVHistory()
vparams = MVHistory()
anim  = Animation()
function callback(model::GP{TLike,TInf},iter) where {TLike,TInf}
    push!(elbos,:loglike,AugmentedGaussianProcesses.expecLogLikelihood(model))
    push!(elbos,:gaussian,-AugmentedGaussianProcesses.GaussianKL(model))
    if AugmentedGaussianProcesses.isaugmented(TLike())
        push!(elbos,:gamma,-AugmentedGaussianProcesses.GammaImproperKL(model))
        push!(elbos,:poisson,-AugmentedGaussianProcesses.PoissonKL(model))
        push!(elbos,:polyagamma,-AugmentedGaussianProcesses.PolyaGammaKL(model))
    end
    push!(elbos,:ELBO,AugmentedGaussianProcesses.ELBO(model))
    for i in 1:model.nPrior
        push!(vparams,Symbol(:k,i),getvariance(model.kernel[i]))
        p = getlengthscales(model.kernel[i])
        for (j,p_j) in enumerate(p)
            push!(lparams,Symbol("k",i,"_j",j),p_j)
        end
    end
    if isa(model.inference.optimizer_η₁[1],ALRSVI)
        for i in 1:model.nLatent
            push!(ρparams,Symbol("opt_η₁_",i),model.inference.optimizer_η₁[i].ρ)
            push!(ρparams,Symbol("opt_η₂_",i),model.inference.optimizer_η₂[i].ρ)
        end
    elseif isa(model.inference.optimizer_η₁[1],VanillaGradDescent)
        for i in 1:model.nLatent
                push!(ρparams,Symbol("opt_η₁_",i),model.inference.optimizer_η₁[i].η)
                push!(ρparams,Symbol("opt_η₂_",i),model.inference.optimizer_η₂[i].η)
        end
    end
    y_pred_test = predict_y(model,X_test)
    y_pred_train = predict_y(model,X)
    py_pred_test = proba_y(model,X_test)
    py_pred_train = proba_y(model,X)
    push!(metrics,:err_train,1-acc(y,y_pred_train))
    push!(metrics,:err_test,1-acc(y_test,y_pred_test))
    push!(metrics,:nll_train,-loglike(y,py_pred_train))
    push!(metrics,:nll_test,-loglike(y_test,py_pred_test))
end

function callbackplot(model,iter)
    y_fgrid =  predict_y(model,X_grid)
    global py_fgrid = proba_y(model,X_grid)
    global μ_fslice,σ_fslice = predict_f(model,X_slice,covf=true)
    remap = collect(values(sort(model.likelihood.ind_mapping)))
    σ_fslice = [sqrt.(max.(0,σ)) for σ in σ_fslice[remap]]
    μ_fslice = μ_fslice[remap]
    global cols = reshape([RGB(permutedims(Vector(py_fgrid[i,:]))[remap]...) for i in 1:N_grid*N_grid],N_grid,N_grid)
    col_doc = [RGB(1.0,0.0,0.0),RGB(0.0,1.0,0.0),RGB(0.0,0.0,1.0)]
    col_name = [:red,:green,:blue]
    global p1= plot(x_grid,x_grid,cols,t=:contour,colorbar=false,framestyle=:box)
    lims = (xlims(p1),ylims(p1))
    p1=plot!(p1,X[:,1],X[:,2],color=col_doc[y],t=:scatter,lab="",markerstrokewidth=0.2)
    if isa(model,SVGP)
     p1=plot!(p1,model.Z[1][:,1],model.Z[1][:,2],color=:black,t=:scatter,lab="")
    end
    p1= plot!(x_grid,x_grid,reshape(y_fgrid,N_grid,N_grid),clims=(0,100),t=:contour,colorbar=false,color=:gray,levels=10)
    xlims!(p1,lims[1]);ylims!(p1,lims[2])
    plot!(p1,[x₁slice,x₁slice],[xmin xmax],color=:black,alpha=1.0,lab="")
    p2=plot()
    for k in 1:model.nLatent
        p2 = plot!(p2,x_grid,μ_fslice[k],color=col_name[k],lab="y=$k")
        p2 = plot!(x_grid,μ_fslice[k].+2*σ_fslice[k],fill=(μ_fslice[k].-2*σ_fslice[k],0.2,col_name[k]),linewidth=0.0,lab="",color=col_name[k])
    end
    frame(anim,p1)
    display(plot(p1,p2))
    return plot(p1,p2)
end

##Which algorithm are tested
fullm = !true
sparsem = !true
stochm = !true
expecm = true

if fullm
    global fmodel = VGP(X,y,kernel,AugmentedLogisticSoftMaxLikelihood(),AnalyticInference(),verbose=3,Autotuning=!true,atfrequency=1,IndependentPriors=true)
    t_full = @elapsed train!(fmodel,iterations=100,callback=callback)

    global y_full = predict_y(fmodel,X_test)
    global y_fall = proba_y(fmodel,X_test)
    global y_ftrain = predict_y(fmodel,X)
    global y_fgrid = predict_y(fmodel,X_grid)
    println("Full predictions computed")
    full_score = 0
    for (i,pred) in enumerate(y_full)
        if pred == y_test[i]
            global full_score += 1
        end
    end
    println("Full model Accuracy is $(full_score/length(y_test)) in $t_full s for l = $l")
    callbackplot(fmodel,1)
end


# end #End for loop on kernel lengthscale
if sparsem
    global smodel = SVGP(X,y,kernel,AugmentedLogisticSoftMaxLikelihood(),AnalyticInference(),10,verbose=0,Autotuning=true,atfrequency=1,IndependentPriors=true)
    # smodel.AutotuningFrequency=5
    # smetrics, callback = AugmentedGaussianProcesses.getMultiClassLog(smodel,X_test=X_test,y_test=y_test)
    # smodel = AugmentedGaussianProcesses.SparseMultiClass(X,y,verbose=3,kernel=kernel,m=100,Stochastic=false)
    @time train!(smodel,iterations=100)#,callback=callback)
    # t_sparse = @elapsed smodel.train(iterations=100,callback=callback)
    global y_sparse = predict_y(smodel,X_test)
    global y_strain = predict_y(smodel,X)
    global y_sall = proba_y(smodel,X_test)

    println("Sparse predictions computed")
    sparse_score=0
    for (i,pred) in enumerate(y_sparse)
        if pred == y_test[i]
            global sparse_score += 1
        end
    end
    println("Sparse model Accuracy is $(sparse_score/length(y_test))")#" in $t_sparse s")
    # callbackplot(smodel,1)
end
# @profiler train!(smodel,iterations=100)
# @btime train!($smodel,iterations=10);
using GradDescent
if stochm
    global ssmodel = SVGP(X,y,kernel,AugmentedLogisticSoftMaxLikelihood(),StochasticAnalyticInference(100,optimizer=ALRSVI()),10,verbose=3,Autotuning=true,atfrequency=1,IndependentPriors=!true)
    # global ssmodel = SVGP(X,y,kernel,AugmentedLogisticSoftMaxLikelihood(),StochasticAnalyticInference(10,optimizer=ALRSVI(τ=200)),10,verbose=3,Autotuning=true,atfrequency=1,IndependentPriors=true)
    @time train!(ssmodel,iterations=50,callback=callback)
    global y_ssparse = predict_y(ssmodel,X_test)
    global y_sstrain = predict_y(ssmodel,X)
    global y_ssall = proba_y(ssmodel,X_test)

    println("Stochastic Sparse predictions computed")
    ssparse_score=0
    for (i,pred) in enumerate(y_ssparse)
        if pred == y_test[i]
            global ssparse_score += 1
        end
    end
    println("Sparse model Accuracy is $(ssparse_score/length(y_test))")#" in $t_sparse s")
    callbackplot(ssmodel,1)
end

if expecm
    global emodel = SVGP(X,y,kernel,LogisticSoftMaxLikelihood(),NumericalInference(:mcmc,optimizer=VanillaGradDescent(η=0.01)),10,verbose=3,Autotuning=true,atfrequency=1,IndependentPriors=true)
    # global ssmodel = SVGP(X,y,kernel,AugmentedLogisticSoftMaxLikelihood(),StochasticAnalyticInference(10,optimizer=ALRSVI(τ=200)),10,verbose=3,Autotuning=true,atfrequency=1,IndependentPriors=true)
    @time train!(emodel,iterations=200,callback=callback)
    global y_expec = predict_y(emodel,X_test)
    global y_expectrain = predict_y(emodel,X)
    global y_expecall = proba_y(emodel,X_test)

    println("Stochastic Sparse predictions computed")
    expec_score=0
    for (i,pred) in enumerate(y_expec)
        if pred == y_test[i]
            global expec_score += 1
        end
    end
    println("Expected Gradient model Accuracy is $(expec_score/length(y_test))")#" in $t_sparse s")
    callbackplot(emodel,1)
end


display(plot(plot(elbos,title="ELBO values"),plot(ρparams,title="learning rate",yaxis=:log)))
display(plot(plot(lparams,title="lengthscale",yaxis=:log),plot(vparams,title="variance",yaxis=:log)))
display(plot(metrics,title="Metrics"))
1
    # model = ssmodel;
    # AugmentedGaussianProcesses.computeMatrices!(model)
    # AugmentedGaussianProcesses.update_hyperparameters!(model)
    # AugmentedGaussianProcesses.computeMatrices!(model)
    # train!(model;iterations=1)
    # @btime AugmentedGaussianProcesses.update_hyperparameters!(model)
    # Profile.clear()
    # AugmentedGaussianProcesses.computeMatrices!(model)
    # @profile AugmentedGaussianProcesses.update_hyperparameters!(model)
    # @profile AugmentedGaussianProcesses.train!(model;iterations=100)
    # ProfileView.view()

1

# t_ssparse = @elapsed ssmodel.train(iterations=200,callback=callback)

# dim_t = 784
# tkernel = AugmentedGaussianProcesses.ARDKernel([l],dim=dim_t)
# Test = rand(4000,dim_t)
# @elapsed begin
#     A = [Matrix(undef,200,200) for _ in 1:(dim_t+1)];
#     for i in 1:200
#         for j in 1:i
#             global g = AugmentedGaussianProcesses.KernelFunctions.compute_deriv(tkernel,Test[i,:],Test[j,:],true);
#             [a[i,j] = g[iter] for (iter,a) in enumerate(A)]
#         end
#     end
# end
#
# @elapsed AugmentedGaussianProcesses.KernelFunctions.derivativekernelmatrix(tkernel,Test[1:200,:])

function logit(x)
    return 1.0./(1.0.+exp.(-x))
end
#
# callbacktests = false
# if callbacktests
#     plot(fmetrics[:test_error])
#     plot!(sfmetrics[:test_error])
#     plot!(smetrics[:test_error])
#     plot!(ssmetrics[:test_error])
# end
#
# #
# if false
# if size(X,2)==2
#     using Plots
#     pyplot()
#     if truthknown
#         p1=plot(x_test,x_test,reshape(y_test,N_test,N_test),t=:contour,clims=(1,N_class),fill=true,cbar=false,lab="",title="Truth")
#         [plot!(X[y.==i,1],X[y.==i,2],t=:scatter,lab="y=$i",title="Training Truth") for i in 1:N_class]
#         p2=plot()
#         if isdefined(:y_full)
#             p2=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,clims=(1,N_class),fill=true,cbar=false,lab="",title="Fullbatch")
#         end
#         p3=plot()
#         if isdefined(:y_sparse)
#             p3=plot(x_test,x_test,reshape(y_sparse,N_test,N_test),t=:contour,clims=(1,N_class),fill=true,cbar=false,lab="",title="Sparse MultiClass ($(smodel.m) points)")
#             [plot!(smodel.inducingPoints[k][:,1],smodel.inducingPoints[k][:,2],t=:scatter,lab="y=$(smodel.class_mapping[k])",xlims=(-5,5),y_lims=(-5,5)) for k in 1:N_class]
#         end
#         display(plot(p1,p2,p3));
#     else
#         p1=plot()
#         [plot!(X[y.==i,1],X[y.==i,2],t=:scatter,lab="y=$i",title="Training Truth") for i in 1:N_class]
#         p2=plot()
#         [plot!(X_test[y_test.==i,1],X_test[y_test.==i,2],t=:scatter,lab="y=$i",title="Test Truth") for i in 1:N_class]
#         p3=plot()
#         if fullm
#             [plot!(X[y_ftrain.==i,1],X[y_ftrain.==i,2],t=:scatter,lab="y=$(fmodel.class_mapping[i])",title="Training Prediction") for i in 1:N_class]
#         else
#             [plot!(X[y_strain.==i,1],X[y_strain.==i,2],t=:scatter,lab="y=$(smodel.class_mapping[i])",title="Training Prediction") for i in 1:N_class]
#         end
#         p4=plot()
#         if fullm
#             [plot!(X_test[y_full.==i,1],X_test[y_full.==i,2],t=:scatter,lab="y=$(fmodel.class_mapping[i])",title="Test Prediction") for i in 1:N_class]
#         else
#             [plot!(X_test[y_sparse.==i,1],X_test[y_sparse.==i,2],t=:scatter,lab="y=$(smodel.class_mapping[i])",title="Test Prediction") for i in 1:N_class]
#         end
#         display(plot(p1,p2,p3,p4))
#     end
#     if doMCCompare
#         p_full = [plot(x_test,x_test,reshape(full_f_star[i],N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="f_$(fmodel.class_mapping[i])") for i in 1:N_class]
#         println("Latent plots ready")
#         p_logit_full = [plot(x_test,x_test,reshape(logit_f[i],N_test,N_test),t=:contour,fill=true,clims=[0,1],cbar=true,lab="",title="σ(f)_$(fmodel.class_mapping[i])") for i in 1:N_class]
#         println("Logit Latent plots ready")
#         p_val = [plot(x_test,x_test,reshape(broadcast(x->x[i],m_pred),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="mu_approx_$(fmodel.class_mapping[i])") for i in 1:N_class]
#         println("Mu plots ready")
#         p_val_simple = [plot(x_test,x_test,reshape(broadcast(x->x[i],m_base),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="mu_simple_$(fmodel.class_mapping[i])") for i in 1:N_class]
#         println("Mu base plots ready")
#         p_val_mc = [plot(x_test,x_test,reshape(broadcast(x->x[i],m_pred_mc),N_test,N_test),t=:contour,clims=[0,1],fill=true,cbar=true,lab="",title="MC mu_$(fmodel.class_mapping[i])") for i in 1:N_class]
#         println("MC Mu plots ready")
#         p_sig = [plot(x_test,x_test,reshape(broadcast(x->x[i],sig_pred),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="sig_$(fmodel.class_mapping[i])") for i in 1:N_class]
#         println("Sig plots ready")
#         p_sig_mc = [plot(x_test,x_test,reshape(broadcast(x->x[i],sig_pred_mc),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="MC sig_$(fmodel.class_mapping[i])") for i in 1:N_class]
#         println("MC sig plots ready")
#         # p_sparse = [begin plot(x_test,x_test,reshape(sparse_f_star[i],N_test,N_test),t=:contour,fill=true,cbar=false,lab="",title="f_$(smodel.class_mapping[i])");plot!(smodel.inducingPoints[i][:,1],smodel.inducingPoints[i][:,2],t=:scatter,lab="") end for i in 1:N_class]
#         display(plot(p_full...,p_logit_full...,p_val...,p_val_mc...,p_val_simple...,p_sig...,p_sig_mc...,layout=(7,N_class)))
#     end
#     if dolikelihood
#         p1 = [plot()]
#         if isdefined(:fmodel)
#             p1 = [plot(x_test,x_test,reshape(broadcast(x->x[i],y_fall),N_test,N_test),t=:contour,fill=true,clim=(0,1),cbar=true,lab="",title="Full_likelihood_$(fmodel.class_mapping[i])") for i in 1:N_class]
#         end
#         p2 = [plot()]
#         if isdefined(:smodel)
#             p2 = [plot(x_test,x_test,reshape(broadcast(x->x[i],y_sall),N_test,N_test),t=:contour,fill=true,cbar=true,lab="",title="Sparse_likelihood_$(smodel.class_mapping[i])") for i in 1:N_class]
#         end
#         display(plot(p1...,p2...))
#     end
#
# end
# end

# using LinearAlgebra, ForwardDiff
# k = Array(exp(Symmetric(rand(10,10))))
# a = cholesky(Array(exp(Symmetric(rand(10,10))))).L
# isposdef(k)
# # a = rand(10)
# X = rand(10,4)
# Y = rand(20,4)
# function foo(s)
#     kernel = RBFKernel(s[1])
#     A = kernelmatrix(X,kernel)+1e-7I
#     B = kernelmatrix(Y,X,kernel)
#     return B*inv(A)
# end
#
# function foo_tilde(sk)
#     return sum(sin.(sk*sk))
# end
# a= 3.0
# foo(3.0)
# g = ForwardDiff.jacobian(foo,[a])
# g̃ = ForwardDiff.gradient(foo_tilde,a*k)
# function grad_foo(a,g̃)
#     return tr(g̃'*k)
# end
#
# grad_foo(a,g̃)
