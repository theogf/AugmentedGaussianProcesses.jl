using Distributions, Random
using Plots, PyCall
using Clustering, LinearAlgebra
pyplot()
using AugmentedGaussianProcesses
const AGP = AugmentedGaussianProcesses
@pyimport sklearn.model_selection as sp

kernel = AugmentedGaussianProcesses.RBFKernel(1.0)
N_dim = 2
monotone = true
sequential = true
N_data = 1000
N_grid= 100
dpi = 150
tfontsize = 15.0
# X = generate_uniform_data(n,dim,1.0)
σ = 0.4; N_class = N_dim+1
centers = zeros(N_class,N_dim)
for i in 1:N_dim
    centers[i,i] = 1
end
centers[end,:] = (1+sqrt(N_class))/N_dim*ones(N_dim)
centers./= sqrt(N_dim)
distr = [MvNormal(centers[i,:],σ) for i in 1:N_class]
X = zeros(Float64,N_data,N_dim)
y = zeros(Int64,N_data)
true_py = zeros(Float64,N_data)
for i in 1:N_data
    y[i] = rand(1:N_class)
    X[i,:] = rand(distr[y[i]])
    true_py[i] = pdf(distr[y[i]],X[i,:])/sum(pdf(distr[k],X[i,:]) for k in 1:N_class)
end
X = X.- mean.(eachcol(X))'

xmin = minimum(X); xmax = maximum(X)
X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)
x_grid = range(xmin,length=N_grid,stop=xmax)
X_grid = hcat([j for i in x_grid, j in x_grid][:],[i for i in x_grid, j in x_grid][:])

function callbackplot(model,title="")
    y_fgrid = predict_y(model,X_grid)
    global py_fgrid = proba_y(model,X_grid)
    global cols = reshape([RGB(permutedims(Vector(py_fgrid[i,:]))[collect(values(sort(model.likelihood.ind_mapping)))]...) for i in 1:N_grid*N_grid],N_grid,N_grid)
    col_doc = [RGB(1.0,0.0,0.0),RGB(0.0,1.0,0.0),RGB(0.0,0.0,1.0)]
    global p1= Plots.plot(x_grid,x_grid,cols,t=:contour,colorbar=false,grid=:hide,framestyle=:none,yflip=false,dpi=dpi,title=title,titlefontsize=tfontsize)
    lims = (xlims(p1),ylims(p1))
    p1=Plots.plot!(p1,X[:,1],X[:,2],color=col_doc[y],t=:scatter,lab="",markerstrokewidth=0.3)
    if isa(model,OnlineVGP)
        Plots.plot!(X[model.inference.MBIndices,1],X[model.inference.MBIndices,2],color="black",t=:scatter,lab="")
        Plots.plot!(model.Z[1][:,1],model.Z[1][:,2],color="white",t=:scatter,lab="",markersize=8.0)
    end
    # p1=plot!(p1,model.Z[1][:,1],model.Z[1][:,2],color=:black,t=:scatter,lab="")
    p1= Plots.plot!(x_grid,x_grid,reshape(y_fgrid,N_grid,N_grid),clims=(0,100),t=:contour,colorbar=false,color=:gray,levels=10)
    xlims!(p1,lims[1]);ylims!(p1,lims[2])
    frame(anim)
    # display(p1)
    return p1
end


if monotone
    s = sortperm(norm.(eachrow(X)))
    X = X[s,:]; y = y[s]
else
    s = randperm(size(X,1))
    X = X[s,:]; y = y[s]
end
k = 50
b = 20

### Non sparse GP :
t_full = @elapsed fullgp = VGP(X,y,kernel,LogisticSoftMaxLikelihood(),AnalyticVI())
train!(fullgp,iterations=1)
y_full = proba_y(fullgp,X_test)
y_train = proba_y(fullgp,X)
pfull = callbackplot(fullgp,"Full batch GP")
# println("Full GP ($t_full s)\n\tRMSE (train) : $(RMSE(predict_y(fullgp,X),y))\n\tRMSE (test) : $(RMSE(y_full,y_test))")


##### DeterminantalPointProcess for selecting points

t_dpp = @elapsed dppgp = OnlineVGP(X,y,kernel,LogisticSoftMaxLikelihood(),AnalyticSVI(20),DPPAlg(0.6,kernel),sequential,verbose=3,Autotuning=false)
anim = Animation()
t_dpp = @elapsed train!(dppgp,iterations=100,callback=callbackplot)
gif(anim,"multi_online.gif",fps=10)
# t_dpp = @elapsed train!(dppgp,iterations=100)
y_dpp,sig_dpp = proba_y(dppgp,X_test)
y_inddpp = predict_y(dppgp,dppgp.Zalg.centers)
y_traindpp = proba_y(dppgp,X)

pdpp = callbackplot(dppgp,"DPP (m=$(dppgp.Zalg.k))")
# println("DPP ($t_dpp s)\n\tRMSE (train) : $(RMSE(predict_y(dppgp,X),y))\n\tRMSE (test) : $(RMSE(y_dpp,y_test))")
# kl_dpp = KLGP.(y_traindpp,sig_traindpp,y_train,sig_train)
# kl_simple = KLGP.(y_traindpp,sig_traindpp,y_train,noise)
# js_dpp = JSGP.(y_traindpp,sig_traindpp,y_train,sig_train)
# js_simple = JSGP.(y_traindpp,sig_traindpp,y_train,noise)


#### Circle K finding method with constant limit

# t_circle = @elapsed circlegp = OnlineVGP(X,y,kernel,GaussianLikelihood(noise),AnalyticSVI(24),CircleKMeans(0.6),sequential,verbose=3,Autotuning=false)
# # t_circle = @elapsed train!(circlegp,iterations=15,callback=callbackplot)
# t_circle = @elapsed train!(circlegp,iterations=100)
# y_circle,sig_circle = proba_y(circlegp,X_test)
# y_indcircle = predict_y(circlegp,circlegp.Zalg.centers)
# y_traincircle, sig_traincircle = proba_y(circlegp,X)
#
# if dim == 1
#     pcircle = plotting1D(X,y,circlegp.Zalg.centers,y_indcircle,X_test,y_circle,sig_circle,"Circle KMeans (m=$(circlegp.Zalg.k))")
# elseif dim == 2
#     pcircle = plotting2D(X,y,circlegp.Zalg.centers,y_indcircle,x1_test,x2_test,y_circle,minf,maxf,"Circle KMeans (m=$(circlegp.Zalg.k))")
# end
# println("Circle KMeans ($t_circle s)\n\tRMSE (train) : $(RMSE(predict_y(circlegp,X),y))\n\tRMSE (test) : $(RMSE(y_circle,y_test))")
# kl_circle = KLGP.(y_traincircle,sig_traincircle,y_train,sig_train)
# kl_simple = KLGP.(y_traincircle,sig_traincircle,y_train,noise)
# js_circle = JSGP.(y_traincircle,sig_traincircle,y_train,sig_train)
# js_simple = JSGP.(y_traincircle,sig_traincircle,y_train,noise)
#







# plot!(twinx(),X,[kl_const kl_simple js_const js_simple],lab=["KL" "KL_S" "JS" "JS_S"])
# plot!(twinx(),X,[kl_circle js_circle],lab=["KL" "JS"])
#plot!(X,y_trainconst+js_const,fill=(y_trainconst-js_const),alpha=0.3,lab="")

# pdiv_const = plot(X,kl_const,lab="KL")
# pdiv_const = plot!(twinx(),X,js_const,lab="JS",color=:red)
# pdiv_rand = plot(X,kl_rand,lab="KL")
# pdiv_rand = plot!(twinx(),X,js_rand,lab="JS",color=:red)


if dim == 2
    p = contourf(x1_test,x2_test,reshape(y_test,length(x1_test),length(x2_test))')
    display(plot(p,pdpp,pcircle,pfull));
else
    display(plot(pfull,pdpp,pcircle)); gui()
end
