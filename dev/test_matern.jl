using Distributions
using AugmentedGaussianProcesses
using LinearAlgebra
using Random: seed!
using GradDescent
const AGP = AugmentedGaussianProcesses
seed!(42)
doPlots=  true
verbose = 3
using Plots
pyplot()
N_data = 100
N_test = 1000
N_dim = 1
noise = 1e-16
minx=-1.0
maxx=1.0
μ_0 = -1
l= sqrt(0.1); vf = 2.0;vg=1.0; α = 0.1;n_sig = 2
k = RBFKernel(0.3,variance = vf); m = 40
autotuning = false

rmse(y,y_test) = norm(y-y_test,2)/sqrt(length(y_test))
# h(x) = exp.(-x)
X = rand(N_data,N_dim)*(maxx-minx).+minx
x_test = range(minx,stop=maxx,length=N_test)
X_tot = vcat(X,x_test)
y_m = rand(MvNormal(zeros(N_data+N_test),kernelmatrix(X_tot,k)+1e-5*I))
y = y_m .+ rand(TDist(5.0),N_data+N_test)
scatter(X_tot,y_m,lab="True Mean")
scatter!(X_tot,y,lab="Data") |> display
X_test = collect(x_test)
y_test = y[N_data+1:end]
y = y[1:N_data]
s = sortperm(vec(X))
miny = minimum(y); maxy = maximum(y)
miny = miny < 0 ? 1.5*miny : 0.5*miny
maxy = maxy > 0 ? 1.5*maxy : 0.5*maxy
# X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
ps = []; t_full = 0; t_sparse = 0; t_stoch = 0;

fullm = true
println("Testing the Heteroscedastic model")
function callback(model,iter)
    if iter%50 == 0
        y_full_num,sig_full_num = proba_y(model,X_test);
        p2 = plot(X_test,y_full_num+n_sig*sqrt.(sig_full_num),fillrange=(y_full_num-n_sig*sqrt.(sig_full_num)),alpha=0.3,lab="Matern 3/2 Num")
        plot!(x_test,y_full_num,lab="",title="Matern 3/2 Num",ylim=(miny,maxy)) |> display
    end
end

# if fullm
println("Testing the full model")
t_full = @elapsed global model = VGP(X,y,k,Matern3_2Likelihood(0.1),AnalyticVI(),verbose=verbose,optimizer=false)
# model.μ = copy(y_m[1:model.nSamples])
t_full += @elapsed train!(model,iterations=10)#,callback=callback)
global y_full,sig_full = proba_y(model,X_test); rmse_full = rmse(y_full,y_test);

if doPlots
    # p1=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="",title="StudentT")
    p1 = plot(X_test,y_full+n_sig*sqrt.(sig_full),fillrange=(y_full-n_sig*sqrt.(sig_full)),alpha=0.3,lab="Matern 3/2")
    plot!(x_test,y_full,lab="",title="Matern 3/2",ylim=(miny,maxy)) |> display
    # plot!(twinx(),X_test,y_fullg,lab="Latent g")

    push!(ps,p1)
end

##
t_num = @elapsed global model_num = SVGP(X,y,k,Matern3_2Likelihood(0.1),QuadratureVI(optimizer=Momentum(η=1e-5)),50,verbose=verbose,optimizer=false)
# model.μ = copy(y_m[1:model.nSamples])
t_num += @elapsed train!(model_num,iterations=450,callback=callback)
global y_full_num,sig_full_num = proba_y(model_num,X_test); rmse_full = rmse(y_full_num,y_test);

if doPlots
    # p1=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="",title="StudentT")
    p2 = plot(X_test,y_full_num+n_sig*sqrt.(sig_full_num),fillrange=(y_full_num-n_sig*sqrt.(sig_full_num)),alpha=0.3,lab="Matern 3/2 Num")
    plot!(x_test,y_full_num,lab="",title="Matern 3/2 Num",ylim=(miny,maxy)) |> display
    # plot!(twinx(),X_test,y_fullg,lab="Latent g")

    push!(ps,p2)
end

##


t_full != 0 ? println("Heteroscedastic model : RMSE=$(rmse_full), time=$t_full") : nothing
@isdefined(t_gp) ? println("Homoscedastic model : RMSE=$(rmse_gp),time=$t_gp") : nothing

if doPlots
    y_nonoise = y_m[N_data+1:end]
    ptrue=plot(X,y,t=:scatter,lab="Training points",alpha=0.3,markerstrokewidth=0.0,color=:black)
    # plot!(ptrue,x_test,y_test,t=:scatter,lab="Test points",title="Truth")
    plot!(ptrue,x_test,y_nonoise,ylim=(miny,maxy),lab="Noiseless")
    # plot!(ptrue,x_test,y_nonoise+n_sig*noise_ytest,fill=y_nonoise-n_sig*noise_ytest,alpha=0.5,lab="")
    # plot!(twinx(),x_test,y_gnoise,lab="")
    # plot!(twinx(),x_test,noise_ytest,lab="")
    # ptrue=plot(x_test,x_test,reshape(latent(X_test),N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="")
    # plot!(ptrue,X[:,1],X[:,2],t=:scatter,lab="training points",title="Truth")
    display(plot(ptrue,ps...))
end
##
