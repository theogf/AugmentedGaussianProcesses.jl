using Distributions
using AugmentedGaussianProcesses
using LinearAlgebra
using Random: seed!
using ValueHistories
const AGP = AugmentedGaussianProcesses
seed!(42)
doPlots=  true
verbose = 2
using Plots
pyplot()
N_data = 1000
N_test = 1000
N_dim = 1
noise = 1e-16
minx=-1.0
maxx=1.0
μ_0 = -1
l= sqrt(0.1); vf = 2.0;vg=1.0; α = 0.1;n_sig = 2
kernel = RBFKernel(0.3,variance = vf); m = 40
kernel_g = RBFKernel(0.3,variance = vg)
autotuning = false

rmse(y,y_test) = norm(y-y_test,2)/sqrt(length(y_test))
logistic(x) = 1.0./(1.0.+exp.(-x))
h(x) = α*logistic.(x)
# h(x) = exp.(-x)
X = rand(N_data,N_dim)*(maxx-minx).+minx
x_test = range(minx,stop=maxx,length=N_test)
X_tot = vcat(X,x_test)
y_m = rand(MvNormal(zeros(N_data+N_test),kernelmatrix(X_tot,kernel)+1e-5*I))
y_noise = rand(MvNormal(μ_0*ones(N_data+N_test),kernelmatrix(X_tot,kernel_g)+1e-5I))
h_noise = 1.0./sqrt.(h.(y_noise))
y = y_m .+ rand.(Normal.(0,h_noise))
scatter(X_tot,y_m,lab="True Mean")
scatter!(X_tot,y_noise,lab="True latent noise")
display(scatter!(X_tot,y,lab="Data"))
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
sparsem =false
stochm = false
println("Testing the Heteroscedastic model")
global metrics = MVHistory()
function callback(model,iter)
    # push!(metrics,:Poisson,iter,AugmentedGaussianProcesses.PoissonKL(model))
    # push!(metrics,:PolyaGamma,iter,AugmentedGaussianProcesses.PolyaGammaKL(model))
    # push!(metrics,:Gaussian_G,iter,AugmentedGaussianProcesses.GaussianGKL(model))
    # push!(metrics,:Gaussian_F,iter,AugmentedGaussianProcesses.GaussianKL(model))
    # push!(metrics,:Loglike,iter,-AugmentedGaussianProcesses.ExpecLogLikelihood(model))
    # push!(metrics,:ELBO,iter,AugmentedGaussianProcesses.ELBO(model))
    pg = plot(X[s],y_noise[1:N_data][s],lab="true g")
    plot!(pg,X[s],model.likelihood.μ[1][s],lab="μ_g")
    # pl = plot(X[s],model.likelihood.λ[1][s],lab="λ")
    pθ = plot(X[s],model.likelihood.θ[1][s],lab="θ")
    pγ = plot(X[s],model.likelihood.γ[1][s],lab="γ")
    pc = plot(X[s],model.likelihood.c[1][s],lab="c")
    psig = plot(X[s],diag(model.likelihood.Σ[1])[s],lab="Σ")
    # model.μ = copy(y_m[1:model.nSamples])
    display(plot(pg,psig,pθ,pγ,pc))
    sleep(0.1)
end

# if fullm
println("Testing the full model")
t_full = @elapsed global model = VGP(X,y,kernel,AGP.HeteroscedasticLikelihood(kernel_g,AGP.ConstantMean(float(μ_0))),AnalyticVI(),verbose=verbose,optimizer=false)
# model.μ = copy(y_m[1:model.nSamples])
t_full += @elapsed train!(model,iterations=10)#,callback=callback)
global y_full,sig_full = proba_y(model,X_test); rmse_full = rmse(y_full,y_test);

if doPlots
    # p1=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="",title="StudentT")
    p1=plot(x_test,y_full,lab="",title="Heteroscedastic",ylim=(miny,maxy))
    plot!(p1,X_test,y_full+n_sig*sqrt.(sig_full),fill=(y_full-n_sig*sqrt.(sig_full)),alpha=0.3,lab="Heteroscedastic GP")
    # plot!(twinx(),X_test,y_fullg,lab="Latent g")

    push!(ps,p1)
end
# end
##


println("Basic GP")
gpmodel = GP(X,y,kernel,noise=1.0)
t_gp = @elapsed train!(gpmodel,iterations=10)
y_gp,sig_gp = proba_y(gpmodel,X_test); rmse_gp = rmse(y_gp,y_test)
if doPlots
    p4=plot(x_test,y_gp,lab="",title="Homoscedastic",ylim=(miny,maxy))
    plot!(p4,X_test,y_gp+n_sig*sqrt.(sig_gp),fill=(y_gp-n_sig*sqrt.(sig_gp)),alpha=0.3,lab="Homoscedastic GP")
    push!(ps,p4)
end
##

t_full != 0 ? println("Heteroscedastic model : RMSE=$(rmse_full), time=$t_full") : nothing
t_gp != 0 ? println("Homoscedastic model : RMSE=$(rmse_gp),time=$t_gp") : nothing

if doPlots
    y_nonoise = y_m[N_data+1:end]
    y_gnoise = y_noise[N_data+1:end]
    noise_ytest = h_noise[N_data+1:end]
    ptrue=plot(X,y,t=:scatter,lab="Training points",alpha=0.3,markerstrokewidth=0.0,color=:black)
    # plot!(ptrue,x_test,y_test,t=:scatter,lab="Test points",title="Truth")
    plot!(ptrue,x_test,y_nonoise,ylim=(miny,maxy),lab="Noiseless")
    plot!(ptrue,x_test,y_nonoise+n_sig*noise_ytest,fill=y_nonoise-n_sig*noise_ytest,alpha=0.5,lab="")
    plot!(twinx(),x_test,y_gnoise,lab="")
    # plot!(twinx(),x_test,noise_ytest,lab="")
    # ptrue=plot(x_test,x_test,reshape(latent(X_test),N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="")
    # plot!(ptrue,X[:,1],X[:,2],t=:scatter,lab="training points",title="Truth")
    display(plot(ptrue,ps...))
end
##

p_mu = plot(X[s],y_noise[1:N_data][s],lab="true g")
plot!(X[s],model.likelihood.μ[1][s],lab="μ_g")
p_sig = plot(X[s],h.(y_noise[1:N_data][s]),lab="λsig(true g)")
plot!(X[s],model.likelihood.λ[1]*model.likelihood.σg[1][s],lab="λsig(μ_g)")
plot(p_mu,p_sig)
# return true
