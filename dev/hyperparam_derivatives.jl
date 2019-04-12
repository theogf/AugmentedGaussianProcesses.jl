using Distributions, LinearAlgebra, Plots
pyplot()
using AugmentedGaussianProcesses
const AGP = AugmentedGaussianProcesses

N_data = 500
N_dim = 1
M = 10
σ = 0.05

X = rand(N_data,N_dim)
y = rand(MvNormal(kernelmatrix(X,RBFKernel(0.4))+σ*I))
X_grid = reshape(collect(range(0,1,length=500)),:,1)
Xu = rand(M,N_dim)
elbo_y(y,Kfu,Kuu,σ) = -0.5*dot(y,inv(Kfu*inv(Kuu)*Kfu'+σ*I)*y)
elbo_det(Kfu,Kuu,σ) = -0.5*logdet(Kfu*inv(Kuu)*Kfu'+σ*I)
elbo_tr(Kff,Kfu,Kuu,σ) = -0.5*tr(Kff-Kfu*inv(Kuu)*Kfu')/σ
elbo_tot(y,Kff,Kfu,Kuu,σ) = elbo_y(y,Kfu,Kuu,σ)+elbo_det(Kfu,Kuu,σ)+elbo_tr(Kff,Kfu,Kuu,σ)-N_data*log(2*pi)
delbo_y(y,Q,dQ) = 0.5*dot(y'*inv(Q),dQ*inv(Q)*y)
delbo_det(Q,dQ) = -0.5*tr(inv(Q)*dQ)
delbo_tr(Jff,dQ,σ) = -0.5*tr(Jff-dQ)/σ
delbo(y,Jff,Q,dQ) = delbo_y(y,Q,dQ)+delbo_det(Q,dQ)+delbo_tr(Jff,dQ,σ)
A(Kuu,Kfu,σ) = Kuu*Σ(Kuu,Kfu,σ)*Kuu
Σ(Kuu,Kfu,σ) = inv(Kuu+Kfu'*Kfu/σ)
μ(Kuu,Kfu,y,σ) = Kuu*Σ(Kuu,Kfu,σ)*Kfu'*y/σ

##
Nl = 100
l = range(-0.7,0.6,length=Nl)
L = zeros(Nl);  Ly = zeros(Nl); Ldet = zeros(Nl); Ltr = zeros(Nl);
dL = zeros(Nl); dLy = zeros(Nl); dLdet = zeros(Nl); dLtr = zeros(Nl)


@progress for (i,l) in enumerate(l)
    k = RBFKernel(10^l)

    Kff = kernelmatrix(X,k)
    Kuu = kernelmatrix(Xu,k)+1e-5I
    Kfu = kernelmatrix(X,Xu,k)

    Jff = AGP.kernelderivativematrix(X,k)
    Jfu = AGP.kernelderivativematrix(X,Xu,k)
    Juu = AGP.kernelderivativematrix(Xu,k)
    Q = Kfu*inv(Kuu)*Kfu'+σ*I
    dQ = Jfu*inv(Kuu)*Kfu'+Kfu*inv(Kuu)*Jfu'-Kfu*inv(Kuu)*Juu*inv(Kuu)*Kfu'
    L[i] = elbo_tot(y,Kff,Kfu,Kuu,σ); Ly[i] = elbo_y(y,Kfu,Kuu,σ)
    Ldet[i] = elbo_det(Kfu,Kuu,σ); Ltr[i] = elbo_tr(Kff,Kfu,Kuu,σ)
    dL[i] = delbo(y,Jff,Q,dQ); dLy[i] = delbo_y(y,Q,dQ)
    dLdet[i] = delbo_det(Q,dQ); dLtr[i] = delbo_tr(Jff,dQ,σ)
    # println("ELBO : $(elbo_tot(y,Kff,Kfu,Kuu,σ)), Y : $(elbo_y(y,Kfu,Kuu,σ)), det : $(elbo_det(Kfu,Kuu,σ)), tr : $(elbo_tr(Kff,Kfu,Kuu,σ)), const : $(-N_data*log(2*pi)) ")
    # println("derivative ELBO : $(delbo(y,Jff,Q,dQ)), Y : $(delbo_y(y,Q,dQ)), det : $(delbo_det(Q,dQ)), tr : $(delbo_tr(Jff,dQ))")
end
plot(l,L,lab="ELBO")
plot!(l,Ly,lab="ELBO (data)")
plot!(l,Ldet,lab="ELBO (det)")
pELBO = plot!(l,Ltr,lab="ELBO (tr)",xlabel="log(l)",title="ELBO")
pdELBO = plot(l,dL,lab="dELBO")
plot!(l,dLy,lab="dELBO (data)")
plot!(l,dLdet,lab="dELBO (det)")
hline!(pdELBO,[0.0],lab="",color="black")
pdELBO = plot!(l,dLtr,lab="dELBO (tr)",xlabel="log(l)",title="Derivatives ELBO")
scatter(X[:,1],y,lab="data")
# scatter!(Xu[:,1],μ(Kuu,Kfu,y,σ),lab="ind_points",color="red")
# kstar = kernelmatrix(X_grid,Xu,k)
# σstar = 2*sqrt.(kerneldiagmatrix(X_grid,k).+1e-5-diag(kstar*inv(Kuu)*(I-A(Kuu,Kfu,σ)*inv(Kuu))*kstar').+σ)
# plot!(X_grid,kstar*inv(Kuu)*μ(Kuu,Kfu,y,σ))
# plot!(X_grid,kstar*inv(Kuu)*μ(Kuu,Kfu,y,σ)-σstar,fill=kstar*inv(Kuu)*μ(Kuu,Kfu,y,σ)+σstar,alpha=0.3,lab="")
plot(pELBO,pdELBO,layout=(2,1))
10^l[findmax(L)[2]]
