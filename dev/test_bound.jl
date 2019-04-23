using AugmentedGaussianProcesses
using LinearAlgebra, Distributions, Plots



threshold = 0.8
N = 1000
A = qr(rand(N,N)).Q
m = 50
P = zeros(m,N)
P[1:m,1:m] = Diagonal{Float64}(I,m)
P2 = zeros(N,m)
noise = 0.001
y(x1,x2) = x1.*sin.(x2)+x2.*cos.(x1)
elbo(Q,Kff,y) = -0.5*dot(y,inv(Q+noise*I)*y) - 0.5*logdet(Q+noise*I) - 0.5*tr(Kff-Q)/noise
k = RBFKernel(2.0)
alg = CircleKMeans(threshold)
dppalg = DPPAlg(0.85,k)
σ = sqrt(5)
alltr = []
allbound = []
@progress for N in 1000#100:10:500
    global X = rand(Normal(0,σ),N,2)
    # global X = rand(N,2)*5
    # global X = reshape(collect(0:0.5:N) + randn(size(0:0.5:N))*0.1,:,1)
    global Kff = kernelmatrix(X,k)+1e-5I
    init!(alg,X,0,k)
    init!(dppalg,X,0,k)
    update!(alg,X,0,k)
    update!(dppalg,X,0,k)
    global Kuu = kernelmatrix(alg.centers,k)+1e-5I
    global Kuudpp = kernelmatrix(dppalg.centers,k)+1e-5I
    global Kfu = kernelmatrix(X,alg.centers,k)
    global Kfudpp = kernelmatrix(X,dppalg.centers,k)
    global trk = tr(Kff-Kfu*inv(Kuu)*Kfu')
    global trkdpp = tr(Kff-Kfudpp*inv(Kuudpp)*Kfudpp')
    @show
    push!(alltr,trk)
    upbound = (N-alg.k)*(1-threshold^2/maximum(eigvals(Kuu)))
    l = getlengthscales(k)
    # upbound = (N-alg.k)*(1-mean((x->sum(abs2.(x))).(eachrow(Kfu)))/maximum(eigvals(Kuu)))
    # upbound = (N-alg.k)*(1-mean(mean.(eachrow(Kfu)))*threshold^2/maximum(eigvals(Kuu)))
    upbound = (N-alg.k)*(1-alg.k*(l/(sqrt(l^2+σ^2))-(l/sqrt(l^2+4σ^2)-l^2/(l^2+2σ^2)))/maximum(eigvals(Kuu)))
    push!(allbound,upbound)
end
offalg = OfflineKmeans(50)
init!(offalg,X,0,k)
offalg.centers
Kuuoff = kernelmatrix(offalg.centers,k)
Kfuoff = kernelmatrix(X,offalg.centers,k)
K̃ = A*Kff*A'
K̃ᵤ = P*K̃*P'
display(P)
Q̃ = K̃*P'*inv(K̃ᵤ)*P*K̃
trkrand = tr(K̃-Q̃)
trkoff = tr(Kff-Kfuoff*inv(Kuuoff)*Kfuoff')
# trkrand = tr(P'*inv(P*inv(K̃)*P')*P)
elbo_rand = elbo(Q̃,Kff,A*y(eachcol(X)...))
elbo_dpp = elbo(Kfudpp*inv(Kuudpp)*Kfudpp',Kff,y(eachcol(X)...))
elbo_circle = elbo(Kfu*inv(Kuu)*Kfu',Kff,y(eachcol(X)...))
elbo_off = elbo(Kfuoff*inv(Kuuoff)*Kfuoff',Kff,y(eachcol(X)...))
bar(["trace circleK","trace random matrix","trace DPP","trace kmeans"],[trk,trkrand,trkdpp,trkoff],lab="")
bar(["trace² circleK","trace² random matrix","trace² dpp","trace² kmeans"],[tr((Kff-Kfu*inv(Kuu)*Kfu')^2),tr((K̃-Q̃)^2),tr((Kff-Kfudpp*inv(Kuudpp)*Kfudpp')^2),tr((Kff-Kfuoff*inv(Kuuoff)*Kfuoff')^2)],lab="")
bar(["ELBO circleK","ELBO random matrix","ELBO dpp","ELBO kmeans"],[elbo_circle,elbo_rand,elbo_dpp,elbo_off],lab="")
# plot(100:10:500,alltr,lab="Trace",xlabel="N")
# display(plot!(100:10:500,allbound,lab="Upperbound"))
scatter(eachcol(X)...,alpha=0.3,markerstrokewidth=0)
scatter!(eachcol(dppalg.centers)...)
display(scatter!(eachcol(alg.centers)...))
