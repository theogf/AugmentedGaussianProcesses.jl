using AugmentedGaussianProcesses
using LinearAlgebra, Distributions, Plots
using LightGraphs
using RandomMatrices

threshold = 0.7
N = 1000
m = 50
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
    update!(alg,X,0,k)
    global Kuu = kernelmatrix(alg.centers,k)+1e-5I
    global Kfu = kernelmatrix(X,alg.centers,k)
    global trk = tr(Kff-Kfu*inv(Kuu)*Kfu')
    push!(alltr,trk)

    init!(dppalg,X,0,k)
    update!(dppalg,X,0,k)
    global Kuudpp = kernelmatrix(dppalg.centers,k)+1e-5I
    global Kfudpp = kernelmatrix(X,dppalg.centers,k)
    global trkdpp = tr(Kff-Kfudpp*inv(Kuudpp)*Kfudpp')

    Ksparse = copy(Kff)
    Ksparse[Ksparse.<threshold] .= 0
    global g = SimpleGraph(Ksparse)
    global gcenters = LightGraphs.dominating_set(g,MinimalDominatingSet())
    global Kuugraph = kernelmatrix(X[gcenters,:],k)+1e-5I
    global Kfugraph = kernelmatrix(X,X[gcenters,:],k)
    global trkgraph = tr(Kff-Kfugraph*inv(Kuugraph)*Kfugraph')

    upbound = (N-alg.k)*(1-threshold^2/maximum(eigvals(Kuu)))
    l = getlengthscales(k)
    # upbound = (N-alg.k)*(1-mean((x->sum(abs2.(x))).(eachrow(Kfu)))/maximum(eigvals(Kuu)))
    # upbound = (N-alg.k)*(1-mean(mean.(eachrow(Kfu)))*threshold^2/maximum(eigvals(Kuu)))
    upbound = (N-alg.k)*(1-alg.k*(l/(sqrt(l^2+σ^2))-(l/sqrt(l^2+4σ^2)-l^2/(l^2+2σ^2)))/maximum(eigvals(Kuu)))
    push!(allbound,upbound)
end
m = length(gcenters)
offalg = OfflineKmeans(m)
init!(offalg,X,0,k)
offalg.centers
Kuuoff = kernelmatrix(offalg.centers,k)
Kfuoff = kernelmatrix(X,offalg.centers,k)

A = HaarMatrix(N,1.0)
# A = qr(rand(N,N)).Q
P = zeros(m,N)
P[1:m,1:m] = Diagonal{Float64}(I,m)
P2 = zeros(N,m)
K̃ = A*Kff*A'
K̃ᵤ = P*K̃*P'
Q̃ = K̃*P'*inv(K̃ᵤ)*P*K̃
trkrand = tr(K̃-Q̃)
trkoff = tr(Kff-Kfuoff*inv(Kuuoff)*Kfuoff')
# trkrand = tr(P'*inv(P*inv(K̃)*P')*P)
elbo_rand = elbo(Q̃,Kff,A*y(eachcol(X)...))
elbo_dpp = elbo(Kfudpp*inv(Kuudpp)*Kfudpp',Kff,y(eachcol(X)...))
elbo_circle = elbo(Kfu*inv(Kuu)*Kfu',Kff,y(eachcol(X)...))
elbo_off = elbo(Kfuoff*inv(Kuuoff)*Kfuoff',Kff,y(eachcol(X)...))
elbo_graph = elbo(Kfugraph*inv(Kuugraph)*Kfugraph',Kff,y(eachcol(X)...))
b1 = bar(["circleK","random matrix","DPP","kmeans","graph"],[trk,trkrand,trkdpp,trkoff,trkgraph],lab="",title="Trace")
b2 = bar(["M circle","M random matrix","M DPP","M kmeans","M Graph"],[alg.k,m,dppalg.k,m,length(gcenters)],lab="",title="Ind. Points")
b3 = bar(["trace² circleK","trace² random matrix","trace² dpp","trace² kmeans","trace2 graph"],[tr((Kff-Kfu*inv(Kuu)*Kfu')^2),tr((K̃-Q̃)^2),tr((Kff-Kfudpp*inv(Kuudpp)*Kfudpp')^2),tr((Kff-Kfuoff*inv(Kuuoff)*Kfuoff')^2),tr((Kff-Kfugraph*inv(Kuugraph)*Kfugraph')^2)],lab="",title="trace squared")
b4 = bar(["circleK","random","dpp","kmeans","graph"],[-elbo_circle,-elbo_rand,-elbo_dpp,-elbo_off,-elbo_graph] ,lab="",title="neg ELBO")
# plot(100:10:500,alltr,lab="Trace",xlabel="N")
# display(plot!(100:10:500,allbound,lab="Upperbound"))
scatter(eachcol(X)...,alpha=0.3,markerstrokewidth=0);
scatter!(eachcol(dppalg.centers)...,lab="DPP")
scatter!(eachcol(X[gcenters,:])...,lab="Graph")
display(scatter!(eachcol(alg.centers)...,lab="Circle"))
display(plot(b1,b2,b3,b4))




## Work on #points vs expected #

α = 0.01
N = 1000
Adiag = [min(i*α,1.0) for i in 1:N]
Aoffdiag = [max(0.0,1.0-i*α) for i in 1:(N-1)]
A = Bidiagonal(Adiag,Aoffdiag,:L)
P0 = zeros(N); P0[1] = 1.0
Aplus = A
v = zeros(N)
@progress for n in 1:1000
    global Aplus
    v[n] = findmax(Aplus*P0)[2]
    Aplus *= A
end
plot(1:1000,v,lab="")
vals,vectors = eigen(A)
