using AugmentedGaussianProcesses
using LinearAlgebra, Distributions, Plots
using LightGraphs
using RandomMatrices
using Random
using DeterminantalPointProcesses
using DelimitedFiles
using StatsBase, Distances


N = 1000
noise = 0.001
fy(x1,x2) = x1.*sin.(x2)+x2.*cos.(x1)
elbo(Q,Kff,y) = -0.5*dot(y,inv(Q+noise*I)*y) - 0.5*logdet(Q+noise*I) - 0.5*tr(Kff-Q)/noise-0.5*N*log(2.0*π)
function elbo_ind(C)
    Kuu = kernelmatrix(C,k)+1e-5I
    Kfu = kernelmatrix(X,C,k)
    Q = Kfu*inv(Kuu)*Kfu'
    elbo(Q,Kff,y)
end

function initial_lengthscale(X)
    if size(X,1) > 10000
        D = pairwise(SqEuclidean(),X[sample(1:size(X,1),10000,replace=false),:],dims=1)
    else
        D = pairwise(SqEuclidean(),X,dims=1)
    end
    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end

σ = sqrt(5)
X = (rand(N,2).-0.5)*5
# X = rand(Normal(0,σ),N,2)
y = fy(eachcol(X)...)

data = Float64.(readdlm("/home/theo/Downloads/abalone.data",',')[:,2:end])
X = data[:,1:end-1]; X = hcat(zscore.(eachcol(X))...)
y=data[:,end]; y .-=mean(y); y /= sqrt(var(y))
N = size(X,1)

l = sqrt(initial_lengthscale(X))
k = RBFKernel(l)
Kff = kernelmatrix(X,k)+1e-5I
pers = [0.01,0.02,0.05,0.1,0.2]#,0.3,0.4,0.5]#,120,140,160,180,200]
ms = floor.(Int64,pers.*N)#,120,140,160,180,200]
ρs = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99]
# ρs = [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]


unitr_μ,unitr_σ = Vector(undef,length(ms)),Vector(undef,length(ms))
unielbo_μ,unielbo_σ = Vector(undef,length(ms)),Vector(undef,length(ms))
offtr_μ,offtr_σ = Vector(undef,length(ms)),Vector(undef,length(ms))
offelbo_μ,offelbo_σ = Vector(undef,length(ms)),Vector(undef,length(ms))
randtr_μ,randtr_σ = Vector(undef,length(ms)),Vector(undef,length(ms))
randelbo_μ,randelbo_σ = Vector(undef,length(ms)),Vector(undef,length(ms))
kdpptr_μ,kdpptr_σ = Vector(undef,length(ms)),Vector(undef,length(ms))
kdppelbo_μ,kdppelbo_σ = Vector(undef,length(ms)),Vector(undef,length(ms))
nSamples = 10
kdppalg = DeterminantalPointProcess(Symmetric(Kff))
@progress name="Loop over m" for (j,m) in enumerate(ms)
    unitr = Vector(undef,nSamples)
    unielbo = Vector(undef,nSamples)
    offtr = Vector(undef,nSamples)
    offelbo = Vector(undef,nSamples)
    randtr = Vector(undef,nSamples)
    randelbo = Vector(undef,nSamples)
    kdpptr = Vector(undef,nSamples)
    kdppelbo = Vector(undef,nSamples)
    @progress name="iterating m=$m" for i in 1:nSamples
        unicenters = copy(X[sample(1:N,m,replace=false),:])
        Kuuuni = kernelmatrix(unicenters,k)+1e-5I
        Kfuuni = kernelmatrix(X,unicenters,k)
        Quni = Kfuuni*inv(Kuuuni)*Kfuuni'
        unitr[i] = tr(Kff-Quni)
        unielbo[i] = -elbo(Quni,Kff,y)


        offalg = OfflineKmeans(m)
        init!(offalg,X,0,k)
        Kuuoff = kernelmatrix(offalg.centers,k)+1e-5I
        Kfuoff = kernelmatrix(X,offalg.centers,k)
        Qoff = Kfuoff*inv(Kuuoff)*Kfuoff'
        offtr[i] = tr(Kff-Qoff)
        offelbo[i] = -elbo(Qoff,Kff,y)

        A = HaarMatrix(N,1.0)
        P = zeros(m,N)
        P[1:m,1:m] = Diagonal{Float64}(I,m)
        K̃ = A*Kff*A'
        K̃ᵤ = P*K̃*P'
        Q̃ = K̃*P'*inv(K̃ᵤ)*P*K̃
        randtr[i] = tr(K̃-Q̃)
        randelbo[i] = -elbo(Q̃,K̃,A*y)

        kdppset = rand(kdppalg,1,m)[1]
        Kuukdpp = kernelmatrix(X[kdppset,:],k)+1e-5I
        Kfukdpp = kernelmatrix(X,X[kdppset,:],k)
        Qkdpp = Kfukdpp*inv(Kuukdpp)*Kfukdpp'
        kdpptr[i] = tr(Kff-Qkdpp)
        kdppelbo[i] = -elbo(Qkdpp,Kff,y)
    end
    unitr_μ[j] = mean(unitr); unitr_σ[j] = sqrt(var(unitr))
    unielbo_μ[j] = mean(unielbo); unielbo_σ[j] = sqrt(var(unielbo))
    offtr_μ[j] = mean(offtr); offtr_σ[j] = sqrt(var(offtr))
    offelbo_μ[j] = mean(offelbo); offelbo_σ[j] = sqrt(var(offelbo))
    randtr_μ[j] = mean(randtr); randtr_σ[j] = sqrt(var(randtr))
    randelbo_μ[j] = mean(randelbo); randelbo_σ[j] = sqrt(var(randelbo))
    kdpptr_μ[j] = mean(kdpptr); kdpptr_σ[j] = sqrt(var(kdpptr))
    kdppelbo_μ[j] = mean(kdppelbo); kdppelbo_σ[j] = sqrt(var(kdppelbo))
end


##
circtr_μ,circtr_σ = Vector(undef,length(ρs)),Vector(undef,length(ρs))
circelbo_μ,circelbo_σ = Vector(undef,length(ρs)),Vector(undef,length(ρs))
circm_μ = Vector(undef,length(ρs))
graphtr_μ,graphtr_σ = Vector(undef,length(ρs)),Vector(undef,length(ρs))
graphelbo_μ,graphelbo_σ = Vector(undef,length(ρs)),Vector(undef,length(ρs))
graphm_μ = Vector(undef,length(ρs))
@progress name="Looping over ρ" for (j,ρ) in enumerate(ρs)
    circtr = Vector(undef,nSamples)
    circelbo = Vector(undef,nSamples)
    circm = Vector(undef,nSamples)
    graphtr = Vector(undef,nSamples)
    graphelbo = Vector(undef,nSamples)
    graphm = Vector(undef,nSamples)
    Ksparse = copy(Kff)
    Ksparse[Ksparse.<ρ] .= 0
    @progress name="iterating ρ=$ρ" for i in 1:nSamples
        alg = CircleKMeans(ρ)
        init!(alg,X,0,k)
        update!(alg,X[shuffle(1:N),:],0,k)
        Kuucirc = kernelmatrix(alg.centers,k)+1e-5I
        Kfucirc = kernelmatrix(X,alg.centers,k)
        Qcirc = Kfucirc*inv(Kuucirc)*Kfucirc'
        circtr[i] = tr(Kff-Qcirc)
        circelbo[i] = -elbo(Qcirc,Kff,y)
        circm[i] = alg.k

        g = SimpleGraph(Ksparse)
        gcenters = LightGraphs.dominating_set(g,MinimalDominatingSet())
        Kuugraph = kernelmatrix(X[gcenters,:],k)+1e-5I
        Kfugraph = kernelmatrix(X,X[gcenters,:],k)
        Qgraph =Kfugraph*inv(Kuugraph)*Kfugraph'
        graphtr[i] = tr(Kff-Qgraph)
        graphelbo[i] = -elbo(Qgraph,Kff,y)
        graphm[i] = length(gcenters)
    end
    circtr_μ[j] = mean(circtr); circtr_σ[j] = sqrt(var(circtr))
    circelbo_μ[j] = mean(circelbo); circelbo_σ[j] = sqrt(var(circelbo))
    circm_μ[j] = mean(circm)
    graphtr_μ[j] = mean(graphtr); graphtr_σ[j] = sqrt(var(graphtr))
    graphelbo_μ[j] = mean(graphelbo); graphelbo_σ[j] = sqrt(var(graphelbo))
    graphm_μ[j] = mean(graphm)
end
## TRACE
plot(title="Trace")
plot!(ρs,circtr_μ,lab="Circle",color=1)
plot!(ρs,circtr_μ+2*circtr_σ,fill=circtr_μ-2*circtr_σ,lab="",alpha=0.3,color=1)
plot!(ρs,graphtr_μ,lab="Graph",color=2)
plot!(ρs,graphtr_μ+2*graphtr_σ,fill=graphtr_μ-2*graphtr_σ,lab="",alpha=0.3,color=2)
## ELBO
plot(title="ELBO")
plot!(ρs,circelbo_μ,lab="Circle",color=1,xaxis=:log)
plot!(ρs,circelbo_μ+2*circelbo_σ,fill=circelbo_μ-2*circelbo_σ,lab="",alpha=0.3,color=1)
plot!(ρs,graphelbo_μ,lab="Graph",color=2)
plot!(ρs,graphelbo_μ+2*graphelbo_σ,fill=graphelbo_μ-2*graphelbo_σ,lab="",alpha=0.3,color=2)
## #Ind points
plot(title="# Inducing points")
plot!(ρs,circm_μ,lab="Circle",color=1)
plot!(ρs,graphm_μ,lab="Graph",color=2)

##
ondpptr = Vector(undef,nSamples)
ondppelbo = Vector(undef,nSamples)
ondppm = Vector(undef,nSamples)
dpptr = Vector(undef,nSamples)
dppelbo = Vector(undef,nSamples)
dppm = Vector(undef,nSamples)
dppalg = DeterminantalPointProcess(Symmetric(Kff))
@progress name="Looping for DPP" for i in 1:nSamples
    ondppalg = DPPAlg(0.85,k)
    init!(ondppalg,X,0,k)
    update!(ondppalg,X[shuffle(1:N),:],0,k)
    Kuuondpp = kernelmatrix(ondppalg.centers,k)+1e-5I
    Kfuondpp = kernelmatrix(X,ondppalg.centers,k)
    Qondpp = Kfuondpp*inv(Kuuondpp)*Kfuondpp'
    ondpptr[i] = tr(Kff-Qondpp)
    ondppelbo[i] = -elbo(Qondpp,Kff,y)
    ondppm[i] = ondppalg.k

    dppset = rand(dppalg,1)[1]
    Kuudpp = kernelmatrix(X[dppset,:],k)+1e-5I
    Kfudpp = kernelmatrix(X,X[dppset,:],k)
    Qdpp = Kfudpp*inv(Kuudpp)*Kfudpp'
    dpptr[i] = tr(Kff-Qdpp)
    dppelbo[i] = -elbo(Qdpp,Kff,y)
    dppm[i] = length(dppset)
end
##
alpha_v=0.2
nσ = 1
## TRACE
ptrace = plot(title="Trace",xlabel="#Inducing points",ylabel="Trace(K_ff-Q_ff)")
plot!(ms,unitr_μ,lab="Uniform",color=6)
plot!(ms,unitr_μ+nσ*unitr_σ,fill=unitr_μ-nσ*unitr_σ,lab="",alpha=alpha_v,color=6)
plot!(ms,offtr_μ,lab="KMeans",color=1)
plot!(ms,offtr_μ+nσ*offtr_σ,fill=offtr_μ-nσ*offtr_σ,lab="",alpha=alpha_v,color=1)
plot!(ms,randtr_μ,lab="Rand",color=2)
plot!(ms,randtr_μ+nσ*randtr_σ,fill=randtr_μ-nσ*randtr_σ,lab="",alpha=alpha_v,color=2)
plot!(ms,kdpptr_μ,lab="K-DPP",color=5)
plot!(ms,kdpptr_μ+nσ*kdpptr_σ,fill=kdpptr_μ-nσ*kdpptr_σ,lab="",alpha=alpha_v,color=5)
plot!(circm_μ,circtr_μ,lab="Circle",color=3)
plot!(circm_μ,circtr_μ+nσ*circtr_σ,fill=circtr_μ-nσ*circtr_σ,lab="",alpha=alpha_v,color=3)
plot!(graphm_μ,graphtr_μ,lab="Graph",color=4)
plot!(graphm_μ,graphtr_μ+nσ*graphtr_σ,fill=graphtr_μ-nσ*graphtr_σ,lab="",alpha=alpha_v,color=4)
scatter!(ondppm,ondpptr,lab="Sequential DPP",color=:black,markersize=1.0,markerstrokewidth=0)
scatter!(dppm,dpptr,lab="DPP",color=:red,markersize=1.0,markerstrokewidth=0)


## ELBO
pelbo = plot(title="Neg. ELBO",xlabel="#Inducing points",ylabel="-ELBO")
plot!(ms,unielbo_μ,lab="Uniform",color=6)
plot!(ms,unielbo_μ+nσ*unielbo_σ,fill=unielbo_μ-nσ*unielbo_σ,lab="",alpha=alpha_v,color=6)
plot!(ms,offelbo_μ,lab="KMeans",color=1)
plot!(ms,offelbo_μ+nσ*offelbo_σ,fill=offelbo_μ-nσ*offelbo_σ,lab="",alpha=alpha_v,color=1)
plot!(ms,randelbo_μ,lab="Rand",color=2)
plot!(ms,randelbo_μ+nσ*randelbo_σ,fill=randelbo_μ-nσ*randelbo_σ,lab="",alpha=alpha_v,color=2)
plot!(ms,kdppelbo_μ,lab="K-DPP",color=2)
plot!(ms,kdppelbo_μ+nσ*kdppelbo_σ,fill=kdppelbo_μ-nσ*kdppelbo_σ,lab="",alpha=alpha_v,color=2)
plot!(circm_μ,circelbo_μ,lab="Circle",color=3)
plot!(circm_μ,circelbo_μ+nσ*circelbo_σ,fill=circelbo_μ-nσ*circelbo_σ,lab="",alpha=alpha_v,color=3)
plot!(graphm_μ,graphelbo_μ,lab="Graph",color=4)
plot!(graphm_μ,graphelbo_μ+nσ*graphelbo_σ,fill=graphelbo_μ-nσ*graphelbo_σ,lab="",alpha=alpha_v,color=4)
scatter!(ondppm,dppelbo,lab="Sequential DPP",color=:black,markersize=1.0,markerstrokewidth=0)
scatter!(dppm,dppelbo,lab="DPP",color=:red,markersize=1.0,markerstrokewidth=0)
##
if size(X,2) == 2
    scatter(eachcol(X)...,zcolor=y,lab="")
    contour!(minimum(X[:,1]):0.01:maximum(X[:,1]),minimum(X[:,2]):0.01:maximum(X[:,2]),fy,fill=true,fillalpha=0.2,alpha=0.2,colorbar=false)
    display(scatter!(eachcol(X)...,markersize=2.0,color=:black,lab=""))
end
display(ptrace)
display(pelbo)
