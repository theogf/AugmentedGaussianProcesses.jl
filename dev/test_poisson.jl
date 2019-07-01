using AugmentedGaussianProcesses
using Plots, LinearAlgebra, Distributions
using SpecialFunctions
using LaTeXStrings
N = 100
λ = 3.0
X = rand(N,2)
ngrid = 100
xrange = collect(range(0,1,length=ngrid))
Xrange = make_grid(xrange,xrange)

K = kernelmatrix(vcat(X,Xrange),RBFKernel(0.1))
y = rand(MvNormal(zeros(N+size(Xrange,1)),K+1e-2I))
n = rand.(Poisson.(λ.*AugmentedGaussianProcesses.logistic.(y)))
n_train = n[1:N]
ptrue = contourf(xrange,xrange,reshape(n[N+1:end],ngrid,ngrid))
model = VGP(X,n_train,RBFKernel(0.1),PoissonLikelihood(),AnalyticSVI(10),verbose=3)
# model = SVGP(X,n_train,RBFKernel(0.1),PoissonLikelihood(),AnalyticVI(),100,verbose=3)
train!(model,iterations=100)
norm(proba_y(model,X)-n_train)
pred_f = predict_y(model,Xrange)
scatter(eachcol(X)...,zcolor=n_train)
pyplot()
ppred = contourf(xrange,xrange,reshape(pred_f,ngrid,ngrid),title=L"Prediction  \lambda")
pltrue = contourf(xrange,xrange,reshape(λ.*AugmentedGaussianProcesses.logistic.(y[N+1:end]),ngrid,ngrid),title=L"True \lambda")
plot(ppred,pltrue)
model.likelihood.λ[1]
mean(abs.(proba_y(model,X)-λ.*AugmentedGaussianProcesses.logistic.(y[1:N])))
mean(abs.(proba_y(model,Xrange)-λ.*AugmentedGaussianProcesses.logistic.(y[N+1:end])))

using Makie, Colors
function callbackplotmakie(io)
    function callbackmakie(model,iter)
        if iter%1 == 0
            global scene, mainscene, lplot,kplot,i
            y_pred = predict_y(model,Xrange)
            mainscene[2][3][] = Float32.(reshape(y_pred,ngrid,ngrid))
            update_cam!(mainscene,truecam)
            recordframe!(io)
        end
    end
end
##
y_pred = zeros(ngrid,ngrid)+rand(ngrid,ngrid)*0.01
scene = Makie.surface(xrange,xrange,reshape(y_pred,ngrid,ngrid),shading=false)
truecam = cameracontrols(mainscene);
mainscene = Makie.surface(xrange,xrange,y_pred',shading=false);
mainscene.center = false
points = [(Point3(x,y,0),Point3(x,y,n)) for (x,y,n) in zip(eachcol(X)...,n[1:N])]
Makie.linesegments!(mainscene,points,color=RGBA(colorant"blue",0.2));
update_cam!(mainscene,truecam)
mainscene[Axis][:showgrid] = (false,false,false)
# mainscene[Axis][:showaxis] = (false,false,false)
mainscene[Axis][:ticks][:textsize] = 0
mainscene[Axis][:names][:axisnames] = ("","","")
mainscene[Axis][:scale] = [4.0,4.0,0.01]
scene =mainscene
batchsize = 100
model = VGP(X,n_train,AugmentedGaussianProcesses.RBFKernel(0.01),PoissonLikelihood(),AnalyticVI(),verbose=2)
it = 0
iterations = 1
record(scene,string(@__DIR__,"poisson.gif",),framerate=20) do io
    # @progress for (X_batch,y_batch) in eachbatch((X_train,y_train),size=batchsize,obsdim=1)
    # @progress for (X_batch,y_batch) in RandomBatches((X_train,y_train),size=batchsize,count=500,obsdim=1)
        train!(model,iterations=500,callback=callbackplotmakie(io))
        # global it += 1
        # push!(i,it)
        # train!(model,X_batch,y_batch,iterations=10,callback=callbackplotgrid)
    # end
end
