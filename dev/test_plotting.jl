using AugmentedGaussianProcesses
using Plots, LinearAlgebra, Distributions
using KernelFunctions
N = 100
λ = 3.0
X = rand(N,1)
ngrid = 100
xrange = collect(range(0,1,length=ngrid))

K = kernelmatrix(SqExponentialKernel(10.0),vcat(X,xrange),obsdim=1)
y1 = rand(MvNormal(zeros(N+size(xrange,1)),K+0.02I))
y2 = rand(MvNormal(zeros(N+size(xrange,1)),K+0.02I))
scatter(xrange,y1[N+1:end])
scatter!(xrange,y2[N+1:end])

model = MOSVGP(X,[y1[1:N],y2[1:N]],SqExponentialKernel(10.0),LaplaceLikelihood(),AnalyticVI(),2,10,verbose=3)
model = VGP(X,y1[1:N],SqExponentialKernel(10.0),LaplaceLikelihood(),AnalyticSVI(10),verbose=3)
# model = SVGP(X,n_train,RBFKernel(0.1),PoissonLikelihood(),AnalyticVI(),100,verbose=3)
train!(model,100)

pred_f,sig_f = predict_f(model,xrange,covf=true)
scatter(eachcol(X)...,zcolor=n_train)
scatter(X,y1[1:N])
scatter(xrange,y1[N+1:end])
scatter!(xrange,y2[N+1:end])
plot(model,xrange)
plot(model,xrange,showX=true)

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
