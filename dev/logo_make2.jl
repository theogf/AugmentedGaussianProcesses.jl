using Makie, Colors
using AugmentedGaussianProcesses
using LinearAlgebra
const AGP = AugmentedGaussianProcesses

up = 0.1
yshift = -0.5*up
Xgreen = [0.5]; ygreen = [up+yshift]
Xred = [0.0]; yred = [0.0+yshift]
Xpurple = [1.0]; ypurple = [0.0+yshift]

green = RGB(0.133,0.541,0.133)
red=  RGB(0.8,0.2,0.2)
purple = RGB(0.702,0.332,0.8)

p_green = Point2(Xgreen[1],ygreen[1])
p_red = Point2(Xred[1],yred[1])
p_purple = Point2(Xpurple[1],ypurple[1])

k = RBFKernel(0.2,variance=0.2)
noise = 0.0001
gp = GP([Xred[1],Xgreen[1],Xpurple[1]],[yred[1],ygreen[1],ypurple[1]],k,noise=noise)

grid = collect(range(-1,9.5,length=1000))

p,sig = proba_y(gp,grid);sig = sqrt.(sig)

nsig = 0.1
alpha = 0.4
lw = 3.0
ms = 0.6
function f(mu,sigma)
    function f(x)
         exp(-0.5*norm(x-mu)^2/sigma^2)
    end
end

ygrid = range(-2,3,length=200)


ksig = 2.0
# fgrid = f.(pgreen,siggreen)
Y = hcat([f(mu,sigma).(ygrid) for (mu,sigma) in zip(p,sig)]...)
Xgrid = hcat([j for i in grid, j in ygrid][:],[i for i in grid, j in ygrid][:])

scene = plot(grid,p,color=:black,linewidth=lw,backgroundcolor=:white)
fill_between!(grid,p-nsig*sig,p+nsig*sig,where = trues(size(grid)),color=RGBA(colorant"blue",alpha),transparency=true)
# plot!(grid,pred,color=red,linewidth=lw)
# fill_between!(grid,pred-nsig*sigred,pred+nsig*sigred,where = trues(size(grid)),color=RGBA(red,alpha))
# plot!(grid,ppurple,color=purple,linewidth=lw)
# fill_between!(grid,ppurple-nsig*sigpurple,ppurple+nsig*sigpurple,where = trues(size(grid)),color=RGBA(purple,alpha)  )
scatter!([p_green,p_red,p_purple],color=[green,red,purple],markersize=ms)#,limits=FRect2D((-1.0,-1.0),(100.0,2.5)))
text!("AugmentedGaussianProcesses.jl",position=(2.1,0.0),font="Franklin Gothic Medium",textsize=0.5,align=(:left,:center),color=:white)
scene[Axis][:showgrid] = (false,false)
scene[Axis][:showaxis] = (false,false)
scene[Axis][:ticks,:textsize] = (0.0,0.0)
scene[Axis][:names][:axisnames] = ("","")
save("logo2.png",scene)

# scatter(eachrow(Xgrid)...,color=RGBA.(green,Ygreen[:]))
# scatter(eachrow(Xgrid)...,color=RGBA.(red,Yred[:]))
# scatter(eachrow(Xgrid)...,color=RGBA.(purple,Ypurple[:]))
