using AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using ValueHistories
using Makie
using AbstractPlotting: vbox
using Colors
using LinearAlgebra
using GradDescent
using DelimitedFiles
cd(@__DIR__)
seed!(42)


N_class = 3
N_test = 50
N_grid = 100
minx=-5.0
maxx=5.0
noise = 1.0

N_dim=1
N_iterations = 500
m = 50
art_noise = 0.3
dpi=600
##
function create_multiclass_data(N_data,N_dim)
    σ = 0.5; N_class = N_dim+1
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
    return X,y
end

function create_regression_data(N_data,N_dim,noise=0.1)
    X = rand(N_data,N_dim)
    y = N_dim == 1 ? X.*sin.(X)+randn(N_data)*noise : X[:,1].*sin.(X[:,2])+randn(N_data)*noise
    return X,y
end

function create_classification_data(N_data,N_dim,noise=0.1)
    X = rand(N_data,N_dim)
    y = N_dim == 1 ? X.*sin.(X).-0.5 +randn(N_data)*noise : X[:,1].*sin.(X[:,2]).- 0.5 +randn(N_data)*noise
    y=  sign.(y)
    return X,y
end


function callbackmakie(model::GP)
    scene = Makie.Scene()
    if model.nDim == 1
        makie1D!(scene,model)
    elseif model.nDim == 2
        makie2D!(scene,model)
    else
        @error "Cannot plot model if inputs are in more than 2 dimensions"
    end
end

function makie1D!(scene::Scene,model::GP)
    N_grid = 100
    xmin = minimum(model.X); xmax = maximum(model.X)
    d = xmax-xmin; xmax += 0.1*d; xmin -= 0.1*d
    x_grid = collect(range(xmin,length=N_grid,stop=xmax))
    makie1D!(scene,model,x_grid)
    return scene
end

function makie1D!(scene::Scene,model::GP{<:RegressionLikelihood},x_grid::AbstractVector)
    μ_grid,σ²_grid = proba_y(model,x_grid)
    if model.nLatent == 1
        Makie.scatter!(scene,model.X[:,1],model.y[1],markersize=0.01,color=:black)
        Makie.lines!(scene,x_grid,μ_grid,linewidth=3.0)
        Makie.fill_between!(x_grid,μ_grid.+sqrt.(σ²_grid),μ_grid-sqrt.(σ²_grid),where = trues(length(x_grid)),alpha=0.3)
        return scene
    else
        ps = []
        for i in 1:model.nLatent
            p = Makie.scatter(model.X[:,1],model.y[i],markersize=0.01,color=:black,title="y$i")
            Makie.lines!(p,x_grid,μ_grid[i],linewidth=3.0)
            Makie.fill_between!(x_grid,μ_grid[i].+sqrt.(σ²_grid[i]),μ_grid[i]-sqrt.(σ²_grid[i]),where = trues(length(x_grid)),alpha=0.3)
            push!(ps,p)
        end
        scene = hbox(ps...)
    end
    return scene
end

function makie1D!(scene::Scene,model::GP{<:ClassificationLikelihood},x_grid::AbstractVector)
    μ_grid,σ²_grid = predict_f(model,x_grid,covf=true)
    py_grid = proba_y(model,x_grid)
    if model.nLatent == 1
        p = Makie.scatter(model.X[:,1],model.y[1],markersize=0.01,color=:black)
        Makie.lines!(scene,x_grid,py_grid,linewidth=3.0)
        scene = p
        # fill_between!(x_grid,μ_grid.+sqrt.(σ²_grid),μ_grid-sqrt.(σ²_grid),where = trues(length(x_grid)),alpha=0.3)
    else
        ps = []
        for i in 1:model.nLatent
            p = Makie.scatter(model.X[:,1],model.y[i],markersize=0.01,color=:black,title="y$i")
            Makie.lines!(scene,x_grid,py_grid[i],linewidth=3.0)
        end
        scene = hbox(ps...)
    end
    scene
end

function makie2D!(scene::Scene,model::GP)
    N_grid = 100
    N_fill = 1000
    xmin = minimum.(eachcol(model.X)); xmax = maximum.(eachcol(model.X))
    d = xmax.-xmin; xmax .+= 0.01*d; xmin .-= 0.01*d
    global x1_grid = collect(range(xmin[1],length=N_grid,stop=xmax[1]))
    global x2_grid = collect(range(xmin[2],length=N_grid,stop=xmax[2]))
    global x_grid = hcat([j for i in x1_grid, j in x2_grid][:],[i for i in x1_grid, j in x2_grid][:])
    μ_grid,σ²_grid = predict_f(model,x_grid,covf=true)
    z_sigma = range(minimum(μ_grid-sqrt.(σ²_grid)),maximum(μ_grid+sqrt.(σ²_grid)),length=N_fill)
    Z_min = reshape(μ_grid - sqrt.(σ²_grid),N_grid,N_grid)
    Z_max = reshape(μ_grid + sqrt.(σ²_grid),N_grid,N_grid)
    global V = [0.2((Z_min[i,j] <= z_sigma[k]) && (Z_max[i,j] >= z_sigma[k])) for i in 1:N_grid, j in 1:N_grid, k in 1:N_fill]
    scene= volume(xmin[1]..xmax[1],xmin[2]..xmax[2],minimum(z_sigma)..maximum(z_sigma),Float64.(V),algorithm=:absorption,color=RGBA(1,0,0,0.5))
    scatter!(scene,model.X[:,1],model.X[:,2],model.y[1],markersize=0.01,color=:black)
    surface!(scene,x1_grid,x2_grid,reshape(μ_grid,N_grid,N_grid))
    # fill_between!(x_grid,μ_grid.+sqrt.(σ²_grid),μ_grid-sqrt.(σ²_grid),where = trues(N_grid),alpha=0.3)
    return scene
    global y_fgrid = predict_y(model,X_grid)
    global py_fgrid = Matrix(proba_y(model,X_grid))[:,collect(values(sort(model.likelihood.ind_mapping)))]
    global μ_fgrid = predict_f(model,X_grid,covf=false)
    global cols = reshape([parse.(Colorant,RGB(py_fgrid[i,:]...)) for i in 1:N_grid*N_grid],N_grid,N_grid)
    global col_doc = [RGB(1.0,0.0,0.0),RGB(0.0,1.0,0.0),RGB(0.0,0.0,1.0)]
    global scale = 1.0
    scene = Scene()
    Makie.scatter!(scene,[1,0,0],[0,1,0],[0,0,1],color=RGBA(1,1,1,0)) #For 3D plots
    Makie.scatter!(scene,X[:,1],X[:,2],scale*(model.nLatent+1)*ones(size(X,1)),color=col_doc[y],lab="",markerstrokewidth=0.1,transparency=true,shading=false)
    Makie.surface!(scene,collect(x_grid),collect(x_grid),zeros(N_grid,N_grid),grid=:hide,color=cols',lab="",shading=false)#,size=(600,2000))#,colorbar=false,framestyle=:none,,dpi=dpi)
    Makie.lines!(scene,[xmin,xmin,xmax,xmax,xmin],[xmin,xmax,xmax,xmin,xmin],zeros(5),lab="",color=:black,linewidth=2.0,shading=false)
    tsize = 0.8
    minalpha = 0.2
    grads = [cgrad([RGBA(1,1,1,minalpha),RGBA(1,0,0,1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(0,1,0,1)]),cgrad([RGBA(1,1,1,minalpha),RGBA(0,0,1,1)])]
    Makie.text!(scene,"p(y|D)",position=(xmin,xmax,0.0),textsize=tsize,rotation=(Vec3f0(1, 1, 1), pi*2/3))
    sub = ["₃","₂","₁"]
    for i in 1:model.nLatent
        μ = μ_fgrid[collect(values(sort(model.likelihood.ind_mapping)))][i]
        μ = (μ.-minimum(μ))/(maximum(μ)-minimum(μ))
        int_cols = getindex.([grads[i]],μ)
        Makie.surface!(scene,collect(x_grid),collect(x_grid),scale*i*ones(N_grid,N_grid),color=reshape(int_cols,N_grid,N_grid)',shading=false)
        Makie.lines!(scene,[xmin,xmin,xmax,xmax,xmin],[xmin,xmax,xmax,xmin,xmin],scale*i*ones(5),lab="",color=:black,linewidth=2.0)
        Makie.text!(scene,"p(f"*sub[i]*"|D)",position = (xmin,xmax,scale*i),textsize=tsize,rotation=(Vec3f0(1, 1, 1), pi*2/3))
    end

    Makie.lines!(scene,[xmin,xmin,xmax,xmax,xmin],[xmin,xmax,xmax,xmin,xmin],scale*(model.nLatent+1)*ones(5),lab="",color=:black,linewidth=2.0)
    scene[Axis][:showgrid] = (false,false,false)
    scene[Axis][:showaxis] = (false,false,false)
    scene[Axis][:ticks][:textsize] = 0
    scene[Axis][:names][:axisnames] = ("","","")
    Makie.text!(scene,"data",position = (xmin,xmax,scale*(model.nLatent+1)),textsize=tsize,rotation=(Vec3f0(1, 1, 1), pi*2/3))
    scene.center=false
    return scene
end


function initial_lengthscale(X)
    D = pairwise(SqEuclidean(),X')
    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end
# l = sqrt(initial_lengthscale(X))

# kernel = AugmentedGaussianProcesses.RBFKernel(l,variance=10.0)
# setfixed!(kernel.fields.lengthscales)
# setfixed!(kernel.fields.variance)
autotuning = true
N_data = 500
N_dim = 1
## Regression
l=1.0
kernel = AugmentedGaussianProcesses.RBFKernel([l],dim=N_dim,variance=1.0)
X,y = create_regression_data(N_data,N_dim,0.05);# y = repeat(y,1,2); y[:,2] *= -1;
model = VGP(X,y,kernel,AugmentedStudentTLikelihood(5.0),AnalyticInference(),verbose=3)
train!(model,iterations=10)
predict_f(model,X)
callbackmakie(model)
## Classificqtion
X,y = create_classification_data(N_data,N_dim);# y = repeat(y,1,2); y[:,2] *= -1;

model = VGP(X,y,kernel,AugmentedLogisticLikelihood(),AnalyticInference(),verbose=2,Autotuning=autotuning,IndependentPriors=!true)
t_alsm = @elapsed train!(model,iterations=10)
callbackmakie(model)
