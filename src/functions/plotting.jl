using Colors
using AbstractPlotting

function AbstractPlotting.plot(model::AbstractGP;nGrid::Int=100,nsigma::Int=2)
    scene = Makie.Scene()
    plot!(scene,model,nGrid=nGrid,nsigma=nsigma)
end

function AbstractPlotting.plot!(scene::Makie.Scene,model::AbstractGP;nGrid::Int=100,nsigma::Int=2)
    if model.nDim == 1
        makie1D!(scene,model,nGrid=nGrid,nσ=nsigma)
    elseif model.nDim == 2
        makie2D!(scene,model,nGrid=nGrid,nσ=nsigma)
    else
        @error "Cannot plot model if inputs are in more than 2 dimensions"
    end
end

function makie1D!(scene::Makie.Scene,model::AbstractGP;nGrid::Int=100,nσ::Int=2)
    xmin = minimum(model.X); xmax = maximum(model.X)
    d = xmax-xmin; xmax += 0.1*d; xmin -= 0.1*d
    x_grid = collect(range(xmin,length=nGrid,stop=xmax))
    return makie1D!(scene,model,x_grid,nσ)
end

function makie1D!(scene::Scene,model::AbstractGP{T,<:RegressionLikelihood},x_grid::AbstractVector,nσ::Int) where {T}
    μ_grid,σ²_grid = proba_y(model,x_grid)
    if model.nLatent == 1
        Makie.scatter!(scene,model.X[:,1],model.y[1],markersize=0.01,color=:black)
        Makie.lines!(scene,x_grid,μ_grid,linewidth=3.0)
        Makie.fill_between!(x_grid,μ_grid.+nσ*sqrt.(σ²_grid),μ_grid-nσ*sqrt.(σ²_grid),where = trues(length(x_grid)),alpha=0.3)
        return scene
    else
        ps = []
        for i in 1:model.nLatent
            p = Makie.scatter(model.X[:,1],model.y[i],markersize=0.01,color=:black,title="y$i")
            Makie.lines!(p,x_grid,μ_grid[i],linewidth=3.0)
            Makie.fill_between!(x_grid,μ_grid[i].+nσ*sqrt.(σ²_grid[i]),μ_grid[i]-nσ*sqrt.(σ²_grid[i]),where = trues(length(x_grid)),alpha=0.3)
            push!(ps,p)
        end
        return hbox(ps...)
    end
end

function makie1D!(scene::Makie.Scene,model::AbstractGP{T,<:ClassificationLikelihood},x_grid::AbstractVector) where {T}
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

function makie2D!(scene::Makie.Scene,model::AbstractGP;nGrid::Int=100,nσ::Int=2)
    xmin = minimum.(eachcol(model.X)); xmax = maximum.(eachcol(model.X))
    d = xmax.-xmin; xmax .+= 0.01*d; xmin .-= 0.01*d
    global x1_grid = collect(range(xmin[1],length=nGrid,stop=xmax[1]))
    global x2_grid = collect(range(xmin[2],length=nGrid,stop=xmax[2]))
    global X_grid = hcat([j for i in x1_grid, j in x2_grid][:],[i for i in x1_grid, j in x2_grid][:])
    makie2D!(scene,model,x1_grid,x2_grid,X_grid,nσ)
end

function makie2D!(scene::Makie.Scene,model::AbstractGP{T,<:RegressionLikelihood},x1_grid::AbstractVector,x2_grid::AbstractVector,X_grid::AbstractMatrix,nσ::Int) where {T}
    μ_grid,σ²_grid = predict_f(model,X_grid,covf=true)
    scatter!(scene,model.X[:,1],model.X[:,2],model.y[1],markersize=0.01,color=:black)
    surface!(scene,x1_grid,x2_grid,reshape(μ_grid,length(x1_grid),length(x2_grid))')
    wireframe!(scene,x1_grid,x2_grid,reshape(μ_grid-nσ*sqrt.(σ²_grid),length(x1_grid),length(x2_grid))',transparency=true,color=RGBA(1.0,0.0,0.0,0.1))
    wireframe!(scene,x1_grid,x2_grid,reshape(μ_grid+nσ*sqrt.(σ²_grid),length(x1_grid),length(x2_grid))',transparency=true,color=RGBA(1.0,0.0,0.0,0.1))
    return scene
end

function makie2D!(scene::Makie.Scene,model::AbstractGP{T,<:ClassificationLikelihood},x1_grid::AbstractVector,x2_grid::AbstractVector,X_grid::AbstractMatrix,nσ::Int) where {T}
    y_p = proba_y(model,X_grid)
    scatter!(scene,model.X[:,1],model.X[:,2],model.y[1],markersize=0.01,color=:black)
    surface!(scene,x1_grid,x2_grid,reshape(y_p,length(x1_grid),length(x2_grid))')
    wireframe!(scene,x1_grid,x2_grid,reshape(sign.(y_p.-0.5),length(x1_grid),length(x2_grid))',transparency=true,color=RGBA(1.0,0.0,0.0,0.1))
    return scene
end


function makie2D!(scene::Makie.Scene,model::AbstractGP,x_grid::AbstractMatrix,nσ::Int)
    @error "Not implemented yet"
end
