using Makie



function Makie.plot(model::GP;nGrid::Int=100)
    scene = Makie.Scene()
    plot!(scene,model,nGrid=nGrid)
end

function Makie.plot!(scene::Makie.Scene,model::GP;nGrid::Int=100)
    if model.nDim == 1
        makie1D!(scene,model,nGrid=nGrid)
    elseif model.nDim == 2
        makie2D!(scene,model,nGrid=nGrid)
    else
        @error "Cannot plot model if inputs are in more than 2 dimensions"
    end
end

function makie1D!(scene::Makie.Scene,model::GP;nGrid::Int=100)
    xmin = minimum(model.X); xmax = maximum(model.X)
    d = xmax-xmin; xmax += 0.1*d; xmin -= 0.1*d
    x_grid = collect(range(xmin,length=nGrid,stop=xmax))
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

function makie1D!(scene::Makie.Scene,model::GP{<:ClassificationLikelihood},x_grid::AbstractVector)
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

function makie2D!(scene::Makie.Scene,model::GP;nGrid::Int=100)
    N_fill = 1000
    xmin = minimum.(eachcol(model.X)); xmax = maximum.(eachcol(model.X))
    d = xmax.-xmin; xmax .+= 0.01*d; xmin .-= 0.01*d
    x1_grid = collect(range(xmin[1],length=nGrid,stop=xmax[1]))
    x2_grid = collect(range(xmin[2],length=nGrid,stop=xmax[2]))
    x_grid = hcat([j for i in x1_grid, j in x2_grid][:],[i for i in x1_grid, j in x2_grid][:])
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
end
