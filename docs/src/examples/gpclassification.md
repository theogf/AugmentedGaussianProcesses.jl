```@meta
EditURL = "<unknown>/docs/examples/gpclassification.jl"
```

```@example gpclassification
using Plots; pyplot()
using HTTP, CSV, DataFrames
using AugmentedGaussianProcesses


data = HTTP.get("https://www.openml.org/data/get_csv/1586217/phpwRjVjk")
data = CSV.read(data.body)
data.Class[data.Class .== 2] .= -1
data = Matrix(data)
X = data[:, 1:2]
Y = data[:, end]
```

Run sparse classification with increasing number of inducing points

```@example gpclassification
Ms = [4, 8, 16, 32, 64]
models = Vector{AbstractGPModel}(undef, length(Ms) + 1)
kernel = transform(SqExponentialKernel(), 1.0)
for (i, num_inducing) in enumerate(Ms)
    @info "Training with $(num_inducing) points"
    m = SVGP(X, Y,
            kernel,
            LogisticLikelihood(),
            AnalyticVI(),
            num_inducing,
            optimiser = false,
            Zoptimiser = false
        )
    @time train!(m, 20)
    models[i] = m
end

@info "Running full model"
mfull = VGP(X, Y,
            kernel,
            LogisticLikelihood(),
            AnalyticVI(),
            optimiser = false
            )
@time train!(mfull, 5)
models[end] = mfull

function compute_grid(model, n_grid=50)
    mins = [-3.25,-2.85]
    maxs = [3.65,3.4]
    x_lin = range(mins[1], maxs[1], length=n_grid)
    y_lin = range(mins[2], maxs[2], length=n_grid)
    x_grid = Iterators.product(xlin, ylin)
    y_grid, _ =  proba_y(model,vec(collect.(x_grid)))
    return y_grid, x_lin, y_lin
end

function plot_data(X, Y)
    Plots.scatter(eachcol(X)...,
                group = Y,
                alpha=0.2,
                markerstrokewidth=0.0,
                lab="",
                size=(300,500)
            )
end

function plot_model(model, X, Y, title = nothing)
    n_grid = 50
    y_pred, x_lin, y_lin = compute_grid(model, n_grid)
    title = if isnothing(title)
        (model isa SVGP ? "M = $(dim(model[1]))" : "full")
    else
        title
    end
    p = plot_data(X, Y)
    Plots.contour!(p,
                x_lin, y_lin,
                reshape(y_pred, n_grid, n_grid)',
                cbar=false, levels=[0.5],
                fill=false, color=:black,
                linewidth=2.0,
                title=title
                )
    if model isa SVGP
        Plots.scatter!(p,
                    AGP.Zview(model[1]),
                    msize=2.0, color="black",
                    lab="")
    end
    return p
end

Plots.plot(plot_model.(models, Ref(X), Ref(Y))...,
            layout=(1, length(models)),
            size=(1000, 200)
            )
```

## Comparison with Bayesian SVM

```@example gpclassification
mbsvm = VGP(X, Y,
            kernel,
            BayesianSVM(),
            AnalyticVI(),
            optimiser = false
            )
@time train!(mbsvm, 5)

Plots.plot(plot_model.([models[end], mbsvm], Ref(X), Ref(Y), ["Logistic", "BSVM"])...,
            layout=(1, 2))
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

