# # Gaussian Process Classification
# 
# ## Preliminary steps
#
# ### Loading necessary packages

using Plots
using HTTP, CSV
using DataFrames: DataFrame
using AugmentedGaussianProcesses
using MLDataUtils

# ### Loading the banana dataset from OpenML
data = HTTP.get("https://www.openml.org/data/get_csv/1586217/phpwRjVjk")
data = CSV.read(data.body, DataFrame)
data.Class[data.Class .== 2] .= -1
data = Matrix(data)
X = data[:, 1:2]
Y = Int.(data[:, end]);
(X_train, y_train), (X_test, y_test) = splitobs((X, Y), 0.5, ObsDim.First()) # We split the data into train and test set

# ### We create a function to visualize the data
function plot_data(X, Y; size=(300, 500))
    return Plots.scatter(
        eachcol(X)...; xlabel="x", ylabel="y", group=Y, alpha=0.2, markerstrokewidth=0.0, lab="", size=size
    )
end
plot_data(X, Y; size=(500, 800))

# ## Model initialization and training
# Using Gaussian processes to solve binary classification problem is usually defined as
# ```math
#   y \sim \mathrm{Bernoulli}(h(f))
# ```
# where ``h`` is the inverse link.
# Multiple choices exist for ``h`` but we will focus mostly on ``h(x)=\sigma(x)=(1+\exp(-x))^{-1}``, i.e. the logistic function.
# ### Run sparse classification with an increasing number of inducing points
Ms = [4, 8, 16, 32, 64] # Number of inducing points
models = Vector{AbstractGPModel}(undef, length(Ms) + 1)
kernel = with_lengthscale(SqExponentialKernel(), 1.0) # We create a standard kernel with lengthscale 1
for (i, num_inducing) in enumerate(Ms)
    @info "Training with $(num_inducing) points"
    m = SVGP(
        kernel,
        LogisticLikelihood(),
        AnalyticVI(),
        inducingpoints(KmeansAlg(num_inducing), X); # Z is selected via the kmeans algorithm
        optimiser=false, # We keep the kernel parameters fixed
        Zoptimiser=false, # We keep the inducing points locations fixed
    )
    @time train!(m, X_train, y_train, 5) # We train the model on the training data for 5 iterations
    models[i] = m # And store the model
end
# ### We initiliaze and train the non sparse model
@info "Running full model"
mfull = VGP(X_train, y_train, kernel, LogisticLikelihood(), AnalyticVI(); optimiser=false)
@time train!(mfull, 5)
models[end] = mfull

# ## Prediction visualization
# ### We create a prediction and plot function on a grid
function compute_grid(model, n_grid=50)
    mins = [-3.25, -2.85]
    maxs = [3.65, 3.4]
    x_lin = range(mins[1], maxs[1]; length=n_grid)
    y_lin = range(mins[2], maxs[2]; length=n_grid)
    x_grid = Iterators.product(x_lin, y_lin)
    y_grid, _ = proba_y(model, vec(collect.(x_grid)))
    return y_grid, x_lin, y_lin
end

function plot_model(model, X, Y, title=nothing; size=(300, 500))
    n_grid = 50
    y_pred, x_lin, y_lin = compute_grid(model, n_grid)
    title = if isnothing(title)
        (model isa SVGP ? "M = $(AGP.dim(model[1]))" : "full")
    else
        title
    end
    p = plot_data(X, Y; size=size)
    Plots.contour!(
        p,
        x_lin,
        y_lin,
        reshape(y_pred, n_grid, n_grid)';
        cbar=false,
        levels=[0.5],
        fill=false,
        color=:black,
        linewidth=2.0,
        title=title,
    )
    if model isa SVGP
        Plots.scatter!(
            p, eachrow(hcat(AGP.Zview(model[1])...))...; msize=2.0, color="black", lab=""
        )
    end
    return p
end;

# ### Now run the prediction for every model and visualize the differences
Plots.plot(
    plot_model.(models, Ref(X), Ref(Y))...; layout=(1, length(models)), size=(1000, 200)
)

# ## Bayesian SVM vs Logistic link
# ### We now create another full model but with the Bayesian SVM link
@info "Running model with Bayesian SVM Likelihood"
mbsvm = VGP(X_train, y_train, kernel, BayesianSVM(), AnalyticVI(); optimiser=false)
@time train!(mbsvm, 5)
# ### And compare it with the Logistic likelihood
Plots.plot(
    plot_model.(
        [models[end], mbsvm], Ref(X), Ref(Y), ["Logistic", "BSVM"]; size=(500, 250)
    )...;
    layout=(1, 2),
)
