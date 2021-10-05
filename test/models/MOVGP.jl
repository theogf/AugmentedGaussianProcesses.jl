@testset "MOVGP" begin
    N, d = 20, 2
    β = 2.0
    likelihoods = Vector{AGP.AbstractLikelihood}([
        LogisticLikelihood(), LaplaceLikelihood(β)
    ])
    k = SqExponentialKernel() ∘ ScaleTransform(10.0)
    X, f = generate_f(N, d, k)
    X, f2 = generate_f(N, d, k; X=X)
    y_logistic = f .> 0
    y_laplace = f2 + rand(Laplace(β), N)
    ys = [y_logistic, y_laplace]
    floattypes = [Float64]
    model = MOVGP(RowVecs(X), ys, k, likelihoods, AnalyticVI(), 2)

    train!(model, 10)
    predict_y(model, X)
    proba_y(model, X)
end
