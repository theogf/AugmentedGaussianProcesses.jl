# @testset "MOVGP" begin
N, d = 20, 2
β = 3.0
likelihoods = Vector{AGP.AbstractLikelihood}([LogisticLikelihood(), LaplaceLikelihood(β)])
k = SqExponentialKernel() ∘ ScaleTransform(10.0)
X, f = generate_f(N, d, k)
X, f2 = generate_f(N, d, k; X=X)
y_logistic = 2 * f .> 0
y_laplace = -f2 + rand(Laplace(β), N)
floattypes = [Float64]
model = MOVGP(RowVecs(X), [y_logistic, y_laplace], k, likelihoods, AnalyticVI(), 2)
train!(model)
# end
