using AugmentedGaussianProcesses
using Plots
using Distributions, LinearAlgebra
pyplot()

X = range(0,10, length=100)
k = SqExponentialKernel()
K = kernelmatrix(k,X) + 1e-5I
fs = [rand(MvNormal(K)) for _ in 1:2]

lab12 = ["¹" "²"]
##
p = 3
xs = [hcat([f[i:end-p+i-1] for i in 1:p]...) for f in fs]
_fs = [f[p+1:end] for f in fs]
m1 = AGP.MOVGP(
    xs,
    _fs,
    transform(SqExponentialKernel(), 5.0*ones(p)),
    GaussianLikelihood(),
    AnalyticVI(),
    2,
    verbose = 1,
)
# m.A[1][1] = [1.0, 0.0]
# m.A[2][1] = [0.0, 1.0]
train!(m1, 1000)
pred = first.(first(predict_f(m1, xs)))
##
nplus = 300
Xplus = range(10+Float64(X.step), length = nplus, step = Float64(X.step))
arpred = AGP.predict_ar(m1, p, nplus, y_past = _fs)
default(lw = 3.0, legendfontsize = 15.0)
pl = plot(X, fs, lab = "y" .* lab12, lw = 3.0)
plot!(X[p+1:end], pred, lab = "f" .* lab12, lw = 5.0, linestyle = :dot)
plot!(Xplus, arpred, lab = "fₜ" .* lab12, lw = 3.0)
plot!(
    X[p:end],
    collect(AGP.get_μ(m1)),
    lw = 2.0,
    lab = "f̃" .* lab12,
    linestyle = :dash,
) |> display
gatherA = hcat(first.(m1.A)...)
savefig("Wow_noind.png")

##
m2 = AGP.MOSVGP(
    xs,
    _fs,
    transform(SqExponentialKernel(), 5.0*ones(p)),
    GaussianLikelihood(),
    AnalyticVI(),
    2,
    UniformSampling(20),
)
train!(m2, 1000)
pred = first.(first(predict_f(m2, xs)))

##
nplus = 300
Xplus = range(10+Float64(X.step), length = nplus, step = Float64(X.step))
arpred = AGP.predict_ar(m2, p, nplus, y_past = _fs)
default(lw = 3.0, legendfontsize = 15.0)
pl = plot(X, fs, lab = "y" .* lab12, lw = 3.0)
plot!(X[p+1:end], pred, lab = "f" .* lab12, lw = 5.0, linestyle = :dot)
plot!(Xplus, arpred, lab = "fₜ" .* lab12, lw = 3.0) |> display
# plot!(
#     X[p:end],
#     collect(AGP.get_μ(m)),
#     lw = 2.0,
#     lab = "f̃" .* lab12,
#     linestyle = :dash,
# ) |> display
gatherA = hcat(first.(m2.A)...)
savefig("Wow_withind.png")
##
##

X = range(0,1, length=100)
y1 = -sinpi.(10*(X.+1))./(2X.+1) - X.^4 + 0.05*randn(length(X))
y2 = cos.(y1).^2 + sin.(3*X) + 0.05*randn(length(X))
y3 = y2.*(y1.^2) + 3*X + 0.05*randn(length(X))
ys = [y1,y2,y3]

p = 3
xs = [hcat([y[i:end-p+i-1] for i in 1:p]...) for y in ys]
_ys = [y[p+1:end] for y in ys]
m1 = AGP.MOVGP(
    xs,
    _ys,
    transform(SqExponentialKernel(), 0.1),
    GaussianLikelihood(),
    AnalyticVI(),
    3,
    optimiser = true,
    verbose=2
)
# m.A[1][1] = [1.0, 0.0, 0.0]
# m.A[2][1] = [0.0, 1.0, 0.0]
# m.A[3][1] = [0.0, 0.0, 1.0]

train!(m1, 2000)
pred = first.(first(predict_f(m1, xs)))
gatherA = hcat(first.(m1.A)...)
##



##
anim = Animation()
nplus = 300
nplusmax = 300
# @progress for nplus in 1:10:nplusmax
    xplus = range(1+Float64(X.step),length=nplus,step=Float64(X.step))
    pred_ar = AGP.predict_ar(m1, p, nplus, y_past = _ys)
    pl = scatter(X,ys,lab="",xlim=(0,1+Float64(X.step)*nplusmax), color=[:blue :red :green],)
    plot!(X[p+1:end],pred, color=[:blue :red :green],lab="")
    plot!(X[p+1:end],collect(AGP.get_μ(m1)), color=[:blue :red :green], lab="", linestyle=:dot, lw=2.0)
    plot!(xplus,pred_ar, color=[:blue :red :green],lab="") #|> display
    # frame(anim)
# end
# gif(anim,"wowgif.gif",fps=10)
display(pl)
savefig("Wow2.png")

##
nplus = 100
xplus = range(1+Float64(X.step),length=nplus,step=Float64(X.step))
pl = scatter(X,ys,color=[:blue :red :green],lab="")
plot!(X[p+1:end],pred, color=[:blue :red :green], lab="")
plot!(X[p+1:end],collect(AGP.get_μ(m)), color=[:blue :red :green], lab="", linestyle=:dot, lw=2.0)
nSamples= 50
@progress for i in 1:nSamples
    plot!(pl, xplus, AGP.sample_ar(m, p, nplus, y_past = _ys), color=[:blue :red :green], lab="", lw=1.0, alpha=0.2)
end
pl |> display
savefig("Wow3.png")

# AGP._predict_f(m, AGP.Xtest,covf=true)
gatherA = hcat(first.(m.A)...)
# plot(AGP.get_μ(m)[3])
