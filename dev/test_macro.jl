using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using LinearAlgebra, Distributions, Plots
using BenchmarkTools
b = 2.0
C()=1/(2b)
g(y) = 0.0
α(y) = y^2
β(y) = 2*y
γ(y) = 1.0
φ(r) = exp(-sqrt(r)/b)
∇φ(r) = -exp(-sqrt(r)/b)/(2*b*sqrt(r))
ll(y,x) = 0.5*exp(0.5*y*x)*sech(0.5*sqrt(x^2))

##
formula = :(p(y|x)=exp(0.5*y*x)*sech(0.5*sqrt(y^2 - 2*y*x + x^2)))
# formula = :(p(y,x)=exp(0.5*y*x)*sech(0.5*sqrt(0.0 - 0.0*x + x^2)))
formula.args[2].args[2].args

topargs = formula.args[2].args[2].args
if topargs[1] == :*
    @show topargs[1]
    global CC = copy(topargs[2])
    popfirst!(topargs)
    popfirst!(topargs)
else
    global CC = :0
end
args2 = topargs[1]
if args2.args[1] == :exp
    gargs = args2.args[2]
    if gargs.args[1] == :*
        deleteat!(gargs.args,findfirst(isequal(:x),gargs.args))
    else
        @error "BAD BAD BAD"
    end
    global GG = copy(gargs)
    popfirst!(topargs)
else
    global GG = :0
end
args3 = topargs[1]
seq = string(args3)
findh= r"\([^(]*\-.*x.*\+.*x \^ 2[^)]*"
b = occursin(findh,seq)
m = match(findh,seq).match
alphar = r"[^(][^-]*"
malpha = match(alphar,m).match[1:end-1]
betar = r"- [^x]*x"
mbeta = match(betar,m).match[3:end]
gammar = r"\+ [^x]*x \^ 2"
mgamma = match(gammar,m).match[3:end]

AA = :($malpha)
BB = :($(mbeta[1:end-1]))
GG = :($(mgamma == "x ^ 2" ? "1.0" : mgamma[1:end-5]))

loc = findfirst(findh,seq)
newseq = seq[1:loc[1]-1]*"r"*seq[(loc[end]+1):end]
PHI = :($newseq)
##
f_lap = :(p(y|x)=0.5/β * exp(- sqrt((y^2 - 2*y*f + f^2))/β))
display.(AGP.@augmodel NewSuperLaplace Regression (p(y|x)=0.5/β * exp( y * x) * exp(- sqrt((sqrt(y^2) - exp(4.0*y)*x + 2.0*x^2))/β)) β)
pdfstring = "(0.5 / β) * exp(2*y*x) * exp(-(sqrt((y ^ 2 - 2.0 * y * x) + 1.0*x ^ 2)) / β)"

Gstring
##
PhiHstring = match(Regex("(?<=$(AGP.correct_parenthesis(Gstringfull))x\\) \\* ).*"),pdfstring).match
Hstring = match(r"(?<=\().+\-.*x.*\+.+x \^ 2(?=\))",PhiHstring).match
locx = findfirst("x ^ 2",PhiHstring)
count_parenthesis = 1
locf = locx[1]
while count_parenthesis != 0
    global locf = locf - 1
    println(PhiHstring[locf])
    if PhiHstring[locf] == ')'
        global count_parenthesis += 1
    elseif PhiHstring[locf] == '('
        global count_parenthesis -= 1
    end
end
Hstring = PhiHstring[(locf+1):locx[end]]

alphar = r"[^(][^-]*"
alpha_string = match(alphar,Hstring).match[1:end-1]
# betar = r"(?>=- )[^x]+(?= * x)"
betar = r"(?<=- )[^x]+(?= * x)"
mbeta = match(betar,Hstring).match
while last(mbeta) == ' ' || last(mbeta) == '*'
    global mbeta = mbeta[1:end-1]
end
mbeta
gammar = r"(?<=\+ )[^x]*(?=x \^ 2)"
mgamma = match(gammar,m).match == "" ? "1.0" : match(gammar,m).match
##
findnext(isequal(')'),PhiHstring,locx[end])
code = Meta.parse(PhiHstring)
code.args

S = code.args[2].args[2].args[2].args[2].args[2].args
S = code.args[2].args[2].args[2].args[2].args
for args in S.args
    if args == :(x ^ 2)
        @show "BLAH"
    end
end
S = string(code.args[2].args[2])
Hstring = match(r"(?<=\().*x \^ 2.*\-.*x.*\+.*(?=\))",S)

##

txt = AGP.@augmodel("NewLaplace","Regression",C,g,α,β,γ,φ,∇φ)

# NewLaplaceLikelihood() |> display
N = 500
σ = 1.0
X = sort(rand(N,1),dims=1)
K = kernelmatrix(X,RBFKernel(0.1))+1e-4*I
L = Matrix(cholesky(K).L)
y_true = rand(MvNormal(K))
y = y_true+randn(length(y_true))*2
p = scatter(X[:],y,lab="data")
NewLaplaceLikelihood() |> display
m = VGP(X,y,RBFKernel(0.5),NewLaplaceLikelihood(),AnalyticVI(),optimizer=false)
train!(m,iterations=100)
y_p, sig_p = proba_y(m,collect(0:0.01:1))

m2 = VGP(X,y,RBFKernel(0.5),LaplaceLikelihood(b),AnalyticVI(),optimizer=false)
train!(m2,iterations=100)
y_p2, sig_p2 = proba_y(m2,collect(0:0.01:1))

plot!(X,y_true,lab="truth")

plot!(collect(0:0.01:1),y_p,lab="Auto Laplace")
plot!(collect(0:0.01:1),y_p2,lab="Classic Laplace") |> display


# @btime train!($m,iterations=1)
# @btime train!($m2,iterations=1)

###
