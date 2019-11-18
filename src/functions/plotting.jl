# using AbstractPlotting
using RecipesBase


@recipe function f(gp::AbstractGP,x::AbstractVector;showX=false,nSigma=2.0)
    @assert showX isa Bool "showX should be a boolean"
    @assert nSigma isa Real "nSigma should be a Real"
    X = reshape(x,:,1)
    ch1 = Int('f')
    f,sig_f = predict_f(gp,X,covf=true)
    legendfontsize --> 15.0
    if showX
        @series begin
            seriestype --> :scatter
            markersize --> 4.0
            label --> "Data"
            vec(gp.X),gp.y
        end
    end
    if gp.nLatent == 1
        @series begin
            ribbon := nSigma*sqrt.(sig_f)
            fillalpha --> 0.3
            width --> 3.0
            label --> "$(Char(ch1))"
            x,f
        end
    else
        for t in 1:gp.nLatent
            @series begin
                ribbon := nSigma*sqrt.(sig_f[t])
                fillalpha --> 0.3
                width --> 3.0
                label --> "$(Char(ch1-1+t))"
                x,f[t]
            end
        end
    end
end

@recipe function f(gps::MOSVGP,x::AbstractVector;showX::Bool=false,nSigma::Real=2.0)
    @assert showX isa Bool "showX should be a boolean"
    @assert nSigma isa Real "nSigma should be a Real"
    X = reshape(x,:,1)
    nTasks = gps.nTask
    f,sig_f = predict_f(gps,X,covf=true)
    ch1 = Int('f')
    legend := true
    link := :both
    layout := nTasks
    legendfontsize --> 15.0
    for i in 1:nTasks
        if showX
            @series begin
                seriestype --> :scatter
                markersize --> 4.0
                label --> "Data"
                subplot := i
                vec(gps.X),gps.y[i]
            end
        end
        for j in 1:gps.nf_per_task[i]
            @series begin
                ribbon := nSigma*sqrt.(sig_f[i][j])
                fillalpha --> 0.3
                width --> 3.0
                title --> "Task $i"
                label -->  "$(Char(ch1-1+j))"
                subplot := i
                x,f[i][j]
            end
        end
    end
end
