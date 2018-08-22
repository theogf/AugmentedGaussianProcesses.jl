using Documenter, OMGP

makedocs(modules=[OMGP])

deploydocs(
    repo = "github.com/theogf/OMGP.jl.git",
    julia = "0.7"
)
