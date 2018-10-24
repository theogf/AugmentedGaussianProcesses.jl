using Documenter, OMGP

makedocs(modules=[OMGP])

deploydocs(
    repo = "github.com/theogf/OMGP.jl.git",
    julia = "1.0"
)
