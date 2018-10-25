using Documenter, OMGP

makedocs(modules=[OMGP])

deploydocs(
    deps = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/theogf/OMGP.jl.git",
    julia = "1.0"
)
