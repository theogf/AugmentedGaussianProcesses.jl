using Documenter, GradDescent

makedocs()

deploydocs(
    deps = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/jacobcvt12/GradDescent.jl.git",
    julia = "0.6"
)
