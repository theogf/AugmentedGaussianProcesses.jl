using Documenter, AugmentedGaussianProcesses

makedocs(modules=[AugmentedGaussianProcesses])

deploydocs(
    deps = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/theogf/AugmentedGaussianProcesses.jl.git",
    julia = "1.0"
)
