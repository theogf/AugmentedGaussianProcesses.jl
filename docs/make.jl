using Documenter, AugmentedGaussianProcesses

makedocs(modules=[AugmentedGaussianProcesses],
         format = :html,
         sitename= "AugmentedGaussianProcesses.jl",
         pages = [
         "Home"=>"index.md",
         "Background"=>"background.md",
         "User Guide"=>"userguide.md",
         "Kernels"=>"kernel.md",
         "Examples"=>"examples.md",
         "API"=>"api.md"
         ]
         )

deploydocs(
    deps = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/theogf/AugmentedGaussianProcesses.jl.git",
)
