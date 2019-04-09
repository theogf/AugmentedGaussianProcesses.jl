using Documenter, AugmentedGaussianProcesses

makedocs(modules=[AugmentedGaussianProcesses],
         sitename= "AugmentedGaussianProcesses.jl",
         pages = [
         "Home"=>"index.md",
         "Background"=>"background.md",
         "User Guide"=>"userguide.md",
         "Kernels"=>"kernel.md",
         "Examples"=>"examples.md",
         "Other Julia GP Packages"=>"comparison.md",
         "API"=>"api.md"
         ]
         )

deploydocs(
    deps = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/theogf/AugmentedGaussianProcesses.jl.git",
    target = "build"
)
