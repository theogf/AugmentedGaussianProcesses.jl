using Documenter, AugmentedGaussianProcesses

makedocs(modules=[AugmentedGaussianProcesses],
         format = Documenter.HTML(
         assets = ["assets/icon.ico"]),
         sitename= "AugmentedGaussianProcesses",
         authors="Theo Galy-Fajou",
         analytics="UA-129106538-2",
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
    target = "build"
)
