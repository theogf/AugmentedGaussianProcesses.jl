using Documenter, AugmentedGaussianProcesses

makedocs(modules=[AugmentedGaussianProcesses],
         format = Documenter.Writers.HTMLWriter.HTML(
         assets = ["assets/icon.ico"],
         analytics="UA-129106538-2"),
         sitename= "AugmentedGaussianProcesses",
         authors="Theo Galy-Fajou",
         pages = [
         "Home"=>"index.md",
         "Background"=>"background.md",
         "User Guide"=>"userguide.md",
         "Kernels"=>"kernel.md",
         "Examples"=>"examples.md",
         "Julia GP Packages"=>"comparison.md",
         "API"=>"api.md"
         ]
         )

deploydocs(
    deps = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/theogf/AugmentedGaussianProcesses.jl.git",
    target = "build"
)
