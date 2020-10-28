using Documenter, Literate
using AugmentedGaussianProcesses

# Create the notebooks

# Regression
Literate.markdown(joinpath(@__DIR__, "examples", "gpregression.jl"),
                    joinpath(@__DIR__, "src", "examples");
                    documenter = true,
                )
Literate.notebook(joinpath(@__DIR__, "examples", "gpregression.jl"),
                    joinpath(@__DIR__, "notebooks")
                )

# Classification
Literate.markdown(joinpath(@__DIR__, "examples", "gpclassification.jl"),
                    joinpath(@__DIR__, "src", "examples");
                    documenter = true,
                ) 
Literate.notebook(joinpath(@__DIR__, "examples", "gpclassification.jl"),
                joinpath(@__DIR__, "src", "examples")
            ) 

# Multi-Class Classification
Literate.markdown(joinpath(@__DIR__, "examples", "multiclassgp.jl"),
                    joinpath(@__DIR__, "src", "examples");
                    documenter = true,
                ) 
Literate.notebook(joinpath(@__DIR__, "examples", "multiclassgp.jl"),
                joinpath(@__DIR__, "src", "examples")
            ) 

# Online GP
Literate.markdown(joinpath(@__DIR__, "examples", "onlinegp.jl"),
                    joinpath(@__DIR__, "src", "examples");
                    documenter = true,
                ) 
Literate.notebook(joinpath(@__DIR__, "examples", "onlinegp.jl"),
                joinpath(@__DIR__, "src", "examples")
            ) 

# Make the docs

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
         "Examples"=>[
             "GP Regression" => "examples/gpregression.md",
             "GP Classification" => "examples/gpclassification.md",
             "Multi-Class GP" => "examples/multiclassgp.md",
             "Online GP" => "examples/onlinegp.md",
            ],
         "Julia GP Packages"=>"comparison.md",
         "API"=>"api.md"
         ]
         )

# Deploy the docs

deploydocs(
    deps = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/theogf/AugmentedGaussianProcesses.jl.git",
    target = "build"
)
