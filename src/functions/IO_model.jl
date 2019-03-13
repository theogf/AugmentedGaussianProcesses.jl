# module IO_model
#TODO Rework the whole setup
using JLD2, FileIO
# using AugmentedGaussianProcesses
import AugmentedGaussianProcesses.KernelModule
import AugmentedGaussianProcesses.KernelModule.HyperParametersModule#: HyperParameter, OpenBound, NullBound, Interval
export save_trained_model, save_model,load_trained_model,load_model
#Needs to be redefined
function save_model(filename::String,model::GPModel)
    try
        jldopen(filename,"w") do file
            writedata(file,model)
            writegeneral(file,model)
            writekernel(file,model)
            writeextrabits(file,model)
        end
    catch
        @warn "Writing in the file $filename failed, please enter a new filename or press enter to exit"
        newname = readline(stdin)
        if newname != ""
            save_model(newname,model)
        end
    end
end

function save_trained_model(filename::String,model::GPModel)
    # try
    if typeof(model) <: BatchGPRegression
        @warn "Model not saved, BatchGPRegression (non sparse) does not require training, it does not make sense to save it"
        return nothing
    end
        jldopen(filename,"w") do file
            writegeneral(file,model)
            writekernel(file,model)
        end
    # catch
        # @warn "Writing in the file $filename failed, please enter a new filename or press enter to exit"
        # newname = readline(stdin)
        # if newname != ""
            # save_trained_model(newname,model)
        # end
    # end
end

function load_model(filename::String)
    # try
        jldopen(filename,"r") do file
            writegeneral(file,model)
            writekernel(file,model)
        end
    # catch
    #     @warn "Reading the file $filename failed, please enter a new filename or press enter to exit"
    #     newname = readline(stdin)
    #     if newname != ""
    #         read_model(newname,model)
    #     end
    # end
end

function load_trained_model(filename::String)
    # try
        jldopen(filename,"r") do file
            sparse = read(file,"Sparse")
            modeltype = read(file,"ModelType")
            model = create_model(sparse,modeltype)
            readgeneral(file,model)
            readkernel(file,model)
            model.Trained=true
            return model
        end
    # catch
    #     @warn "Reading the file $filename failed, please enter a new filename or press enter to exit"
    #     newname = readline(stdin)
    #     if newname != ""
    #         read_model(newname,model)
    #     end
    # end
end


function writedata(file::JLD2.JLDFile,model::GPModel)
    write(file,"X",model.X)
    write(file,"y",model.y)
end

function writegeneral(file::JLD2.JLDFile,model::GPModel)
    write(file,"ModelType",model.ModelType)
    write(file,"Sparse",typeof(model)<:SparseModel)
    write(file,"Name",model.Name)
    write(file,"nSamples",model.nSamples)
    write(file,"nDim",model.nDim)
    write(file,"nFeatures",model.nFeatures)
    write(file,"TopMatrix",model.TopMatrixForPrediction)
    write(file,"DownMatrix",model.DownMatrixForPrediction)
    write(file,"MatricesPrecomputed",model.MatricesPrecomputed)
    write(file,"μ",model.μ)
    write(file,"Σ",model.Σ)
    writemodelspecific(file,model)
end


function writemodelspecific(file::JLD2.JLDFile,model::FullBatchModel)
    write(file,"X",model.X)
end

function writemodelspecific(file::JLD2.JLDFile,model::SparseModel)
end
function writekernel(file::JLD2.JLDFile,model::GPModel)
    # write(file,"kernel",model.kernel)
    write(file,"iskernelARD",isARD(model.kernel))
    write(file,"kernelname",model.kernel.fields.name)
    write(file,"kernelvariance",getvariance(model.kernel))
    write(file,"kernellengthscales",getlengthscales(model.kernel))
    writematrices(file,model)
end

function writematrices(file::JLD2.JLDFile,model::FullBatchModel)
    write(file,"invK",model.invK)
end

function writematrices(file::JLD2.JLDFile,model::SparseModel)
    write(file,"invKmm",model.invKmm)
    write(file,"indPoints",model.inducingPoints)
end

function writeextrabits(file,model::GPModel)
end

function create_model(sparse::Bool,modeltype::AugmentedGaussianProcesses.GPModelType)
    if modeltype == AugmentedGaussianProcesses.GPModelType(1) #BSVM
        return model = sparse ? SparseBSVM() : BatchBSVM()
    elseif modeltype == AugmentedGaussianProcesses.GPModelType(2) #XGPC
        return model = sparse ? SparseXGPC() : BatchXGPC()
    elseif modeltype == AugmentedGaussianProcesses.GPModelType(3) #Regression
        return model = sparse ? SparseGPRegression() :
        BatchGPRegression()
    elseif modeltype == AugmentedGaussianProcesses.GPModelType(4) #StudentT
        return model = sparse ? SparseStudentT() : BatchStudentT()
    else
        @warn "Model writing for this model is not implemented yet"
    end
end

function readgeneral(file::JLD2.JLDFile,model::GPModel)
    model.nSamples = read(file,"nSamples")
    model.nDim = read(file,"nDim")
    model.nFeatures = read(file,"nFeatures")
    model.MatricesPrecomputed = read(file,"MatricesPrecomputed")
    if model.MatricesPrecomputed
        model.TopMatrixForPrediction = read(file,"TopMatrix")
        model.DownMatrixForPrediction = read(file,"DownMatrix")
    else
        model.TopMatrixForPrediction = 0
        model.DownMatrixForPrediction = 0
    end
    model.μ = read(file,"μ")
    model.Σ = read(file,"Σ")
    readmodelspecific(file,model)
end

function readmodelspecific(file::JLD2.JLDFile,model::FullBatchModel)
    model.X = read(file,"X")
end

function readmodelspecific(file::JLD2.JLDFile,model::SparseModel)
end

function readkernel(file::JLD2.JLDFile,model::GPModel)
    # model.kernel
    kernelname = read(file,"kernelname")
    ARD = read(file,"iskernelARD")
    variance = read(file,"kernelvariance")
    lengthscales = read(file,"kernellengthscales")
    if kernelname == "Radial Basis"
        model.kernel = RBFKernel(lengthscales,variance=variance)
    else
        @warn "Loading for this kernel is not implemented yet"
    end
    readmatrices(file,model)
end

function readmatrices(file::JLD2.JLDFile,model::FullBatchModel)
    model.invK = read(file,"invK")
end

function readmatrices(file::JLD2.JLDFile,model::SparseModel)
    model.invKmm = read(file,"invKmm")
    model.inducingPoints = read(file,"indPoints")
end
# end#End module
