using PyPlot
using Formatting
type DataPlot
  x
  y
  color
  linestyle
  linewidth
  marker
  x_err
  y_err
  time
  time_err
  p
  p_time
  function DataPlot(x,y)
    color = "blue"
    linestyle = "--"
    return new(x,y,color,linestyle)
  end
  function DataPlot(x,y,color,linestyle,linewidth,marker;x_err=[],y_err=[],time=[],time_err=[])
    return new(x,y,color,linestyle,linewidth,marker,x_err,y_err,time,time_err)
  end
end
NC =  Dict("LogReg"=>"Linear Model","GPC"=>"SVGPC","SPGGPC"=>"X-GPC","Accuracy"=>"Avg. Test Error","MedianL"=>"Avg. Median Neg. Test Log-Likelihood")
colors=Dict("SVGPC"=>"b","SXGPC"=>"r","LogReg"=>"y")
linestyles=Dict(8=>":",16=>"--",64=>"-.",128=>"-")
markers=Dict(8=>"",16=>"o",64=>"x",128=>"+")
# linestyles=Dict(4=>"-",8=>":",10=>"-",16=>"-.",32=>"--",50=>":",64=>"-.",100=>"-.",128=>"-",200=>"--",256=>"--")
metrics = Dict("Accuracy"=>3,"MeanL"=>5,"MedianL"=>7,"ELBO"=>9)
LogRegConvert = Dict(1=>1,3=>3,4=>4,5=>13,6=>14,7=>11,8=>12)

gwidth = 2.0
gmarkersize= 5.0
function DataConversion(array,name)
    if name == "Accuracy"
        return 1-array
    elseif name == "MedianL"
        return -array
    elseif name == "MeanL"
        return -array
    end
end

function InducingPointsComparison(metric,MPoints=[8,16,64,256];step=1)
    dataset="Shuttle"
    figure("Comparison of inducing points accuracy and time",figsize=(16,8)); clf();
    p = Dict("GPC"=>Array{Any,1}(),"SPGGPC"=>Array{Any,1}())
    lab = Dict("GPC"=>Array{Any,1}(),"SPGGPC"=>Array{Any,1}())
    for M in MPoints
        Results = Dict{String,Any}()
        Results["GPC"] = readdlm("../final_cluster_results/InducingPointsExperiment/data_M$M/ConvergenceExperimentExperiment_$(dataset)Dataset/Results_GPC.txt")
        Results["SPGGPC"] = readdlm("../final_cluster_results/InducingPointsExperiment/data_M$M/ConvergenceExperimentExperiment_$(dataset)Dataset/Results_SPGGPC.txt")
        for (name,res) in Results
            res[:,metrics[metric]] = DataConversion(res[:,metrics[metric]],metric)
            push!(lab[name],NC[name]*" M=$M")
            new_p,=semilogx(res[1:step:end,1],res[1:step:end,metrics[metric]],markersize=gmarkersize,color=colors[name],marker=markers[M],linewidth=gwidth,linestyle=linestyles[M],label=NC[name]*" M=$M")
            push!(p[name],new_p)
        end
    end


    xlabel("Training Time in Seconds",fontsize=15.0)
    ylabel(NC[metric],fontsize=15.0)
    title("Shuttle",fontsize=18.0,fontweight="semibold")

    legend([p["SPGGPC"];p["GPC"]],[lab["SPGGPC"];lab["GPC"]])
    xlim([0.3,4500])
    tight_layout()
    savefig("../../Plots/$(dataset)InducingPointsPlot.png")
end

function PlotAll()
    file_list = readdlm("file_list_finished")
    for file in file_list
        PlotMetricvsTime(file,"Final",time=true,writing=true,corrections=true)
    end
end
function PlotMetricvsTime(dataset,metric;final=false,AT=true,time=true,writing=false,corrections=true)
    Results = Dict{String,Any}();
    println("Working on dataset $dataset")
    # colors=Dict("GPC"=>"b","SPGGPC"=>"r","LogReg"=>"y")
    time_line = Dict("SVGPC"=>[1:1:99;100:10:999;1000:100:9999;10000:1000:20000],"SXGPC"=>[1:1:99;100:10:999;1000:100:20000])
    p = Dict{String,Any}()

    FinalMetrics = ["MedianL","Accuracy"]
    # NC =  Dict("LogReg"=>"Linear Model","GPC"=>"SVGPC","SPGGPC"=>"X-GPC","Accuracy"=>"Avg. Test Error","MedianL"=>"Avg. Median Neg. Test Log likelihood")
    Results["SVGPC"] = readdlm("../$(final?"final_results":"cluster_results")/ConvergenceExperiment$(AT?"_AT":"")/"*dataset*"Dataset/Results_SVGPC.txt")
    Results["SXGPC"] = readdlm("../$(final?"final_results":"cluster_results")/ConvergenceExperiment$(AT?"_AT":"")/"*dataset*"Dataset/Results_SXGPC.txt")
    # Results["LogReg"] = readdlm("../final_cluster_results/ConvergenceExperiment/ConvergenceExperiment_"*dataset*"Dataset/Results_LogReg.txt")
    maxx = Results["SVGPC"][end,1]
    minx = Results["SXGPC"][1,1]
    if metric != "Final"
        figure("Convergence on dataset "*dataset*" ",figsize=(16,9));clf();
    else
        figure("Convergence on dataset "*dataset*" ",figsize=(16,4.7));clf();
    end
    step=1
    if corrections
        if dataset == "aXa"
            #Divide acc stderr by 2
            Results["SXGPC"][:,4] *= 0.5;
            Results["SVGPC"][:,4] *= 0.5;
        elseif dataset == "Bank_marketing"
            #Divide acc stderr by 2
            Results["SXGPC"][:,4] *= 0.5;
            Results["SVGPC"][:,4] *= 0.5
        elseif dataset == "Electricity"
            #Divide acc stderr by 2
            Results["SXGPC"][:,4] *= 0.5;
            Results["SVGPC"][:,4] *= 0.5
        elseif dataset == "German"
            Results["SXGPC"][:,4] *= 0.5;
            Results["SVGPC"][:,4] *= 0.5
        end
    end
    if time
        time_line = Dict("SVGPC"=>Results["SVGPC"][:,1],"SXGPC"=>Results["SXGPC"][:,1])
    # elseif dataset == "German" || dataset == "Diabetis"
    #     time_line = Dict("SVGPC"=>[1:1:99;100:10:999;1000:100:9999;10000:1000:50000],"SXGPC"=>collect(1:length(Results["SXGPC"][:,1])))
    # elseif dataset == "Covtype"
    #     time_line = Dict("SVGPC"=>[1:1:99;100:10:999;1000:100:9999;10000:1000:10000],"SXGPC"=>[1:1:99;100:10:999;1000:100:9999;10000:1000:50000])
    # elseif  dataset == "SUSY" || dataset == "HIGGS"
    #     time_line = Dict("SVGPC"=>[1:1:99;100:10:999;1000:100:9999;10000:1000:100000],"SXGPC"=>[1:1:99;100:10:999;1000:100:9999;10000:1000:50000])
    end
    if metric == "Final"
        iter=1
        giter = 2
        for (mname,mmetric) in metrics
            if in(mname,FinalMetrics)
                subplot(1,2,giter)
                for name in sort(collect(keys(Results)),rev=true)
                    if name != "LogReg"
                        Results[name][:,mmetric] = DataConversion(Results[name][:,mmetric],mname)
                        x = [time_line[name][1:step:end];maxx]
                        my = [Results[name][1:step:end,mmetric];Results[name][end,mmetric]]
                        sy = [Results[name][1:step:end,mmetric+1];Results[name][end,mmetric+1]]/sqrt(10)
                        new_p, = semilogx(x,my,color=colors[name],linewidth=gwidth,label=NC[name])
                        fill_between(x,my-sy,my+sy,alpha=0.2,facecolor=colors[name])
                        p[name]=new_p
                    else
                        Results[name][:,LogRegConvert[mmetric]] = DataConversion(Results[name][:,LogRegConvert[mmetric]],mname)
                        mean_lg = [Results[name][LogRegConvert[mmetric]],Results[name][LogRegConvert[mmetric]]]
                        std_lg = [Results[name][LogRegConvert[mmetric+1]],Results[name][LogRegConvert[mmetric+1]]]/sqrt(10)
                        new_p, = semilogx([Results[name][1],maxx],mean_lg,color=colors[name],linewidth=gwidth,label=NC[name])
                        p[name]=new_p
                        # fill_between([results[1],maxfigure],mean_lg-std_lg,mean_lg+std_lg,alpha=0.2,facecolor=colors[name])
                    end
                end
                if time
                    xlabel("Training Time in Seconds",fontsize=15.0)
                    xlim([0.5*minx,1.5*maxx])
                else
                    xlabel("Iterations")
                end
                title(dataset,fontsize=18.0,fontweight="semibold")
                ylabel(NC[mname],fontsize=15.0)
                legend([p["SXGPC"];p["SVGPC"];p["LogReg"]],[NC["SXGPC"];NC["SVGPC"];NC["LogReg"]])
                giter-=1
            end
        end
    elseif metric != "All"
        println(name)
        for (name,results) in Results
            semilogx(time_line[name][1:step:end],results[1:step:end,metrics[metric]],color=colors[name],label=name)
            fill_between(time_line[name][1:step:end],results[1:step:end,metrics[metric]]-results[1:step:end,metrics[metric]+1]/sqrt(10),results[1:step:end,metrics[metric]]+results[1:step:end,metrics[metric]+1]/sqrt(10),alpha=0.2,facecolor=colors[name])
        end
        if time
            xlabel("Time [s]")
        else
            xlabel("Iterations")
        end
        ylabel(metric)
        legend(loc=4)
    else
        giter = 1
        for (mname,mmetric) in metrics

            subplot(2,2,giter)
            for (name,results) in Results
                semilogx(time_line[name][1:step:end],results[1:step:end,mmetric],color=colors[name],label=name)
                fill_between(time_line[name][1:step:end],results[1:step:end,mmetric]-results[1:step:end,mmetric+1]/sqrt(10),results[1:step:end,mmetric]+results[1:step:end,mmetric+1]/sqrt(10),alpha=0.2,facecolor=colors[name])
            end
            if time
                xlabel("Time [s]")
            else
                xlabel("Iterations")
            end
            ylabel(mname)
            legend(loc=4)
            giter+=1
        end
    end
    tight_layout()
    if writing
        savefig("../../Plots/"*(metric=="Final"?"Final":"")*"Convergence_vs_"*(time?"time":"iterations")*"_on_"*dataset*".png")
        close()
    end
end

Handpicked = Dict("aXa"=> 562, "Bank_marketing"=>606, "Click_Prediction"=>1255,
                    "Cod-rna"=>1000,"Covtype"=>1000,"Diabetis"=>378,"Electricity"=>888,
                    "German"=>318,"HIGGS"=>10000,"Ijcnn1"=>1016,"Shuttle"=>589,
                    "SUSY"=>10353,"wXa"=>1024)
sizes = Dict("aXa"=>(36974,123),"Bank_marketing"=>(45211,43),"Click_Prediction"=>(399482,12),"Cod-rna"=>(8,343564),"Covtype"=>(581012,54),
                    "Diabetis"=>(768,8),"Electricity"=>(45312,8),"German"=>(1000,20),"HIGGS"=>(11000000,28),"Ijcnn1"=>(141691,22),"Mnist"=>(70000,780),"Poker"=>(1025010,10),
                    "Protein"=>(24837,357),"Shuttle"=>(58000,9),"SUSY"=>(5000000,18),"Vehicle"=>(98528,100),"wXa"=>(34780,300))
DatasetNameCorrection = Dict("aXa"=>"aXa","Bank_marketing"=>"Bank Marketing","Click_Prediction"=>"Click Prediction",
                            "Cod-rna"=>"Cod RNA", "Covtype"=>"Cov Type", "Diabetis"=>"Diabetis","Electricity"=>"Electricity",
                            "German"=>"German","HIGGS"=>"Higgs","Ijcnn1"=>"IJCNN","Shuttle"=>"Shuttle","SUSY"=>"SUSY","Vehicle"=>"Vehicle","wXa"=>"wXa")
function Table()
    dataset_list = readdlm("file_list_finished")
    Methods = ["SPGGPC","GPC"]#,"LogReg"]
    MetricNames = Dict("Error"=>3,"NLL"=>5,"Time"=>1)
    MetricsOrder = ["Error","NLL","Time"]
    full_table = Array{String,1}()
    first_line = "Dataset & n / \$ d \$ & & \\textbf{$(NC[Methods[1]])} & $(NC[Methods[2]]) \\\\\\hline"
    push!(full_table,first_line)

    for dataset in dataset_list
        Res = ConvergenceDetector(dataset,"MedianL",epsilon=1e-3,window=5)
        println("Working on dataset $dataset")
        for metric in MetricsOrder
            new_line =""
            # new_line = new_line*"& "
            if metric == "Error"
                new_line = new_line*"\\multirow{3}{*}{$(DatasetNameCorrection[dataset])} &\\multirow{2}{*}{$(sizes[dataset][1])}"
            elseif  metric == "NLL"
                new_line = new_line*" & \\multirow{2}{*}{$(sizes[dataset][2])}"
            else
                new_line = new_line*" & "
            end
            new_line = new_line*" & $metric";
            for m in Methods
                if metric != "Time"
                    mean_v = format(Res[m][MetricNames[metric]],precision=2); std_v = format(Res[m][MetricNames[metric]+1],precision=2) ;
                else
                    mean_v = format(Res[m][MetricNames[metric]]); std_v = format(Res[m][MetricNames[metric]+1]) ;
                end
                 new_line = new_line*" &  \$ $mean_v \\pm $std_v \$"
            end
            new_line = new_line*"\\\\"
            if metric == "Time"
                new_line = new_line*"\\hline"
            end
            push!(full_table,new_line)
        end
    end
    writedlm("Latex/Table",full_table)
    return full_table
end
function ConvergenceDetector(dataset,metric;epsilon=1e-3,window=5)
    Methods = ["GPC","SPGGPC"]#,"LogReg"]
    ConvResults = Dict{String,Any}()
    figure(2);clf()
    giter=1
    for m in Methods
        Res = readdlm("../final_cluster_results/ConvergenceExperiment/ConvergenceExperiment_"*dataset*"Dataset/Results_"*m*".txt")
        t = Res[:,1]
        iter=2
        if m == "GPC"
            Res[:,3] = DataConversion(Res[:,3],"Accuracy")
            Res[:,7] = DataConversion(Res[:,7],"MedianL")
            while t[iter]<Handpicked[dataset] && iter < length(t)
                iter+=1
            end
            ConvResults[m] = Res[iter,[1,2,3,4,7,8]]
        elseif m =="SPGGPC"
            values = DataConversion(Res[:,metrics[metric]],metric)
            Res[:,3] = DataConversion(Res[:,3],"Accuracy")
            Res[:,7] = DataConversion(Res[:,7],"MedianL")
            converged = false
            conv = abs.(values[2:end]-values[1:end-1])./abs.(values[1:end-1])
            while iter <= length(conv)
                if mean(conv[max(1,iter-window+1):iter])<epsilon
                    # println(mean(conv[max(1,iter-window+1):iter]))
                    ConvResults[m] = Res[iter,[1,2,3,4,7,8]];
                    converged = true
                    break;
                end
                iter+=1
            end
            if !converged
                ConvResults[m] = Res[end,[1,2,3,4,7,8]]
            end
        elseif m == "LogReg"
             r = Res[[1,2,LogRegConvert[3],LogRegConvert[4],LogRegConvert[7],LogRegConvert[8]]]
             r[3] = 1-r[3]; r[5]=-r[5]
             ConvResults[m] =r
        end
    end
    return ConvResults
end


function PlotAccvsm(folder="results/AccvsMExperimentUSPS")
  actual_folder = pwd()
  cd(folder)
  try
    file_list = readdir()
    colors = ["blue","red"]
    linestyles = ["-","-"]
    markers = ["o","o"]
    linewidth=3.0
    plots = Dict{String,DataPlot}()
    for i in 1:length(file_list)
      data = readdlm(file_list[i])
      plots[file_list[i][9:end-4]] = DataPlot(data[:,1],data[:,2],colors[i],linestyles[i],linewidth,markers[i],y_err=data[:,3],time=data[:,4],time_err=data[:,5])
    end
    fig = figure("Accuracy vs # Inducing Points")
    clf()
    for (name,dataplot) in plots
      dataplot.p = errorbar(dataplot.x,1-dataplot.y,yerr=dataplot.y_err,color=dataplot.color,linestyle=dataplot.linestyle,label=name,marker=dataplot.marker,linewidth=dataplot.linewidth)
    end
    ylabel("Prediction Error",fontsize=32.0)
    yticks(fontsize=20.0)
    xticks(fontsize=20.0)
    legend(fontsize=32.0,loc=9)
    ax = axes()
    xl = [item[:get_text]() for item in xticks()[2]]; xl[5]="\$ \\mathbf{N} \$";
    xt = xticks()[1]; xt[5]=1350;
    xticks(xt,xl)
    #setp(ax[:get_yticklabels](),fontsize=15.0)
    ax[:set_xscale]("log")
    axright = ax[:twinx]()
    axright[:spines]["right"][:set_position](("axes",1))
    #axright[:set_yscale]("log")

    #setp(ax[:get_xticklabels](),fontsize=15.0)
    ylabel("Time [s]",fontsize=32.0)

    lab = Dict("KMeans"=>"Error N","Random Subset"=>"Time")
    for (name,dataplot) in plots
      dataplot.p_time = plot(dataplot.x,dataplot.time,linewidth=linewidth,color=dataplot.color,linestyle="--",marker="o",label=lab[name])
    end
    legend(fontsize=32.0,loc=7)
    setp(axright[:get_yticklabels](),fontsize=10.0)
    xlabel("# Inducing Points",fontsize=32.0)

    xticks(fontsize=20.0)
    yticks(fontsize=20.0)
    xlim([1,10000])

    cd(actual_folder)
  catch
    cd(actual_folder)
    error("reading $folder failed")
  end

end

function PlotTimeAccvsTime(folder)
  actual_folder = pwd()
  cd(folder)
    file_list = readdir()
    #           BSVM,  ECM,    GPC,     SSBSVM, SSGPC,    SVM
    colors = Dict("B-BSVM"=>"blue","ECM" =>"green","GPC"=>"purple","S-BSVM"=>"red","S-GPC"=>"orange","SVM"=>"black")
    linestyles = Dict("B-BSVM"=>"-","ECM"=>"-","GPC"=>"--","S-BSVM"=>"-","S-GPC"=>"--","SVM"=>"--")
    #markers = ["o","x","o","x","x",".",".","x"]
    markers = ["None","None","None","None","None","None","None","None"]
    linewidth = 3.0
    plots = Dict{String,DataPlot}()
    for i in 1:length(file_list)
      data = readdlm(file_list[i])
      smoothed_data = []
      step = 3;
      method=  file_list[i][9:end-4]
      if method == "GPC" || method == "ECM"
        smoothed_data = data
      elseif method == "BSVM"
        smoothed_data = data
        method = "B-BSVM"
      elseif method == "SSBSVM"
        smoothed_data = data
        method = "S-BSVM"
      elseif method == "SVM"
        smoothed_data = data[1:step:end,:]
      elseif method == "SSGPC"
        smoothed_data = data[1:step+2:end,:]
        method = "S-GPC"
      end
      plots[method] = DataPlot(smoothed_data[:,1],smoothed_data[:,2],colors[method],linestyles[method],linewidth,markers[i])
    end
    figure("Accuracy vs # Time")
    clf()
    plots["S-BSVM"].linewidth =4.0
    plots["B-BSVM"].linewidth =4.0
    plotsorder = ["SVM","S-GPC","S-BSVM","ECM","B-BSVM","GPC"]
    for name in plotsorder
      plots[name].p = semilogx(plots[name].x,1-0.01-plots[name].y,color=plots[name].color,linestyle=plots[name].linestyle,label=name,marker=plots[name].marker,linewidth=plots[name].linewidth)
    end
    pplots = Array{PyCall.PyObject,1}()
    legendorder = ["S-BSVM","B-BSVM","ECM","GPC","S-GPC","SVM"]
      for name in legendorder
        push!(pplots,plots[name].p[1])
      end
    legendorder[1:2] = ["S-BSVM (ours)", "B-BSVM (ours)"]
    legend(pplots,legendorder,fontsize=20)
    xlabel("Time [s]",fontsize=32.0)
    ylabel("Prediction Error",fontsize=32.0)
    xticks(fontsize=20.0)
    yticks(fontsize=20.0)
    ylim([0.08, 0.5])
    xlim([3e-1, 1e4])
  cd(actual_folder)
end

function PlotPredicvsTime(folder;Score=1)
  actual_folder = pwd()
  cd(folder)
  try
    file_list = readdir()
    colors = ["blue","black","purple","red","green","orange"]
    linestyles = ["-","--","-","--","-","--"]
    markers = ["+","x","s","o","^","o"]
    plots = []
    for i in 1:length(file_list)
      data = readdlm(file_list[i])
      if Score==1
        push!(plots,DataPlot(file_list[i][9:end-4],data[:,1],data[:,7],colors[i],linestyles[i],markers[i];x_err=data[:,2]/sqrt(10),y_err=data[:,8]/sqrt(10)))
      elseif Score==2
        push!(plots,DataPlot(file_list[i][9:end-4],data[:,1],data[:,5],colors[i],linestyles[i],markers[i];x_err=data[:,2]/sqrt(10),y_err=data[:,6]/sqrt(10)))
      elseif Score==3
        push!(plots,DataPlot(file_list[i][9:end-4],data[:,1],data[:,9],colors[i],linestyles[i],markers[i];x_err=data[:,2]/sqrt(10),y_err=data[:,10]/sqrt(10)))
      end
    end
    figure("Accuracy vs # Time")
    clf()
    for i in 1:length(plots)
      errorbar(x=plots[i].x,y=plots[i].y,xerr=plots[i].x_err,yerr=plots[i].y_err,color=plots[i].color,linestyle="None",label=plots[i].name,marker=plots[i].marker,markersize=10.0)
    end
    xlabel("Time [s]",fontsize=25)
    ylabel("Mean Error on Prediction Probability",fontsize=25)
    #ylim([0, 0.5])
    legend(fontsize=20)
  catch
    cd(actual_folder)
    err("reading $folder failed")
  end
  cd(actual_folder)
end



function MakeTable()
  actual_folder = pwd()
  methods = ["SPGGPC" "SBSVM" "LogReg" "GPC"]
  n_methods = length(methods)
  results = []
  datasets = []
  cd("../cluster_results/Epsilon E3")
  try
    folder_list = readdir()
    n_folder = length(folder_list)
    for folder in folder_list
      if folder[1:4] == "Pred"
        cd(folder)
        push!(datasets,folder[30:end-7])
        file_list = readdir()
        n_files = length(file_list)
        current = []
        for file in file_list
          data = squeeze(readdlm(file),1)
          println(file)
          push!(current,data)
        end
        println((folder))
        push!(results,hcat(hcat([current[i] for i in 1:n_files])...).')
        cd("..")
      end
    end
  catch
    cd(actual_folder)
    error("wesh")
  end
  cd(actual_folder)
  return (results,methods,datasets)
end

function ReadFromTable(table)
  for i in 1:length(table[2])
    println("$(table[2][i]) :")
    println("Loss")
    println(hcat(table[2]',1-table[1][i][:,3],table[1][i][:,4]))
    println("Brier Score")
    println(hcat(table[2]',table[1][i][:,7],table[1][i][:,8]))
  end
end

function MakeLatexTable(table)
  lines = []
  sizes = Dict("BreastCancer"=> (263,9), "Splice"=>(2991,60), "Diabetis"=>(768,8),"Thyroid"=>(215,5),"Heart"=>(270,13),"RingNorm"=>(7400,200),"German"=>(1000,20),"Waveform"=>(5000,21),"Flare"=>(144,9))
  sizes = Dict("aXa"=>(36974,123),"Bank_marketing"=>(45211,43),"Click_prediction"=>(399482,12),"Cod-rna"=>(8,343564),"Covtype"=>(581012,54),
  "Credit_card"=>(284807,30),"Electricity"=>(45312,8),"Ijcnn1"=>(141691,22),"Mnist"=>(70000,780),"Poker"=>(1025010,10),
  "Protein"=>(24837,357),"Shuttle"=>(58000,9),"Vehicle"=>(98528,100),"wXa"=>(34780,300))
  for i in 1:length(table[3])
    (nSamples,nFeatures) = sizes[table[3][i]]
    min_loss = format(min(1-table[1][i][:,3]...),precision=2)
    min_brier = format(min(table[1][i][:,7]...),precision=2)
    new_line = table[3][i]*" & $(nSamples) & $nFeatures"
    for j in 1:size(table[1][i],1)
      new_line = new_line*" & "*(format(1-table[1][i][j,3],precision=2)==min_loss ? "\\boldmath":"")*"\$ $(format(1-table[1][i][j,3],precision=2)[2:end]) \\pm $(format(table[1][i][j,4],precision=2)[2:end]) \$ &"*(format(table[1][i][j,7],precision=2)==min_brier ? "\\boldmath":"")*" \$ $(format(table[1][i][j,7],precision=2)[2:end]) \\pm $(format(table[1][i][j,8],precision=2)[2:end]) \$"
    end
    push!(lines,new_line*"\\\\")
  end
  writedlm("results/LatexFormatPredictions.txt",lines)
  return lines
end

function BiggerMakeLatexTable(table)
  lines = []
  sizes = Dict("aXa"=>(36974,123),"Bank_marketing"=>(45211,43),"Click_prediction"=>(399482,12),"Cod-rna"=>(8,343564),"Covtype"=>(581012,54),
  "Credit_card"=>(284807,30),"Electricity"=>(45312,8),"Ijcnn1"=>(141691,22),"Mnist"=>(70000,780),"Poker"=>(1025010,10),
  "Protein"=>(24837,357),"Shuttle"=>(58000,9),"Vehicle"=>(98528,100),"wXa"=>(34780,300))
  for i in 1:length(table[3])
    (nSamples,nFeatures) = sizes[table[3][i]]
    min_loss = format(min(1-table[1][i][:,3]...),precision=2)
    min_brier = format(min(table[1][i][:,5]...),precision=2)
    max_AUC = format(max(table[1][i][:,9]),precision=2)
    min_time = format(min(table[1][i][:,1]...))
    new_line = table[3][i]*" & $(nSamples) & $nFeatures"
    for j in 1:size(table[1][i],1)
      new_line = new_line*" & "*(format(table[1][i][j,1])==min_time ? "\\boldmath":"")*"\$ $(format(table[1][i][j,1]))\$"*" & "*(format(1-table[1][i][j,3],precision=2)==min_loss ? "\\boldmath":"")*"\$ $(format(1-table[1][i][j,3],precision=2)[2:end]) \\pm $(format(table[1][i][j,4],precision=2)[2:end]) \$ &"*(format(table[1][i][j,5],precision=2)==min_brier ? "\\boldmath":"")*" \$ $(format(table[1][i][j,5],precision=2)[2:end]) \\pm $(format(table[1][i][j,6],precision=2)[2:end]) \$"
    end
    push!(lines,new_line*"\\\\")
  end
  writedlm("results/BiggerLatexFormatPredictions.txt",lines)
  return lines
end

function FinalMakeLatexTable(table)
  lines = []
  sizes = Dict("aXa"=>(36974,123),"Bank_marketing"=>(45211,43),"Click_Prediction"=>(399482,12),"Cod-rna"=>(343564,8),"Covtype"=>(581012,54),
  "Credit_card"=>(284807,30),"Electricity"=>(45312,8),"Ijcnn1"=>(141691,22),"Mnist"=>(70000,780),"Poker"=>(1025010,10),
  "Protein"=>(24837,357),"Shuttle"=>(58000,9),"Vehicle"=>(98528,100),"wXa"=>(34780,300))
  for i in 1:length(table[3])
    (nSamples,nFeatures) = sizes[table[3][i]]
    min_loss = format(min(1-table[1][i][:,3]...),precision=2)
    max_AUC = format(max(table[1][i][:,9]...),precision=2)
    min_brier = format(min(table[1][i][:,5]...),precision=2)
    min_time = format(min(table[1][i][:,1]...))
    new_line = "\\multirow\{3\}\{*\}\{$(table[3][i])\}"*" & \\multirow\{3\}\{*\}\{$(nSamples)\} & \\multirow\{3\}\{*\}\{$nFeatures\} & Error "
    for j in 1:size(table[1][i],1)
      new_line = new_line*(format(1-table[1][i][j,3],precision=2)==min_loss ? " & \\boldmath":" & ")*"\$ $(format(1-table[1][i][j,3],precision=2)[2:end]) \\pm $(format(table[1][i][j,4],precision=2)[2:end]) \$"
    end
    push!(lines,new_line*"\\\\")
    new_line = " & & & Brier Score"
    for j in 1:size(table[1][i],1)
      new_line = new_line*(format(table[1][i][j,5],precision=2)==min_brier ? " & \\boldmath":" & ")*"\$ $(format(table[1][i][j,5],precision=2)[2:end]) \\pm $(format(table[1][i][j,6],precision=2)[2:end]) \$"
    end
    push!(lines,new_line*"\\\\")
    new_line = " & & & Time [s]"
    for j in 1:size(table[1][i],1)
      new_line = new_line*(format(table[1][i][j,1])==min_time ? " & \\boldmath":" & ")*" \$ $(format(table[1][i][j,1])) \$ "
    end
    push!(lines,new_line*"\\\\")
    new_line = " & & & AUC"
    for j in 1:size(table[1][i],1)
        new_line = new_line*(format(table[1][i][j,9],precision=2)==max_AUC ? " & \\boldmath":" & ")*"\$ $(format(table[1][i][j,9],precision=2)[2:end]) \\pm $(format(table[1][i][j,10],precision=2)[2:end]) \$"
    end
    push!(lines,new_line*"\\\\ \\hline")
  end
  writedlm("results/FinalLatexFormatPredictions.txt",lines)
  return lines
end

function PlotAutotuning(dataset,method;step=1)
  actualfolder = pwd()
  try
    cd(String("results/AutotuningExperiment"*dataset))
    results = readdlm(String("Results_"*method*".txt"))
    grid = readdlm(String("Grid"*method*".txt"))
    figure(String("Autotuning vs GridSearch for "*method*" on "*dataset*" dataset"))
    clf()
    semilogx(grid[1:step:end,1],1-grid[1:step:end,3],linestyle="-",linewidth=6.0,color="blue",label="Validation Loss")
    fill_between(grid[1:step:end,1],1-grid[1:step:end,3]-grid[1:step:end,4]/sqrt(10),1-grid[1:step:end,3]+grid[1:step:end,4]/sqrt(10),alpha=0.2,facecolor="blue")
    plot(results[:,5],1-results[:,3],color="red",marker="o",linestyle="None",markersize=22.0,label=L"$\theta$ learned by S-BSVM")
    xlabel(L"\theta",fontsize=35.0)
    ylabel("Validation Loss",fontsize=32.0)
    ylim([0, 0.55])
    xticks(fontsize=20.0)
    yticks(fontsize=20.0)
    legend(fontsize=28.0,numpoints=1,loc=4)
  catch
    cd(actualfolder)
    error("wesh")
  end
  cd(actualfolder)
end

function ROC_Drawing(stochastic::Bool=true)
  data = readdlm(String("results/BigDataExperimentSUSY/ROC_"*(stochastic?"S":"")*"SBSVM.txt"))
  AUC = readdlm(String("results/BigDataExperimentSUSY/AUC_"*(stochastic?"S":"")*"SBSVM.txt"))[1]
  dp = readdlm("results/BigDataExperimentSUSY/DeepLearning.txt")
  spec = data[:,1]; sens = data[:,2]
  figure(String("ROC "*(stochastic?"S":"")*"SBSVM"));
  clf()
  plot(sens,spec,linewidth=5.0,label="S-BSVM                 (AUC=0.84)")
  fill_between(sens,zeros(length(sens)),spec,alpha=0.1,color="blue")
  plot(dp[:,1],dp[:,2],linewidth=4.0,linestyle="--",color="red",label="Deep Learning      (AUC=0.88)")
  plot([0,1],[0,1],linestyle="--",linewidth=4.0,label="Random Classifier (AUC = 0.5)")
  text(0.5,0.05,"AUC = $(round(AUC*10000)/10000)",fontsize=32.0)
  xlabel("False Positive Rate",fontsize=32.0)
  ylabel("True Positive Rate",fontsize=32.0)
  xticks(fontsize=20.0)
  yticks(fontsize=20.0)
  legend(fontsize=30.0,loc=4)
end
