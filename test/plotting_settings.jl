
upscale = 1.0 #8x upscaling in resolution
fntsm = Plots.font("sans-serif", 10.0*upscale)
fntlg = Plots.font("sans-serif", 14.0*upscale)
default(titlefont=fntlg, guidefont=fntlg, tickfont=fntsm, legendfont=fntsm)
default(size=(800*upscale,600*upscale)) #Plot canvas size
default(dpi=300) #Only for PyPlot - presently broken
