# -*- coding: utf-8 -*-

"""Main module."""
from plot_iverson_lsdtt import preprocessing as pp
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


class PLOTIV(object):
  """This class holds the routines/ informations for plotting results from Iverson (2000) landslide triggering model implemented in LSDTopoTools
    Still under development, things might change quitequickly and be unstable.
    Authors:
    B.G and S.M.M
  """
  def __init__(self, prefix = "TEST", path = "./", verbose = True):
    """
      Initialise the plotting and preprocessed the data required. It automatically reads all the inputs from a path and prefix
      The inputs are formatted and produce by the cpp code in LSDTopoTools LSDPoreWaterColumns and LSDPorewaterParams files.
      Please refer to the Documentation (Schroedinger doc).
      Best,
      B.G
    """
    super(PLOTIV, self).__init__()
    self.verbose = verbose
    self.prefix = prefix
    self.path = path

    print("Loading the files ...") if self.verbose else 0

    try:
      # Loading the time series into the model
      self.df_TS_depth_F_c = pd.read_csv(self.path+self.prefix+"_time_series_depth_F_c.csv")
      self.df_TS_depth_F_f = pd.read_csv(self.path+self.prefix+"_time_series_depth_F_f.csv")
      self.df_TS_depth_F_w = pd.read_csv(self.path+self.prefix+"_time_series_depth_F_w.csv")
      self.df_TS_depth_FS = pd.read_csv(self.path+self.prefix+"_time_series_depth_FS.csv")
      self.df_TS_depth_Psi = pd.read_csv(self.path+self.prefix+"_time_series_depth_Psi.csv")

      self.df_failure_pot = pd.read_csv(self.path+self.prefix+"_potfailfile.csv")

      self.depths = self.df_TS_depth_Psi["depth"].values
      self.times = self.df_TS_depth_Psi.columns.values[1:]
    except FileNotFoundError:
      print("Ignored the 1D files. Just the 2D routines will work for that")

    self.df_input_prec = None

    print("Done") if self.verbose else 0


    # print("depth are:")
    # print(self.depths)
    # print("times are:")
    # print(self.times)

  def ingest_preprocessed_precipitation_input(self, file = "./preprocessed.csv"):
    """
    This ingest in the object an input precipitation file in the model
    """
    self.df_input_prec = pd.read_csv(file)
    self.df_input_prec = self.df_input_prec[self.df_input_prec["intensity_mm_sec"] != -9999]

  def plot_input_precipitation(self, fig = None, ax = None, return_fig = False, dpi = 500):
    """
    Plots a bar diagram of the input prec
    B.G 05/02/2019
    """
    if(self.df_input_prec is None):
      print("FATAL ERROR::Cannot plot precipitation data if I haven't ingested the file")
      quit()

    if(fig is None or ax is None):
      fig,ax = plt.subplots()
    # print(np.cumsum(self.df_input_prec["duration_s"])/24/3600)
    # quit()
    ax.bar(np.cumsum(self.df_input_prec["duration_s"])/24/3600, self.df_input_prec["intensity_mm_sec"]*3600*24,width =self.df_input_prec["duration_s"]/24/3600, color = "blue", alpha = 0.2 , edgecolor = "blue", align = "edge")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("precipitation (mm/day)")
    ax.set_xlim(np.nanmin(np.cumsum(self.df_input_prec["duration_s"])/24/3600), np.nanmax(np.cumsum(self.df_input_prec["duration_s"])/24/3600))

    plt.savefig(self.path+self.prefix+"_input_prec.png", dpi = dpi)

    # Returning or deleting
    if(return_fig):
      return fig,ax
    else:
      plt.close(fig)



  def plot_psi_vs_depth(self, fig = None, ax = None, return_fig = False ,dpi = 500, colormap = 'gnuplot2', lw = 0.5):
    """
    This plots transient pore water pressure function of depth and time.
    B.G.
    05/02/2017
    """
    # Creating the subplots
    if(fig is None or ax is None):
      fig,ax = plt.subplots()

    # Dealing with the line color: It is quite hard to color it by values and here is an hacky way
    norm = matplotlib.colors.Normalize(vmin=np.nanmin(self.times), vmax=np.nanmax(self.times)) # normalization to min max
    tcmap = matplotlib.cm.get_cmap(colormap) # chosing the colormap from https://matplotlib.org/examples/color/colormaps_reference.html

        # each time will be plotted
    for t in self.times:
      # No need to explain that innit
      ax.plot(self.df_TS_depth_Psi[t],-1*self.depths, lw = lw)#, color = tcmap(norm(float(t))))

    # hacky plotting of the right colormap
    # cb = ax.scatter(self.df_TS_depth_Psi[self.times[-1]],-1*self.depths, c = self.depths, s=0, cmap = colormap, vmin=np.nanmin(self.times), vmax=np.nanmax(self.times))
    # plt.colorbar(cb)

    # Just moving the axis to the top
    ax.xaxis.set_ticks_position('top')
    ax.set_xlabel(r"$\Psi^{*}$")

    ax.set_ylabel("Depth (m)")

    plt.savefig(self.path+self.prefix+"_Psi_depth_TS.png", dpi = dpi)

    # Returning or deleting
    if(return_fig):
      return fig,ax
    else:
      plt.close(fig)

  def col_plot_of_FS(self, dpi = 500, highligh_specific_time_in_sec = []):
    """
      No time to explain.
    """
    fig,ax = plt.subplots()
       # Adding the highlighed times
    htimes = []
    for t in range(len(highligh_specific_time_in_sec)):
      # First I need to get the time equivalence
      this_time = "-9999"
      for tt in range(self.times.shape[0]):
        # print(float(self.times[tt])," is bigger than ", highligh_specific_time_in_sec[t], "??? according to python: ", float(self.times[tt])>highligh_specific_time_in_sec[t])
        if(float(self.times[tt])>highligh_specific_time_in_sec[t] and this_time == "-9999"):
          this_time = self.times[tt]
          htimes.append(this_time)
          # print(this_time)
        # Now I can plot it in bold red

    def _plot_of_FS(toplot, this_colormap,savename, ntickx = 5, nticky = 5, colorbar_label = "", highligh_specific_time_in_sec = []):
      """
      Nested function for nasty work
      """
      nrows = toplot.shape[0];ncols = toplot.shape[1]
      cb = ax.imshow(toplot,aspect = "auto",  cmap = this_colormap)
      cbar = plt.colorbar(cb)
      cbar.ax.set_ylabel(colorbar_label, rotation=270)
      ax.xaxis.tick_top()
      ax.set_xlabel("time (days)")
      ax.set_ylabel("Depth (m)")
      ax.xaxis.set_label_position('top')
      # OK let's deal with the axis ticks now, #PainInTheAss
      # ax.get_xticks is fucked up here so I need to first assign it manually for some reasons
      # First I need to create normalization tools, basically I want to transform the row col ticks to time/depth
      # for example tick at row #1000 -> normalise to 0,1 -> inverse normalise to depth
      xnorm_LAB = matplotlib.colors.Normalize(vmin=np.nanmin(self.times.astype(np.float32)), vmax=np.nanmax(self.times.astype(np.float32)))
      x_norm_RC = matplotlib.colors.Normalize(vmin=0, vmax=np.nanmax(ncols))
      ynorm_LAB = matplotlib.colors.Normalize(vmin=np.nanmin(self.depths.astype(np.float32)), vmax=np.nanmax(self.depths.astype(np.float32)))
      y_norm_RC = matplotlib.colors.Normalize(vmin=0, vmax=np.nanmax(nrows))
      xticks = []
      yticks = []
      xticklabels = []
      yticklabels = []
      for i in range(ntickx):
        xticks.append(i*round(ncols/ntickx))
        xticklabels.append("%.2f"%(xnorm_LAB.inverse(x_norm_RC(xticks[-1]))/24/3600))

      for i in range(nticky):
        yticks.append(i*round(nrows/nticky))
        yticklabels.append("%.2f"%ynorm_LAB.inverse(y_norm_RC(yticks[-1])))

      for vl in highligh_specific_time_in_sec:
        ax.axvline(x_norm_RC.inverse(xnorm_LAB(float(vl))), 0,1, lw =1.5, color = "k", zorder = 50)
      
      ax.set_xticks(xticks); ax.set_yticks(yticks);ax.set_xticklabels(xticklabels);ax.set_yticklabels(yticklabels)

      # DOne
      plt.savefig(self.path + self.prefix + savename, dpi = dpi)
      plt.clf()

    print("I am plotting the factor of safety figure: %s"%(self.path + self.prefix + "_test_2d_FS.png")) if self.verbose else 0
    toplot = self.df_TS_depth_FS[self.df_TS_depth_FS.columns[1:]].to_numpy()    
    _plot_of_FS(toplot, "RdBu","_test_2d_FS.png", ntickx = 5, nticky = 5, colorbar_label = "FS",highligh_specific_time_in_sec = htimes)

    print("I am plotting the F_w time series: %s"%(self.path + self.prefix + "_test_2d_F_w.png")) if self.verbose else 0
    fig,ax = plt.subplots()
    toplot = self.df_TS_depth_F_w[self.df_TS_depth_F_w.columns[1:]].to_numpy()
    _plot_of_FS(toplot, "RdBu","_test_2d_F_w.png", ntickx = 5, nticky = 5, colorbar_label = r"$F_{w}$",highligh_specific_time_in_sec = htimes)


    print("I am plotting the F_c time series: %s"%(self.path + self.prefix + "_test_2d_F_c.png")) if self.verbose else 0
    fig,ax = plt.subplots()
    toplot = self.df_TS_depth_F_c[self.df_TS_depth_F_c.columns[1:]].to_numpy()
    _plot_of_FS(toplot, "RdBu","_test_2d_F_c.png", ntickx = 5, nticky = 5, colorbar_label = r"$F_{c}$",highligh_specific_time_in_sec = htimes)

    print("I am plotting the Psi time series: %s"%(self.path + self.prefix + "_test_2d_Psi.png")) if self.verbose else 0
    fig,ax = plt.subplots()
    toplot = self.df_TS_depth_Psi[self.df_TS_depth_Psi.columns[1:]].to_numpy()
    _plot_of_FS(toplot, "magma","_test_2d_Psi.png", ntickx = 5, nticky = 5, colorbar_label = r"$\Psi$",highligh_specific_time_in_sec = htimes)    


  def fig7_8_or_10_11_Iverson2000(self, n_times = 20, xaxis = "Psi", highligh_specific_time_in_sec = []):
    """
    Creates a figure similar then fig. 7 in Iverson 2000 paper "Landslide triggering by rain infiltration"

    """

    fig,ax = plt.subplots()

    # Getting the selected times
    get_times = []
    for i in range(n_times):
      get_times.append(self.times[round(self.times.shape[0]/n_times)*i])
    # print( get_times)
    # Getting the last time anyway
    get_times.append(self.times[-1])

    tempget_times = np.array(get_times, dtype = np.float64)

    # Dealing with the line color: It is quite hard to color it by values and here is an hacky way
    norm = matplotlib.colors.Normalize(vmin=np.nanmin(tempget_times), vmax=np.nanmax(tempget_times)) # normalization to min max
    tcmap = matplotlib.cm.get_cmap("jet") # chosing the colormap from https://matplotlib.org/examples/color/colormaps_reference.html
    
    if(xaxis == "Psi"):
      tdf = self.df_TS_depth_Psi
      print("I am plotting the Psi evolution, Iverson (2000) figure 7-8 style: %s"%(self.path + self.prefix + "_fig7Iverson2000.png")) if self.verbose else 0

    else:
      tdf = self.df_TS_depth_FS
      print("I am plotting the factor of safety evolution, Iverson (2000) figure 10-11 style: %s and %s (a version focused on the stability area between 0 and 2)"%(self.path + self.prefix + "_fig10Iverson2000.png", self.path + self.prefix + "_fig10Iverson2000_zoom.png")) if self.verbose else 0

    ax.plot(tdf[get_times[0]], self.depths, color = "k", lw = 1.5)
    for t in get_times:
      # print(norm(float(t)))
      ax.plot(tdf[t], self.depths, color = tcmap(norm(float(t))), lw = 0.75, ls = "--")

    # Adding the highlighed times
    for t in range(len(highligh_specific_time_in_sec)):
      # First I need to get the time equivalence
      this_time = "-9999"
      for tt in range(self.times.shape[0]):
        # print(float(self.times[tt])," is bigger than ", highligh_specific_time_in_sec[t], "??? according to python: ", float(self.times[tt])>highligh_specific_time_in_sec[t])
        if(float(self.times[tt])>highligh_specific_time_in_sec[t] and this_time == "-9999"):
          this_time = self.times[tt]
          # print(this_time)
        # Now I can plot it in bold red
      ax.plot(tdf[this_time], self.depths, color = tcmap(norm(float(this_time))), lw = 1.5)


    # colorbar hacky
    cb = ax.scatter(tdf[get_times[0]], self.depths, s =0, c = self.depths, cmap = "jet",vmin=np.nanmin(tempget_times), vmax=np.nanmax(tempget_times) )
    cbar = plt.colorbar(cb)
    cbar.ax.set_ylabel("Time (s)")

    ax.invert_yaxis()
    if(xaxis == "Psi"):
      ax.set_xlabel("Pressure Head (m)")
      suffix = "_fig7Iverson2000.png"
    else:
      ax.set_xlabel("FS")
      ax.axvline(1,0,1,color = "k", lw = 1.75)
      suffix = "_fig10Iverson2000.png"

    ax.set_ylabel("Depth (m)")
    #Done

    plt.savefig(self.path+self.prefix+suffix, dpi = 500)

    if(xaxis == "FS"):
      # creating a zoom on the critical area
      ax.set_xlim(0,2)
      plt.savefig(self.path+self.prefix+"_fig10Iverson2000_zoom.png", dpi = 500)

    plt.clf()


  def failure_time(self, observed_failures_in_sec = [], gradual_motions_in_sec_start = [], gradual_motions_in_sec_end = []):
    """
    A bit like my life lolz
    """
    if(isinstance(gradual_motions_in_sec_start,np.ndarray)):
      gradual_motions_in_sec_start = gradual_motions_in_sec_start.tolist()
      gradual_motions_in_sec_end = gradual_motions_in_sec_end.tolist()

    if(self.df_input_prec is None):
      print("FATAL ERROR::I need to have ingested the precipitation file first. Use ingest_preprocessed_precipitation_input function")
      quit()

    # returns the figure first to get the prec
    fig,ax = self.plot_input_precipitation(fig = None, ax = None, return_fig = True, dpi = 500)

    fig.set_figheight(5)
    fig.set_figwidth(6)

    ax_fail = ax.twinx()

    #plot observed failures
    for i in observed_failures_in_sec:
      ax.axvline(i/24/3600,0,1,color = "r", lw = 2, ls = "--", zorder = 10)
    for i in range(len(gradual_motions_in_sec_start)):
      ax.axvspan(gradual_motions_in_sec_start[i]/24/3600, gradual_motions_in_sec_end[i]/24/3600, alpha = 0.3, color = "r", lw =0, zorder = 10)


    # get the indices of the break line
    index_breaks = self.df_failure_pot.index[self.df_failure_pot["time"] == -9999].tolist()
    index_breaks.append(self.df_failure_pot.index[-1])
    last_i = -1
    ls_of_df = []
    for i in index_breaks:
      if(i != 0):
        ls_of_df.append(self.df_failure_pot[last_i+1:i])
      last_i = i


    for df in ls_of_df:
      ax_fail.fill_between(df["time"]/24/3600,df["min_depth"], df["max_depth"]+0.1*np.nanmax(self.depths), color = "#55FF00", alpha = 0.2, lw =0, zorder = 8)
      ax_fail.plot(df["time"]/24/3600, df["min_depth"], color = "#55FF00", lw = 1.5, zorder =8)
      if(df["time"].values.shape[0]>0):
        ax_fail.scatter(df["time"].values[0]/24/3600,df["min_depth"].values[0],marker = "*", color = "#55FF00", lw = 0, s=50, zorder =9)


    ax_fail.set_ylim(np.nanmin(self.depths),np.nanmax(self.depths)+0.1*np.nanmax(self.depths))

    ax_fail.invert_yaxis()

    ax_fail.set_ylabel("Depth of failure (m)")

    plt.savefig(self.path+self.prefix+"_are_you_a_failure.png", dpi = 500)
    plt.clf()



  def plot_spatial_slope_analysis(self, base_raster_name, time_as_diplayed, alpha = 0.6):
    """
    Plot two map figures: the min FS and its depth.
    Calculated when spatial analysis will be on.
    Requires lsdtopytools to be installed.
    """

    from lsdtopytools import LSDDEM, quickplot as qp

    # Loading the raster
    this_dem = LSDDEM(path = self.path , file_name = base_raster_name, already_preprocessed = True)
    FS_dem = LSDDEM(path = self.path, file_name = self.prefix + "_minFS_at_%s.bil"%(time_as_diplayed),already_preprocessed = True)

    # Getting the base figure
    fig, ax = qp.plot_nice_topography(this_dem ,figure_width = 6, cmap = "gist_earth", hillshade = True, alpha_hillshade = 1, output = "return")

    cb = ax.imshow(FS_dem.cppdem.get_PP_raster(), extent = FS_dem.extent, vmin = 0, vmax = 2, cmap ="seismic_r", alpha = alpha, zorder = 5)

    plt.colorbar(cb)

    plt.savefig(self.path+self.prefix+"FS_map_at_%s.png"%(time_as_diplayed), dpi = 500)
    plt.clf()




