from calendar import month_name
from datetime import date, timedelta
from time import time

import geopandas as gpd
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from pyevtk.hl import gridToVTK
from shapely.geometry import Point
from sklearn.ensemble \
    import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics \
    import precision_score, recall_score
from statsmodels.nonparametric.kernel_density \
    import KDEMultivariate, EstimatorSettings

import predictivehp.aux_functions as af
from predictivehp.processing.data_processing import *

# from paraview.simple import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
settings = EstimatorSettings(efficient=True,
                             n_jobs=8)


class MyKDEMultivariate(KDEMultivariate):
    def resample(self, size: int):
        # print("\nResampling...", end=" ")

        n, d = self.data.shape
        indices = np.random.randint(0, n, size)

        cov = np.diag(self.bw) ** 2
        means = self.data[indices, :]
        norm = np.random.multivariate_normal(np.zeros(d), cov, size)

        # simulated and checked points
        s_points = np.transpose(means + norm)
        c_points = af.checked_points(s_points)

        # print(f"\n{size - c_points.shape[1]} invalid points found")

        if size == c_points.shape[1]:
            # print("\nfinished!")
            return s_points

        a_points = self.resample(size - c_points.shape[1])

        return np.hstack((c_points, a_points))


class STKDE:
    def __init__(self,
                 n: int = 1000,
                 year: str = "2017",
                 bw=None, df=None, sample_number=3600, training_months=10,
                 number_of_groups=1,
                 window_days=7, month_division=10, name="STKDE"):
        """
        n: Número de registros que se piden a la database.
        year: Año de los registros pedidos
        t_model: Entrenamiento del modelo, True en caso de que se quieran
        usar los métodos contour_plot o heatmap.
        """
        self.name, self.n, self.year, self.bw = name, n, year, bw
        self.X_months = training_months
        self.ng, self.wd, self.md = number_of_groups, window_days, month_division
        self.sn = sample_number

        self.hr, self.ap, self.pai = None, None, None
        self.hr_by_group, self.ap_by_group, self.pai_by_group = None, None, None

        # training data 3000
        # testing data  600
        self.df = df
        print('-' * 30)
        print('\t\tSTKDE')
        print(af.print_mes(self.X_months, self.X_months + 1, self.wd))

        print('-' * 30)


    @af.timer
    def fit(self, df, X, y, predict_groups):
        """

        :param pd.DataFrame df: Initial Dataframe.
        :param pd.DataFrame  X: Training data.
        :param pd.DataFrame  Y: Testing data.
        :param list  predict_groups: List with data separate in groups
                                    and with corresponding windows.
        """
        self.data, self.X, self.y, self.pg = df, X, y, predict_groups

        print("\nBuilding KDE...")

        self.kde = MyKDEMultivariate(
            [np.array(self.X[['x']]),
             np.array(self.X[['y']]),
             np.array(self.X[['y_day']])],
            'ccc', bw=self.bw)

        self.bw = self.kde.bw

    @af.timer
    def data_barplot(self, pdf: bool = False):
        """
        Bar Plot
        pdf: True si se desea guardar el plot en formato pdf
        """

        print("\nPlotting Bar Plot...")

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.tick_params(axis='x', length=0)

        for i in range(1, 13):
            count = self.data[
                (self.data["date"].apply(lambda x: x.month) == i)
            ].shape[0]

            plt.bar(x=i, height=count, width=0.25, color=["black"])
            plt.text(x=i - 0.275, y=count + 5, s=str(count))

        plt.xticks(
            [i for i in range(1, 13)],
            [datetime.datetime.strptime(str(i), "%m").strftime('%b')
             for i in range(1, 13)])

        sns.despine()

        plt.xlabel("Month",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold'},
                   labelpad=10
                   )
        plt.ylabel("Count",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold'},
                   labelpad=7.5
                   )

        if pdf:
            plt.savefig(f"output/barplot.pdf", format='pdf')

        plt.show()

    @af.timer
    def spatial_pattern(self,
                        pdf: bool = False):
        """
        Spatial pattern of incidents
        pdf: True si se desea guardar el plot en formato pdf
        """

        print("\nPlotting Spatial Pattern of incidents...", sep="\n\n")

        print("\tReading shapefiles...", end=" ")
        dallas_districts = gpd.GeoDataFrame.from_file(
            "../Data/Councils/councils.shp")
        dallas = gpd.read_file('../Data/shapefiles/streets.shp')
        print("finished!")

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_facecolor('xkcd:black')

        # US Survey Foot: 0.3048 m
        # print("\n", f"EPSG: {dallas.crs['init'].split(':')[1]}")  # 2276

        geometry = [Point(xy) for xy in zip(
            np.array(self.y[['x']]),
            np.array(self.y[['y']]))
                    ]
        geo_df = gpd.GeoDataFrame(self.y,
                                  crs=dallas.crs,
                                  geometry=geometry)

        print("\tPlotting Districts...", end=" ")

        handles = []

        for district, data in dallas_districts.groupby('DISTRICT'):
            data.plot(ax=ax,
                      color=d_colors[district],
                      linewidth=2.5,
                      edgecolor="black")
            handles.append(mpatches.Patch(color=d_colors[district],
                                          label=f"Dallas District {district}"))

        handles.sort(key=lambda x: int(x._label.split(' ')[2]))
        handles = [Line2D([], [], marker='o', color='red', label='Incident',
                          linestyle="None"),
                   Line2D([0], [0], color="steelblue", label="Streets")] + \
                  handles

        print("finished!")

        print("\tPlotting Streets...", end=" ")
        dallas.plot(ax=ax,
                    alpha=0.4,
                    color="steelblue",
                    zorder=2,
                    label="Streets")
        print("finished!")

        print("\tPlotting Incidents...", end=" ")

        geo_df.plot(ax=ax,
                    markersize=17.5,
                    color='red',
                    marker='o',
                    zorder=3,
                    label="Incidents")

        print("finished!")

        plt.title(f"Dallas Incidents - Spatial Pattern\n"
                  f"{self.year}",
                  fontdict={'fontsize': 20},
                  pad=25)

        plt.legend(loc="lower right",
                   frameon=False,
                   fontsize=13.5,
                   handles=handles)

        ax.set_axis_off()
        plt.show()

        if pdf:
            plt.savefig("output/spatial_pattern.pdf", format='pdf')
        plt.show()

    @af.timer
    def contour_plot(self,
                     bins: int,
                     ti: int,
                     pdf: bool = False):
        """
        Draw the contour lines
        bins:
        ti:
        pdf: True si se desea guardar el plot en formato pdf
        """

        print("\nPlotting Contours...")

        dallas = gpd.read_file('../Data/shapefiles/streets.shp')

        fig, ax = plt.subplots(figsize=(15, 12))
        ax.set_facecolor('xkcd:black')

        dallas.plot(ax=ax,
                    alpha=.4,
                    color="gray",
                    zorder=1)

        x, y = np.mgrid[
               np.array(self.y[['x']]).min():
               np.array(self.y[['x']]).max():bins * 1j,
               np.array(self.y[['y']]).min():
               np.array(self.y[['y']]).max():bins * 1j
               ]

        z = self.kde.pdf(np.vstack([x.flatten(),
                                    y.flatten(),
                                    ti * np.ones(x.size)]))
        # z_2 = z * 3000 * (10 ** 6) / (.304) # P. Elwin

        contourplot = plt.contour(x, y, z.reshape(x.shape),
                                  cmap='jet',
                                  zorder=2)

        plt.title(f"Dallas Incidents - Contourplot\n"
                  f"n = {self.data.shape[0]}    Year = {self.year}",
                  fontdict={'fontsize': 20},
                  pad=20)
        plt.colorbar(contourplot,
                     ax=ax,
                     shrink=.4,
                     aspect=10)

        if pdf:
            plt.savefig("output/dallas_contourplot.pdf", format='pdf')
        plt.show()

    @af.timer
    def heatmap(self,
                bins=100,
                ti=100,
                pdf=False):
        """
        Plots the heatmap associated to a given t_i
        bins:
        ti:
        pdf:
        """
        print("\nPlotting Heatmap...")

        dallas = gpd.read_file('predictivehp/data/streets.shp')

        fig, ax = plt.subplots(figsize=(15, 12))
        ax.set_facecolor('xkcd:black')

        dallas.plot(ax=ax,
                    alpha=.4,  # Ancho de las calles
                    color="gray",
                    zorder=1)

        x, y = np.mgrid[
               np.array(self.y[['x']]).min():
               np.array(self.y[['x']]).max():bins * 1j,
               np.array(self.y[['y']]).min():
               np.array(self.y[['y']]).max():bins * 1j
               ]
        z = self.kde.pdf(np.vstack([x.flatten(),
                                    y.flatten(),
                                    ti * np.ones(x.size)]))

        # Normalizar
        z = z / z.max()

        heatmap = plt.pcolormesh(x, y, z.reshape(x.shape),
                                 shading='gouraud',
                                 alpha=.2,
                                 cmap='jet',
                                 zorder=2,
                                 )

        plt.title(f"Dallas Incidents - Heatmap\n"
                  f"n = {self.data.shape[0]}   Year = {self.year}",
                  fontdict={'fontsize': 20},
                  pad=20)
        cbar = plt.colorbar(heatmap,
                            ax=ax,
                            shrink=.5,
                            aspect=10)
        cbar.solids.set(alpha=1)
        # ax.set_axis_off()

        if pdf:
            plt.savefig("output/dallas_heatmap.pdf", format='pdf')
        plt.show()

    @af.timer
    def generate_grid(self,
                      bins: int = 100):
        """
        :return:
        """
        print("\nCreating 3D grid...")

        x, y, t = np.mgrid[
                  np.array(self.y[['x']]).min():
                  np.array(self.y[['x']]).max():bins * 1j,
                  np.array(self.y[['y']]).min():
                  np.array(self.y[['y']]).max():bins * 1j,
                  np.array(self.y[['y_day']]).min():
                  np.array(self.y[['y_day']]).max():60 * 1j
                  ]

        print(x.shape)

        print("\n\tEstimating densities...")

        d = self.kde.pdf(
            np.vstack([
                x.flatten(),
                y.flatten(),
                t.flatten()
            ])).reshape((100, 100, 60))

        print("\nExporting 3D grid...")

        gridToVTK("STKDE grid",
                  x, y, t,
                  pointData={"density": d,
                             "y_day": t})

    @af.timer
    def plot_4d(self,
                jpg: bool = False,
                interactive: bool = False):
        """
        Docstring
        """

        print("\nCreating 4D plot...")

        # get the material library
        materialLibrary1 = GetMaterialLibrary()

        # Create a new 'Render View'
        renderView1 = CreateView('RenderView')
        # renderView1.GetRenderWindow().SetFullScreen(True)
        renderView1.ViewSize = [1080, 720]
        renderView1.AxesGrid = 'GridAxes3DActor'
        renderView1.CenterOfRotation = [2505251.4078454673, 6981929.658190809,
                                        336000.0]
        renderView1.StereoType = 0
        renderView1.CameraPosition = [2672787.133709079, 6691303.18204621,
                                      510212.0148339354]
        renderView1.CameraFocalPoint = [2505211.707979299, 6981538.82059641,
                                        335389.1971413433]
        renderView1.CameraViewUp = [-0.23106610676072656, 0.40064135354217617,
                                    0.8866199637603102]
        renderView1.CameraParallelScale = 97832.66331727587
        renderView1.Background = [0.0, 0.0, 0.0]
        renderView1.OSPRayMaterialLibrary = materialLibrary1

        # init the 'GridAxes3DActor' selected for 'AxesGrid'
        renderView1.AxesGrid.XTitle = 'X'
        renderView1.AxesGrid.YTitle = 'Y'
        renderView1.AxesGrid.ZTitle = 'T'
        renderView1.AxesGrid.XTitleFontFile = ''
        renderView1.AxesGrid.YTitleFontFile = ''
        renderView1.AxesGrid.ZTitleFontFile = ''
        renderView1.AxesGrid.XLabelFontFile = ''
        renderView1.AxesGrid.YLabelFontFile = ''
        renderView1.AxesGrid.ZLabelFontFile = ''

        # ----------------------------------------------------------------
        # restore active view
        SetActiveView(renderView1)
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # setup the data processing pipelines
        # ----------------------------------------------------------------

        # Si no existe STKDE_grid.vts, crear malla (INCOMPLETO)

        # create a new 'XML Structured Grid Reader'

        print("\n\tLoading 3D grid...", end=" ")

        densities = XMLStructuredGridReader(FileName=[
            '/Users/msmendozaelguera/Desktop/iPre/Modeling/Python/STKDE '
            'grid.vts'])
        densities.PointArrayStatus = ['density', 'y_day']

        print("finished!")

        # create a new 'GDAL Vector Reader'
        print("\tLoading Dallas Street shapefile...", end=" ")

        dallasMap = GDALVectorReader(
            FileName='/Users/msmendozaelguera/Desktop/iPre/Modeling/Data'
                     '/shapefiles'
                     '/streets.shp')

        print("finished!")

        print("\tPlotting Contours...", end=" ")

        # create a new 'Contour'
        aboveSurfaceContour = Contour(Input=densities)
        aboveSurfaceContour.ContourBy = ['POINTS', 'y_day']
        aboveSurfaceContour.Isosurfaces = [360.0]
        aboveSurfaceContour.PointMergeMethod = 'Uniform Binning'

        # create a new 'Contour'
        aboveContour = Contour(Input=aboveSurfaceContour)
        aboveContour.ContourBy = ['POINTS', 'density']
        aboveContour.ComputeScalars = 1
        aboveContour.Isosurfaces = [1.2217417871681798e-13]
        aboveContour.PointMergeMethod = 'Uniform Binning'

        # create a new 'Contour'
        belowSurfaceContour = Contour(Input=densities)
        belowSurfaceContour.ContourBy = ['POINTS', 'y_day']
        belowSurfaceContour.Isosurfaces = [307.0]
        belowSurfaceContour.PointMergeMethod = 'Uniform Binning'

        # create a new 'Contour'
        belowContour = Contour(Input=belowSurfaceContour)
        belowContour.ContourBy = ['POINTS', 'density']
        belowContour.ComputeScalars = 1
        belowContour.Isosurfaces = [1.8379040711040815e-12]
        belowContour.PointMergeMethod = 'Uniform Binning'

        print("finished!")

        # ----------------------------------------------------------------
        # setup the visualization in view 'renderView1'
        # ----------------------------------------------------------------

        print("\tRendering Volume...", end=" ")

        # show data from densities
        densitiesDisplay = Show(densities, renderView1)

        # get color transfer function/color map for 'density'
        densityLUT = GetColorTransferFunction('density')
        densityLUT.RGBPoints = [0.0, 0.278431372549, 0.278431372549,
                                0.858823529412,
                                5.418939399286293e-13, 0.0, 0.0,
                                0.360784313725,
                                1.0799984117458697e-12, 0.0, 1.0, 1.0,
                                1.625681819785888e-12, 0.0, 0.501960784314,
                                0.0,
                                2.1637862916031283e-12, 1.0, 1.0, 0.0,
                                2.7056802315317578e-12, 1.0, 0.380392156863,
                                0.0,
                                3.247574171460387e-12, 0.419607843137, 0.0,
                                0.0,
                                3.7894681113890166e-12, 0.878431372549,
                                0.301960784314,
                                0.301960784314]
        densityLUT.ColorSpace = 'RGB'
        densityLUT.ScalarRangeInitialized = 1.0

        # get opacity transfer function/opacity map for 'density'
        densityPWF = GetOpacityTransferFunction('density')
        densityPWF.Points = [0.0, 0.0, 0.5, 0.0, 3.7894681113890166e-12, 1.0,
                             0.5, 0.0]
        densityPWF.ScalarRangeInitialized = 1

        # trace defaults for the display properties.
        densitiesDisplay.Representation = 'Volume'
        densitiesDisplay.ColorArrayName = ['POINTS', 'density']
        densitiesDisplay.LookupTable = densityLUT
        densitiesDisplay.Scale = [1.0, 1.0, 1000.0]
        densitiesDisplay.OSPRayScaleArray = 'density'
        densitiesDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
        densitiesDisplay.SelectOrientationVectors = 'None'
        densitiesDisplay.ScaleFactor = 13997.517379119061
        densitiesDisplay.SelectScaleArray = 'None'
        densitiesDisplay.GlyphType = 'Arrow'
        densitiesDisplay.GlyphTableIndexArray = 'None'
        densitiesDisplay.GaussianRadius = 699.875868955953
        densitiesDisplay.SetScaleArray = ['POINTS', 'density']
        densitiesDisplay.ScaleTransferFunction = 'PiecewiseFunction'
        densitiesDisplay.OpacityArray = ['POINTS', 'density']
        densitiesDisplay.OpacityTransferFunction = 'PiecewiseFunction'
        densitiesDisplay.DataAxesGrid = 'GridAxesRepresentation'
        densitiesDisplay.SelectionCellLabelFontFile = ''
        densitiesDisplay.SelectionPointLabelFontFile = ''
        densitiesDisplay.PolarAxes = 'PolarAxesRepresentation'
        densitiesDisplay.ScalarOpacityFunction = densityPWF
        densitiesDisplay.ScalarOpacityUnitDistance = 2272.3875215933476

        # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
        densitiesDisplay.DataAxesGrid.XTitleFontFile = ''
        densitiesDisplay.DataAxesGrid.YTitleFontFile = ''
        densitiesDisplay.DataAxesGrid.ZTitleFontFile = ''
        densitiesDisplay.DataAxesGrid.XLabelFontFile = ''
        densitiesDisplay.DataAxesGrid.YLabelFontFile = ''
        densitiesDisplay.DataAxesGrid.ZLabelFontFile = ''

        # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
        densitiesDisplay.PolarAxes.Scale = [1.0, 1.0, 1000.0]
        densitiesDisplay.PolarAxes.PolarAxisTitleFontFile = ''
        densitiesDisplay.PolarAxes.PolarAxisLabelFontFile = ''
        densitiesDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
        densitiesDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

        print("finished!")

        print("\tMaking some adjustments...", end=" ")

        # show data from belowContour
        belowContourDisplay = Show(belowContour, renderView1)

        # trace defaults for the display properties.
        belowContourDisplay.Representation = 'Wireframe'
        belowContourDisplay.ColorArrayName = ['POINTS', 'density']
        belowContourDisplay.LookupTable = densityLUT
        belowContourDisplay.LineWidth = 2.0
        belowContourDisplay.Scale = [1.0, 1.0, 1000.0]
        belowContourDisplay.OSPRayScaleArray = 'Normals'
        belowContourDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
        belowContourDisplay.SelectOrientationVectors = 'None'
        belowContourDisplay.ScaleFactor = 7461.155641368963
        belowContourDisplay.SelectScaleArray = 'None'
        belowContourDisplay.GlyphType = 'Arrow'
        belowContourDisplay.GlyphTableIndexArray = 'None'
        belowContourDisplay.GaussianRadius = 373.05778206844815
        belowContourDisplay.SetScaleArray = ['POINTS', 'Normals']
        belowContourDisplay.ScaleTransferFunction = 'PiecewiseFunction'
        belowContourDisplay.OpacityArray = ['POINTS', 'Normals']
        belowContourDisplay.OpacityTransferFunction = 'PiecewiseFunction'
        belowContourDisplay.DataAxesGrid = 'GridAxesRepresentation'
        belowContourDisplay.SelectionCellLabelFontFile = ''
        belowContourDisplay.SelectionPointLabelFontFile = ''
        belowContourDisplay.PolarAxes = 'PolarAxesRepresentation'

        # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
        belowContourDisplay.DataAxesGrid.XTitleFontFile = ''
        belowContourDisplay.DataAxesGrid.YTitleFontFile = ''
        belowContourDisplay.DataAxesGrid.ZTitleFontFile = ''
        belowContourDisplay.DataAxesGrid.XLabelFontFile = ''
        belowContourDisplay.DataAxesGrid.YLabelFontFile = ''
        belowContourDisplay.DataAxesGrid.ZLabelFontFile = ''

        # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
        belowContourDisplay.PolarAxes.Scale = [1.0, 1.0, 1000.0]
        belowContourDisplay.PolarAxes.PolarAxisTitleFontFile = ''
        belowContourDisplay.PolarAxes.PolarAxisLabelFontFile = ''
        belowContourDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
        belowContourDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

        # show data from aboveContour
        aboveContourDisplay = Show(aboveContour, renderView1)

        # trace defaults for the display properties.
        aboveContourDisplay.Representation = 'Wireframe'
        aboveContourDisplay.ColorArrayName = ['POINTS', 'density']
        aboveContourDisplay.LookupTable = densityLUT
        aboveContourDisplay.LineWidth = 2.0
        aboveContourDisplay.Scale = [1.0, 1.0, 1000.0]
        aboveContourDisplay.OSPRayScaleArray = 'Normals'
        aboveContourDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
        aboveContourDisplay.SelectOrientationVectors = 'None'
        aboveContourDisplay.ScaleFactor = 12060.679544949253
        aboveContourDisplay.SelectScaleArray = 'None'
        aboveContourDisplay.GlyphType = 'Arrow'
        aboveContourDisplay.GlyphTableIndexArray = 'None'
        aboveContourDisplay.GaussianRadius = 603.0339772474626
        aboveContourDisplay.SetScaleArray = ['POINTS', 'Normals']
        aboveContourDisplay.ScaleTransferFunction = 'PiecewiseFunction'
        aboveContourDisplay.OpacityArray = ['POINTS', 'Normals']
        aboveContourDisplay.OpacityTransferFunction = 'PiecewiseFunction'
        aboveContourDisplay.DataAxesGrid = 'GridAxesRepresentation'
        aboveContourDisplay.SelectionCellLabelFontFile = ''
        aboveContourDisplay.SelectionPointLabelFontFile = ''
        aboveContourDisplay.PolarAxes = 'PolarAxesRepresentation'

        # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
        aboveContourDisplay.DataAxesGrid.XTitleFontFile = ''
        aboveContourDisplay.DataAxesGrid.YTitleFontFile = ''
        aboveContourDisplay.DataAxesGrid.ZTitleFontFile = ''
        aboveContourDisplay.DataAxesGrid.XLabelFontFile = ''
        aboveContourDisplay.DataAxesGrid.YLabelFontFile = ''
        aboveContourDisplay.DataAxesGrid.ZLabelFontFile = ''

        # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
        aboveContourDisplay.PolarAxes.Scale = [1.0, 1.0, 1000.0]
        aboveContourDisplay.PolarAxes.PolarAxisTitleFontFile = ''
        aboveContourDisplay.PolarAxes.PolarAxisLabelFontFile = ''
        aboveContourDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
        aboveContourDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

        # show data from dallasMap
        dallasMapDisplay = Show(dallasMap, renderView1)

        # trace defaults for the display properties.
        dallasMapDisplay.Representation = 'Wireframe'
        dallasMapDisplay.AmbientColor = [0.5019607843137255,
                                         0.5019607843137255,
                                         0.5019607843137255]
        dallasMapDisplay.ColorArrayName = ['POINTS', '']
        dallasMapDisplay.DiffuseColor = [0.7137254901960784,
                                         0.7137254901960784,
                                         0.7137254901960784]
        dallasMapDisplay.MapScalars = 0
        dallasMapDisplay.InterpolateScalarsBeforeMapping = 0
        dallasMapDisplay.PointSize = 0.5
        dallasMapDisplay.LineWidth = 0.5
        dallasMapDisplay.Position = [0.0, 0.0, 305000.0]
        dallasMapDisplay.Scale = [1.0, 1.0, 1000.0]
        dallasMapDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
        dallasMapDisplay.SelectOrientationVectors = 'None'
        dallasMapDisplay.ScaleFactor = 17341.236314833164
        dallasMapDisplay.SelectScaleArray = 'None'
        dallasMapDisplay.GlyphType = 'Arrow'
        dallasMapDisplay.GlyphTableIndexArray = 'None'
        dallasMapDisplay.GaussianRadius = 867.0618157416583
        dallasMapDisplay.SetScaleArray = [None, '']
        dallasMapDisplay.ScaleTransferFunction = 'PiecewiseFunction'
        dallasMapDisplay.OpacityArray = [None, '']
        dallasMapDisplay.OpacityTransferFunction = 'PiecewiseFunction'
        dallasMapDisplay.DataAxesGrid = 'GridAxesRepresentation'
        dallasMapDisplay.SelectionCellLabelFontFile = ''
        dallasMapDisplay.SelectionPointLabelFontFile = ''
        dallasMapDisplay.PolarAxes = 'PolarAxesRepresentation'

        # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
        dallasMapDisplay.DataAxesGrid.XTitleFontFile = ''
        dallasMapDisplay.DataAxesGrid.YTitleFontFile = ''
        dallasMapDisplay.DataAxesGrid.ZTitleFontFile = ''
        dallasMapDisplay.DataAxesGrid.XLabelFontFile = ''
        dallasMapDisplay.DataAxesGrid.YLabelFontFile = ''
        dallasMapDisplay.DataAxesGrid.ZLabelFontFile = ''

        # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
        dallasMapDisplay.PolarAxes.Translation = [0.0, 0.0, 305000.0]
        dallasMapDisplay.PolarAxes.Scale = [1.0, 1.0, 1000.0]
        dallasMapDisplay.PolarAxes.PolarAxisTitleFontFile = ''
        dallasMapDisplay.PolarAxes.PolarAxisLabelFontFile = ''
        dallasMapDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
        dallasMapDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

        # setup the color legend parameters for each legend in this view

        # get color legend/bar for densityLUT in view renderView1
        densityLUTColorBar = GetScalarBar(densityLUT, renderView1)
        densityLUTColorBar.WindowLocation = 'AnyLocation'
        densityLUTColorBar.Position = [0.031037827352085365,
                                       0.6636363636363637]
        densityLUTColorBar.Title = 'density'
        densityLUTColorBar.ComponentTitle = ''
        densityLUTColorBar.TitleFontFile = ''
        densityLUTColorBar.LabelFontFile = ''
        densityLUTColorBar.ScalarBarLength = 0.3038636363636361

        # set color bar visibility
        densityLUTColorBar.Visibility = 1

        # show color legend
        densitiesDisplay.SetScalarBarVisibility(renderView1, True)

        # show color legend
        belowContourDisplay.SetScalarBarVisibility(renderView1, True)

        # show color legend
        aboveContourDisplay.SetScalarBarVisibility(renderView1, True)

        # ----------------------------------------------------------------
        # setup color maps and opacity mapes used in the visualization
        # note: the Get..() functions create a new object, if needed
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # finally, restore active source
        SetActiveSource(None)
        # ----------------------------------------------------------------

        print("finished!")

        if jpg:
            print("\tSaving .jpg file...", end=" ")

            SaveScreenshot('STKDE_4D.jpg', ImageResolution=(1080, 720))

            plt.subplots(figsize=(10.80, 7.2))

            img = mpimg.imread('STKDE_4D.png')
            plt.imshow(img)

            plt.axis('off')
            plt.show()

            print("finished!")
        if interactive:
            print("\tInteractive Window Mode ON...", end=" ")
            Interact()
            print("finished!")

    def predict(self):
        f_nodos_by_group, f_delitos_by_group = {}, {}
        for i in range(1, self.ng + 1):
            x, y, t = \
                np.array(self.pg[f'group_{i}']['t2_data']['x']), \
                np.array(self.pg[f'group_{i}']['t2_data']['y']), \
                np.array(self.pg[f'group_{i}']['t2_data']['y_day'])

            if i == 1:
                # Data base, para primer grupo
                x_training = pd.Series(
                    self.X["x"]).tolist() + pd.Series(
                    self.pg[f'group_{i}']['t1_data']['x']).tolist()
                y_training = pd.Series(
                    self.X["y"]).tolist() + pd.Series(
                    self.pg[f'group_{i}']['t1_data']['y']).tolist()
                t_training = pd.Series(
                    self.X["y_day"]).tolist() + pd.Series(
                    self.pg[f'group_{i}']['t1_data'][
                        'y_day']).tolist()

            else:
                # Data agregada del grupo anterior, para re-entrenamiento
                for j in range(1, i):
                    x_training += pd.Series(
                        self.pg[f'group_{j}']['t2_data'][
                            'x']).tolist()
                    y_training += pd.Series(
                        self.pg[f'group_{j}']['t2_data'][
                            'y']).tolist()
                    t_training += pd.Series(
                        self.pg[f'group_{j}']['t2_data'][
                            'y_day']).tolist()

            if i > 1:
                # re-entrenamos modelo
                stkde = MyKDEMultivariate(
                    [np.array(x_training),
                     np.array(y_training),
                     np.array(t_training)],
                    'ccc', bw=self.bw)

            else:
                stkde = self.kde

            stkde.resample(len(x_training))

            m = np.repeat(max(t_training), x.size)
            f_delitos = stkde.pdf([x, y, m])

            x, y, t = np.mgrid[
                      np.array(x_training).min():
                      np.array(x_training).max():100 * 1j,
                      np.array(y_training).min():
                      np.array(y_training).max():100 * 1j,
                      np.array(t_training).max():
                      np.array(t_training).max():1 * 1j
                      ]

            # pdf para nodos. checked_points filtra que los puntos estén dentro del área de dallas
            f_nodos = stkde.pdf(af.checked_points(
                np.array([x.flatten(), y.flatten(), t.flatten()])))
            f_delitos_by_group[i], f_nodos_by_group[i] = f_delitos, f_nodos
        return f_delitos_by_group, f_nodos_by_group

    def calculate_hr(self, c):
        f_delitos_by_group, f_nodos_by_group = self.predict()
        hr_by_group, ap_by_group = [], []
        for i in range(1, self.ng + 1):
            f_delitos, f_nodos = f_delitos_by_group[i], f_nodos_by_group[i]
            c = np.linspace(0, f_nodos.max(), 100)
            hits = [np.sum(f_delitos >= c[i]) for i in range(c.size)]
            area_h = [np.sum(f_nodos >= c[i]) for i in range(c.size)]
            HR = [i / len(f_delitos) for i in hits]
            area_percentaje = [i / len(f_nodos) for i in area_h]
            if i == 1:
                # caso base para el grupo 1 (o cuando se utiliza solo un grupo), sirve para función plotter
                self.hr, self.ap = HR, area_percentaje
            hr_by_group.append(HR), ap_by_group.append(area_percentaje)
        self.hr_by_group, self.ap_by_group = hr_by_group, ap_by_group
        return self.hr_by_group, self.ap_by_group

    def calculate_pai(self, c):
        pai_by_group = []
        if self.hr_by_group:
            for g in range(1, self.ng + 1):
                PAI = [float(self.hr_by_group[g - 1][i]) / float(
                    self.ap_by_group[g - 1][i]) for i in
                       range(len(self.hr_by_group[g - 1]))]
                pai_by_group.append(PAI)
                if g == 1:
                    self.pai = PAI
        else:
            f_delitos_by_group, f_nodos_by_group = self.predict()
            hr_by_group, ap_by_group = [], []
            for i in range(1, self.ng + 1):
                f_delitos, f_nodos = f_delitos_by_group[i], f_nodos_by_group[i]
                c = np.linspace(0, f_nodos.max(), 100)
                hits = [np.sum(f_delitos >= c[i]) for i in range(c.size)]
                area_h = [np.sum(f_nodos >= c[i]) for i in range(c.size)]
                HR = [i / len(f_delitos) for i in hits]
                area_percentaje = [i / len(f_nodos) for i in area_h]
                PAI = [
                    float(HR[i]) / float(area_percentaje[i]) if float(
                        HR[i]) else 0
                    for i in
                    range(len(HR))]
                if i == 1:
                    self.ap, self.hr, self.pai = area_percentaje, HR, PAI
                pai_by_group.append(PAI), ap_by_group.append(
                    area_percentaje), hr_by_group.append(HR)
                self.hr_by_group, self.ap_by_group = hr_by_group, ap_by_group
        self.pai_by_group = pai_by_group
        return self.pai_by_group, self.hr_by_group, self.ap_by_group


class RForestRegressor:
    def __init__(self, i_df=None, shps=None,
                 xc_size=100, yc_size=100, n_layers=7,
                 n_weeks=4, f_date=date(2017, 10, 31),
                 # nx=None, ny=None,
                 read_data=False, read_df=False, name='RForestRegressor'):
        """

        :param pd.DataFrame i_df: Initial Dataframe. Corresponde a los
            datos extraídos en primera instancia desde la Socrata API
        :param int xc_size: Ancho de las celdas en metros
        :param int yc_size: Largo de las celdas en metros
        :param int layers_n: Nro. de capas
        :param bool read_data: True si se desea
        :param bool read_df: True para leer el df con la
            información de las celdas
        """

        self.name = name
        self.shps = shps
        self.xc_size, self.yc_size = xc_size, yc_size
        self.n_layers = n_layers
        self.nx, self.ny, self.hx, self.hy = None, None, None, None
        self.n_weeks = n_weeks
        self.f_date = f_date
        self.weeks = []

        self.rfr = RandomForestRegressor(n_jobs=8)
        self.ap, self.hr, self.pai = None, None, None

        # m_dict = {month_name[i]: None for i in range(1, 13)}
        # self.incidents = {'Incidents': m_dict}
        # for i in range(1, n_capas + 1):
        #     self.incidents.update({f"NC Incidents_{i}": m_dict})

        if read_df and read_data:
            st = time()
            print("\nReading df pickle...", end=" ")
            self.X = pd.read_pickle('predictivehp/data/X.pkl')
            print(f"finished! ({time() - st:3.1f} sec)")
            st = time()
            print("Reading data pickle...", end=" ")
            self.data = pd.read_pickle('predictivehp/data/data.pkl')
            print(f"finished! ({time() - st:3.1f} sec)")
        else:
            self.data = i_df
            self.generate_df()
            self.assign_cells()

            self.to_pickle('data.pkl')
            self.to_pickle('X.pkl')

    @af.timer
    def generate_df(self):
        """
        La malla se genera de la esquina inf-izquierda a la esquina sup-derecha,
        partiendo con id = 0.

        n = i + j*n_x

        OBS.

        El numpy nd-array indexa de la esquina sup-izq a la inf-der
        [i:filas, j:columnas]

        El Pandas Dataframe comienza a indexar de la esquina inf-izq a la
        sup-der. [j, i]

        Hay que tener cuidado al momento de pasar desde la perspectiva
        matricial a la perspectiva normal de malla de ciudad, debido a las
        operaciones trasposición y luego up-down del nd-array entregan las
        posiciones reales para el pandas dataframe.

        :return: Pandas Dataframe con la información
        """

        print("\nGenerating dataframe...\n")

        # Creación de la malla
        print("\tCreating mgrid...")
        x_min, y_min, x_max, y_max = self.shps['streets'].total_bounds

        x_bins = abs(x_max - x_min) / self.xc_size
        y_bins = abs(y_max - y_min) / self.yc_size

        x, y = np.mgrid[x_min: x_max: x_bins * 1j, y_min: y_max: y_bins * 1j, ]

        # Creación del esqueleto del dataframe
        print("\tCreating dataframe columns...")
        i_date = self.f_date + timedelta(days=1)  # For prediction
        c_date = self.f_date
        for _ in range(self.n_weeks):
            c_date -= timedelta(days=7)
            self.weeks.append(c_date + timedelta(days=1))
        self.weeks.reverse()
        self.weeks.append(i_date)
        X_cols = pd.MultiIndex.from_product(
            [[f"Incidents_{i}" for i in range(self.n_layers + 1)], self.weeks]
        )
        self.X = pd.DataFrame(columns=X_cols)

        # Creación de los parámetros para el cálculo de los índices
        print("\tFilling df...")
        self.nx = x.shape[0] - 1
        self.ny = y.shape[1] - 1
        self.hx = (x.max() - x.min()) / self.nx
        self.hy = (y.max() - y.min()) / self.ny

        # Manejo de los puntos de incidentes para poder trabajar en (x, y)
        geometry = [Point(xy) for xy in zip(np.array(self.data[['x']]),
                                            np.array(self.data[['y']]))]
        self.data = gpd.GeoDataFrame(self.data, crs=2276, geometry=geometry)
        self.data.to_crs(epsg=3857, inplace=True)
        self.data['Cell'] = None

        # Nro. incidentes en la i-ésima capa de la celda (i, j)
        for week in self.weeks:
            print(f"\t\t{week}... ", end=' ')
            wi_date = week
            wf_date = week + timedelta(days=1)
            fil_incidents = self.data[
                (wi_date <= self.data.date) & (self.data.date <= wf_date)
                ]
            D = np.zeros((self.nx, self.ny), dtype=int)

            for _, row in fil_incidents.iterrows():
                xi, yi = row.geometry.x, row.geometry.y
                nx_i = af.n_i(xi, x.min(), self.hx)
                ny_i = af.n_i(yi, y.min(), self.hy)
                D[nx_i, ny_i] += 1

            # Actualización del pandas dataframe
            for i in range(self.n_layers + 1):
                self.X.loc[:, (f"Incidents_{i}", week)] = \
                    af.to_df_col(D) if i == 0 else \
                        af.to_df_col(af.il_neighbors(D, i))
            print('finished!')

        # Adición de las columnas 'geometry' e 'in_dallas' al df
        print("\tPreparing df for filtering...")

        self.X['geometry'] = [Point(i) for i in
                              zip(x[:-1, :-1].flatten(),
                                  y[:-1, :-1].flatten())]
        self.X['in_dallas'] = 0

        # Filtrado de celdas (llenado de la columna 'in_dallas')
        self.X = af.filter_cells(df=self.X, shp=self.shps['councils'])
        self.X.drop(columns=[('in_dallas', '')], inplace=True)

    @af.timer
    def to_pickle(self, file_name):
        """
        Genera un pickle de self.df o self.data dependiendo el nombre
        dado (data.pkl o df.pkl)

        :param str file_name: Nombre del pickle a generar
        :return: pickle de self.df o self.data
        """

        print("\nPickling dataframe...", end=" ")
        if file_name == "df.pkl":
            self.X.to_pickle(f"predictivehp/data/{file_name}")
        if file_name == "data.pkl":
            if self.data is None:
                self.generate_df()
            self.data.to_pickle(f"predictivehp/data/{file_name}")

    @af.timer
    def assign_cells(self, week=date(2017, 11, 1)):
        """
        Asigna el número de celda asociado a cada incidente en self.data

        :return:
        """
        print("\nAssigning cells...\n")
        i_date = week
        f_date = week + timedelta(days=6)
        data = self.data[
            (i_date <= self.data.date) & (self.data.date <= f_date)
            ]
        x_min, y_min, x_max, y_max = self.shps['streets'].total_bounds

        x_bins = abs(x_max - x_min) / self.xc_size
        y_bins = abs(y_max - y_min) / self.yc_size

        x, y = np.mgrid[x_min:x_max:x_bins * 1j, y_min:y_max:y_bins * 1j, ]

        nx = x.shape[0] - 1
        ny = y.shape[1] - 1

        hx = (x.max() - x.min()) / nx
        hy = (y.max() - y.min()) / ny

        for idx, inc in data.iterrows():
            xi, yi = inc.geometry.x, inc.geometry.y
            nx_i = af.n_i(xi, x.min(), hx)
            ny_i = af.n_i(yi, y.min(), hy)
            cell_idx = ny_i + ny * nx_i

            self.data.loc[idx, 'Cell'] = cell_idx

    @af.timer
    def ml_algorithm(self, f_importance=False, pickle=False):
        """
        Produce la predicción de acuerdo a los datos entregados, utilizando
        un approach de machine learning con clasificador RandomForest (rfc) y
        entrega el output asociado.

        :param f_importance: True para imprimir estadísticas
            asociadas a los features utilizados para entrenar el classifier
        :param pickle: True si se quiere generar un pickle de las estadísticas
            asociadas al entrenamiento del classifier
        """

        print("\nInitializing...")

        # Preparación del input para el algoritmo
        print("\n\tPreparing input...")

        # Jan-Sep
        x_ft = self.X.loc[
               :,
               [('Incidents_0', month_name[i]) for i in range(1, 10)] +
               [('Incidents_1', month_name[i]) for i in range(1, 10)] +
               [('Incidents_2', month_name[i]) for i in range(1, 10)] +
               [('Incidents_3', month_name[i]) for i in range(1, 10)] +
               [('Incidents_4', month_name[i]) for i in range(1, 10)] +
               [('Incidents_5', month_name[i]) for i in range(1, 10)] +
               [('Incidents_6', month_name[i]) for i in range(1, 10)] +
               [('Incidents_7', month_name[i]) for i in range(1, 10)]
               ]
        # Oct
        x_lbl = self.X.loc[
                :,
                [('Incidents_0', 'October'), ('Incidents_1', 'October'),
                 ('Incidents_2', 'October'), ('Incidents_3', 'October'),
                 ('Incidents_4', 'October'), ('Incidents_5', 'October'),
                 ('Incidents_6', 'October'), ('Incidents_7', 'October')]
                ]
        x_lbl[('Dangerous', '')] = x_lbl.T.any().astype(int)
        x_lbl = x_lbl[('Dangerous', '')]

        # Algoritmo
        print("\tRunning algorithms...")

        rfc = RandomForestClassifier(n_jobs=8)
        rfc.fit(x_ft, x_lbl.to_numpy().ravel())
        x_pred_rfc = rfc.predict(x_ft)

        if f_importance:
            cols = pd.Index(['features', 'r_importance'])
            rfc_fi_df = pd.DataFrame(columns=cols)
            rfc_fi_df['features'] = x_ft.columns.to_numpy()
            rfc_fi_df['r_importance'] = rfc.feature_importances_

            if pickle:
                rfc_fi_df.to_pickle('rfc.pkl')

        print("\n\tx\n")

        # Sirven para determinar celdas con TP/FN
        self.X[('Dangerous_Oct', '')] = x_lbl
        self.X[('Dangerous_pred_Oct', '')] = x_pred_rfc

        # Comparación para determinar si las celdas predichas son TP/FN
        self.X[('TP', '')] = 0
        self.X[('FN', '')] = 0
        self.X[('TP', '')] = np.where(
            (self.X[('Dangerous_Oct', '')] == self.X[
                ('Dangerous_pred_Oct', '')]) &
            (self.X[('Dangerous_Oct', '')] == 1),
            1,
            0
        )
        self.X[('FN', '')] = np.where(
            (self.X[('Dangerous_Oct', '')] != self.X[
                ('Dangerous_pred_Oct', '')]) &
            (self.X[('Dangerous_pred_Oct', '')] == 0),
            1,
            0
        )

        rfc_score = rfc.score(x_ft, x_lbl)
        rfc_precision = precision_score(x_lbl, x_pred_rfc)
        rfc_recall = recall_score(x_lbl, x_pred_rfc)
        print(
            f"""
    rfc score           {rfc_score:1.3f}
    rfc precision       {rfc_precision:1.3f}
    rfc recall          {rfc_recall:1.3f}
        """
        )

        print("\n\ty\n")

        y_ft = self.X.loc[
               :,
               [('Incidents_0', month_name[i]) for i in range(2, 11)] +
               [('Incidents_1', month_name[i]) for i in range(2, 11)] +
               [('Incidents_2', month_name[i]) for i in range(2, 11)] +
               [('Incidents_3', month_name[i]) for i in range(2, 11)] +
               [('Incidents_4', month_name[i]) for i in range(2, 11)] +
               [('Incidents_5', month_name[i]) for i in range(2, 11)] +
               [('Incidents_6', month_name[i]) for i in range(2, 11)] +
               [('Incidents_7', month_name[i]) for i in range(2, 11)]
               ]
        y_lbl = self.X.loc[
                :,
                [('Incidents_0', 'November'), ('Incidents_1', 'November'),
                 ('Incidents_2', 'November'), ('Incidents_3', 'November'),
                 ('Incidents_4', 'November'), ('Incidents_5', 'November'),
                 ('Incidents_6', 'November'), ('Incidents_7', 'November')]
                ]

        y_lbl[('Dangerous', '')] = y_lbl.T.any().astype(int)
        y_lbl = y_lbl[('Dangerous', '')]

        y_pred_rfc = rfc.predict(y_ft)

        rfc_score = rfc.score(y_ft, y_lbl.to_numpy().ravel())
        rfc_precision = precision_score(y_lbl, y_pred_rfc)
        rfc_recall = recall_score(y_lbl, y_pred_rfc)

        print(
            f"""
    rfc score           {rfc_score:1.3f}
    rfc precision       {rfc_precision:1.3f}
    rfc recall          {rfc_recall:1.3f}
            """
        )

        # Confusion Matrix

        # print("\tComputando matrices de confusión...", end="\n\n")
        #
        # c_matrix_x = confusion_matrix(
        #         x_lbl[('Dangerous', '')], x_predict[:, 0]
        # )
        #
        # print(c_matrix_x, end="\n\n")
        #
        # c_matrix_y = confusion_matrix(
        #         y_lbl[('Dangerous', '')], y_predict[:, 0]
        # )
        #
        # print(c_matrix_y)

    @af.timer
    def fit(self, X, y):
        """

        Parameters
        ----------
        X : pd.DataFrame
            X_train
        y : pd.DataFrame
            y_train

        Returns
        -------

        """
        print("\tFitting Model...")
        self.rfr.fit(X, y.to_numpy().ravel())

        # Sirven para determinar celdas con TP/FN
        self.X[('Dangerous_Oct', '')] = y

    def predict(self, X):
        """

        Parameters
        ----------
        X: pd.DataFrame
            X_test for prediction

        Returns
        -------

        """
        print("\tMaking predictions...")
        y_pred = self.rfr.predict(X)
        self.X[('Dangerous_pred_Oct', '')] = y_pred

        return y_pred
        # if statistics:
        #     rfr_score = self.rfr.score(X_train, X)
        #     rfr_precision = precision_score(y_test, y_pred)
        #     rfr_recall = recall_score(y_test, y_pred)
        #     print(
        #         f"""
        #         rfr score           {rfr_score:1.3f}
        #         rfr precision       {rfr_precision:1.3f}
        #         rfr recall          {rfr_recall:1.3f}
        #             """
        #     )

    def calculate_hr(self, plot=False, c=0.9):
        """
        Calculates de Hit Rate for the given Framework

        :param [float, np.ndarray] c: Threshold de confianza para
            filtrar hotspots
        :param plot: Plotea las celdas de los incidentes luego de aplicar
            un join
        :rtype: int
        :return:
        """
        if c.size == 1:
            # incidents = pd.DataFrame(self.data)
            # incidents_oct = incidents[incidents.month1 == 'October']  # 332

            data_oct = pd.DataFrame(self.data[self.data.month1 == 'October'])
            data_oct.drop(columns='geometry', inplace=True)
            # print(data_oct['Cell'])
            # print(self.df.shape)

            ans = data_oct.join(other=self.X, on='Cell', how='left')

            # ans = ans[ans[('geometry', '')].notna()]
            # print(f"Data Oct: {data_oct.shape}")
            # print(f"Ans shape: {ans.shape}")
            # print(ans[('Dangerous_pred_Oct_rfr', '')])

            # incidentsh = ans[ans[('Dangerous_pred_Oct', '')] == 1]
            incidentsh = ans[ans[('Dangerous_pred_Oct_rfr', '')] >= c[0]]
            # print(incidentsh.shape)

            hr = incidentsh.shape[0] / data_oct.shape[0]
            # print(hr)
            return hr
        else:
            A = self.X.shape[0]

            def a(x, c):
                return x[x[('Dangerous_pred_Oct_rfr', '')] >= c].shape[0]

            c_arr = c
            hr_l = []
            ap_l = []
            for c in c_arr:
                hr_l.append(self.calculate_hr(c=np.array([c])))
                ap_l.append(a(self.X, c) / A)
            self.hr = np.array(hr_l)
            self.ap = np.array(ap_l)
        plt.savefig(f"frheatmap.pdf", format='pdf')

    def calculate_pai(self, c=0.9):
        """
        Calcula el Predictive Accuracy Index (PAI)

        :param [float, np.ndarray] c:
        :return:
        """

        # data_oct = pd.DataFrame(self.data[self.data.month1 == 'October'])
        # data_oct.drop(columns='geometry', inplace=True)

        # ans = data_oct.join(other=fwork.df, on='Cell', how='left')
        # ans = self.df[self.df[('geometry', '')].notna()]

        # a = self.df[self.df[('Dangerous_pred_Oct', '')] == 1].shape[0]
        # a = self.df[self.df[('Dangerous_pred_Oct_rfr', '')] >= c].shape[0]
        def a(x, c):
            return x[x[('Dangerous_pred_Oct_rfr', '')] >= c].shape[0]

        A = self.X.shape[0]  # Celdas en Dallas
        if c.size == 1:
            hr = self.calculate_hr(c=c)
            ap = a(self.X, c) / A

            # print(f"a: {a} cells    A: {A} cells")
            # print(f"Area Percentage: {ap:1.3f}")
            # print(f"PAI: {hr / ap:1.3f}")

            return hr / ap
        else:
            c_arr = c
            hr_l = []
            ap_l = []
            pai_l = []
            for c in c_arr:
                hr = self.calculate_hr(c=np.array([c]))
                ap = a(self.X, c) / A
                hr_l.append(hr)
                ap_l.append(ap)
                pai_l.append(hr / ap)
            self.hr = np.array(hr_l)
            self.ap = np.array(ap_l)
            self.pai = np.array(pai_l)

    def heatmap(self, c=0):
        """

        :param float c: Threshold a partir del cual se consideran los
            incidentes

        :return:
        """
        # Datos Oct luego de aplicar el rfr
        ans = self.X[[('geometry', ''), ('Dangerous_pred_Oct_rfr', '')]]
        ans = gpd.GeoDataFrame(ans)
        ans = ans[ans[('Dangerous_pred_Oct_rfr', '')] >= c]
        d_streets = self.shps['streets']

        print("\tRendering Plot...")
        fig, ax = plt.subplots(figsize=(20, 15))
        d_streets.plot(ax=ax,
                       alpha=0.4,
                       color="dimgrey",
                       label="Streets")
        ans.plot(ax=ax,
                 column=('Dangerous_pred_Oct_rfr', ''),
                 cmap='jet')

        # Background
        ax.set_axis_off()
        fig.set_facecolor('black')
        plt.show()

    def plot_statistics(self, n=500):
        """

        :return:
        """
        c_arr = np.linspace(0, 1, n)

        def a(x, c): return x[x[('Dangerous_pred_Oct_rfr', '')] >= c].shape[0]

        ap_l, hr_l, pai_l = [], [], []
        for c in c_arr:
            A = self.X.shape[0]  # Celdas en Dallas
            ap = a(self.X, c) / A  # in [0.00, 0.25]
            hr = self.calculate_hr(c=c)
            pai = hr / ap

            ap_l.append(ap), hr_l.append(hr), pai_l.append(pai)

        ap_arr = np.array(ap_l)
        hr_arr, pai_arr = np.array(hr_l), np.array(pai_l)

        af.lineplot(
            x=c_arr, y=ap_arr,
            x_label='c',
            y_label='Area Percentage',
        )
        plt.savefig(
            fname='Statistics_ml_model/ap vs c',
            facecolor='black',
            edgecolor='black'
        )
        plt.show()
        af.lineplot(
            x=c_arr, y=hr_arr,
            x_label='c',
            y_label='Hit Rate',
        )
        plt.savefig(
            fname='Statistics_ml_model/HR vs c',
            facecolor='black',
            edgecolor='black'
        )
        plt.show()
        af.lineplot(
            x=c_arr, y=pai_arr,
            x_label='c',
            y_label='PAI',
        )
        plt.savefig(
            fname='Statistics_ml_model/PAI vs c',
            facecolor='black',
            edgecolor='black'
        )
        plt.show()
        af.lineplot(
            x=ap_arr, y=pai_arr,
            x_label='Area Percentage',
            y_label='PAI',
        )
        plt.savefig(
            fname='Statistics_ml_model/PAI vs ap',
            facecolor='black',
            edgecolor='black'
        )
        plt.show()
        af.lineplot(
            x=ap_arr, y=hr_arr,
            x_label='Area Percentage',
            y_label='Hit Rate',
        )
        plt.savefig(
            fname='Statistics_ml_model/HR vs ap',
            facecolor='black',
            edgecolor='black'
        )
        plt.show()

    @af.timer
    def plot_incidents(self, i_type="real", month="October"):
        """
        Plotea los incidentes almacenados en self.data en el mes dado.
        Asegurarse que al momento de realizar el ploteo, ya se haya
        hecho un llamado al método ml_algorithm() para identificar los
        incidentes TP y FN

        :param str i_type: Tipo de incidente a plotear (e.g. TP, FN, TP & FN)
        :param str month: String con el nombre del mes que se predijo
            con ml_algorithm()
        :return:
        """

        print(f"\nPlotting {month} Incidents...")
        print("\tFiltering incidents...")

        tp_data, fn_data, data = None, None, None

        if i_type == "TP & FN":
            data = gpd.GeoDataFrame(self.X)
            tp_data = data[self.X.TP == 1]
            fn_data = data[self.X.FN == 1]
        if i_type == "TP":
            data = gpd.GeoDataFrame(self.X)
            tp_data = self.X[self.X.TP == 1]
        if i_type == "FN":
            data = gpd.GeoDataFrame(self.X)
            fn_data = self.X[self.X.FN == 1]
        if i_type == "real":
            data = self.data[self.data.month1 == month]
            n_incidents = data.shape[0]
            print(f"\tNumber of Incidents in {month}: {n_incidents}")
        if i_type == "pred":
            data = gpd.GeoDataFrame(self.X)
            all_hp = data[self.X[('Dangerous_pred_Oct', '')] == 1]

        print("\tReading shapefile...")
        d_streets = gpd.GeoDataFrame.from_file(
            "../Data/Streets/streets.shp")
        d_streets.to_crs(epsg=3857, inplace=True)

        print("\tRendering Plot...")
        fig, ax = plt.subplots(figsize=(20, 15))

        d_streets.plot(ax=ax,
                       alpha=0.4,
                       color="dimgrey",
                       zorder=2,
                       label="Streets")

        if i_type == 'pred':
            all_hp.plot(
                ax=ax,
                markersize=2.5,
                color='y',
                marker='o',
                zorder=3,
                label="TP Incidents"
            )
        if i_type == "real":
            data.plot(
                ax=ax,
                markersize=10,
                color='darkorange',
                marker='o',
                zorder=3,
                label="TP Incidents"
            )
        if i_type == "TP":
            tp_data.plot(
                ax=ax,
                markersize=2.5,
                color='red',
                marker='o',
                zorder=3,
                label="TP Incidents"
            )
        if i_type == "FN":
            fn_data.plot(
                ax=ax,
                markersize=2.5,
                color='blue',
                marker='o',
                zorder=3,
                label="FN Incidents"
            )
        if i_type == "TP & FN":
            tp_data.plot(
                ax=ax,
                markersize=2.5,
                color='red',
                marker='o',
                zorder=3,
                label="TP Incidents"
            )
            fn_data.plot(
                ax=ax,
                markersize=2.5,
                color='blue',
                marker='o',
                zorder=3,
                label="FN Incidents"
            )

        # Legends
        handles = [
            Line2D([], [],
                   marker='o',
                   color='darkorange',
                   label='Incident',
                   linestyle='None'),
            Line2D([], [],
                   marker='o',
                   color='red',
                   label='TP Incident',
                   linestyle='None'),
            Line2D([], [],
                   marker='o',
                   color="blue",
                   label="FN Incident",
                   linestyle='None'),
            Line2D([], [],
                   marker='o',
                   color='y',
                   label='Predicted Incidents',
                   linestyle='None')
        ]

        plt.legend(loc="best",
                   bbox_to_anchor=(0.1, 0.7),
                   frameon=False,
                   fontsize=13.5,
                   handles=handles)

        legends = ax.get_legend()
        for text in legends.get_texts():
            text.set_color('white')

        # Background
        ax.set_axis_off()
        fig.set_facecolor('black')
        plt.show()
        plt.close()

    @af.timer
    def plot_hotspots(self):
        """
        Utiliza el método estático asociado para plotear los hotspots
        con los datos ya cargados del framework.

        :return:
        """

        data = self.X[[('geometry', ''),
                       ('Dangerous_Oct', ''),
                       ('Dangerous_pred_Oct', '')]]

        # Quitamos el nivel ''
        data = data.T.reset_index(level=1, drop=True).T

        # Creamos el df para los datos reales (1) y predichos (2).
        data1 = data[['geometry', 'Dangerous_Oct']]
        data2 = data[['geometry', 'Dangerous_pred_Oct']]

        # Filtramos las celdas detectadas como Dangerous para reducir los
        # tiempos de cómputo.
        data1_d = data1[data1['Dangerous_Oct'] == 1]
        data1_nd = data1[data1['Dangerous_Oct'] == 0]
        geodata1_d = gpd.GeoDataFrame(data1_d)
        geodata1_nd = gpd.GeoDataFrame(data1_nd)

        data2_d = data2[data2['Dangerous_pred_Oct'] == 1]
        data2_nd = data2[data2['Dangerous_pred_Oct'] == 0]
        geodata2_d = gpd.GeoDataFrame(data2_d)
        geodata2_nd = gpd.GeoDataFrame(data2_nd)

        self.plot_hotspots_s(geodata1_d, geodata1_nd)
        self.plot_hotspots_s(geodata2_d, geodata2_nd)

    @staticmethod
    def plot_ft_imp_1():
        """
        Barplot de las importancias relativas de los datos agrupados
        por meses.

        :return:
        """

        df = pd.read_pickle('rfc.pkl')
        data = [aux_df.sum()['r_importance'] for aux_df in
                [df[df['features'].isin(
                    [(f'Incidents_{i}', month_name[j]) for i in range(0, 8)])]
                 for j in range(1, 10)]]
        index = [month_name[i] for i in range(1, 10)]

        ax = pd.DataFrame(data=data, index=index, columns=['r_importance']) \
            .plot.bar(y='r_importance', color='black', width=0.25, rot=0,
                      legend=None)

        for i in range(0, 9):
            plt.text(x=i - 0.3, y=data[i] + 0.02 * max(data),
                     s=f'{data[i]:.3f}')

        plt.xlabel("Features",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold',
                             'family': 'serif'},
                   labelpad=10
                   )
        plt.ylabel("Relative Importance",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold',
                             'family': 'serif'},
                   labelpad=7.5
                   )
        plt.xticks(ticks=[i for i in range(0, 9)],
                   labels=[f'{month_name[i]:.3s}' for i in range(1, 10)])
        plt.tick_params(axis='both', length=0, pad=8.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['bottom'].set_color('lightgray')
        ax.spines['left'].set_color('lightgray')

        plt.show()

    @staticmethod
    def plot_ft_imp_2():
        """
        Barplot de las importancias relativas de los datos agrupados en
        capas.

        :return:
        """

        df = pd.read_pickle('rfc.pkl')
        data = [aux_df.sum()['r_importance'] for aux_df in
                [df[df['features'].isin(
                    [(f'Incidents_{i}', month_name[j]) for j in range(1, 10)])]
                 for i in range(0, 8)]]
        index = [i for i in range(0, 8)]

        ax = pd.DataFrame(data=data, index=index, columns=['r_importance']) \
            .plot.bar(y='r_importance', color='black', width=0.25, rot=0,
                      legend=None)

        for i in range(0, 8):
            plt.text(x=i - 0.3, y=data[i] + 0.02 * max(data),
                     s=f'{data[i]:.3f}')

        plt.xlabel("Layers",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold',
                             'family': 'serif'},
                   labelpad=10
                   )
        plt.ylabel("Relative Importance",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold',
                             'family': 'serif'},
                   labelpad=7.5
                   )
        plt.xticks(ticks=[i for i in range(0, 8)],
                   labels=[f'{i}' for i in range(0, 8)])
        plt.tick_params(axis='both', length=0, pad=8.5)  # Hide tick lines

        ax.spines['top'].set_visible(False)  # Hide frame
        ax.spines['right'].set_visible(False)

        ax.spines['bottom'].set_color('lightgray')  # Frame color
        ax.spines['left'].set_color('lightgray')

        plt.show()

    @staticmethod
    def plot_ft_imp_3():
        """
        Heatmap que combina las estadísticas de las importancias
        relativas de los datos agrupados por meses y capas.

        :return:
        """

        df = pd.read_pickle('rfc.pkl')
        df.set_index(keys='features', drop=True, inplace=True)

        data = df.to_numpy().reshape(8, 9).T
        columns = [i for i in range(0, 8)]
        index = [f'{month_name[i]:.3s}' for i in range(1, 10)]

        df = pd.DataFrame(data=data, index=index, columns=columns)

        sns.heatmap(data=df, annot=True, annot_kws={"fontsize": 9})

        plt.xlabel("Layers",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold',
                             'family': 'serif'},
                   labelpad=10
                   )
        plt.ylabel("Months",
                   fontdict={'fontsize': 12.5,
                             'fontweight': 'bold',
                             'family': 'serif'},
                   labelpad=7.5
                   )

        plt.tick_params(axis='both', length=0, pad=8.5)
        plt.yticks(rotation=0)

        plt.show()

    @staticmethod
    def plot_hotspots_s(geodata_d, geodata_nd):
        """
            Plotea las celdas de Dallas reconocidas como peligrosas o
            no-peligrosas de acuerdo al algoritmo.

            :param gpd.GeoDataFrame geodata_d: gdf con los puntos de celdas
                peligrosas
            :param gpd.GeoDataFrame geodata_nd: gdf con los puntos de celdas
                no-peligrosas
            """

        print('Reading shapefiles...')
        d_streets = gpd.GeoDataFrame.from_file(
            filename='../Data/Streets/streets.shp'
        )
        d_districts = gpd.GeoDataFrame.from_file(
            filename='../Data/Councils/councils.shp'
        )
        d_streets.to_crs(epsg=3857, inplace=True)
        d_districts.to_crs(epsg=3857, inplace=True)

        fig, ax = plt.subplots(figsize=(20, 15))
        ax.set_facecolor('xkcd:black')

        for district, data in d_districts.groupby('DISTRICT'):
            data.plot(ax=ax,
                      color=prm.d_colors[district],
                      linewidth=2.5,
                      edgecolor="black")

        handles = [Line2D([], [], marker='o', color='red',
                          label='Dangerous Cell',
                          linestyle='None'),
                   Line2D([], [], marker='o', color="blue",
                          label="Non-Dangerous Cell",
                          linestyle='None')]

        d_streets.plot(ax=ax,
                       alpha=0.4,
                       color="dimgrey",
                       zorder=2,
                       label="Streets")
        geodata_nd.plot(ax=ax,
                        markersize=10,
                        color='blue',
                        marker='o',
                        zorder=3,
                        label="Incidents")
        geodata_d.plot(ax=ax,
                       markersize=10,
                       color='red',
                       marker='o',
                       zorder=3,
                       label="Incidents")

        plt.legend(loc="best",
                   bbox_to_anchor=(0.1, 0.7),
                   frameon=False,
                   fontsize=13.5,
                   handles=handles)

        legends = ax.get_legend()
        for text in legends.get_texts():
            text.set_color('white')

        ax.set_axis_off()
        fig.set_facecolor('black')
        plt.show()
        plt.close()

    @af.timer
    def plot_joined_cells(self):
        """

        :return:
        """

        data_oct = pd.DataFrame(self.data[self.data.month1 == 'October'])
        data_oct.drop(columns='geometry', inplace=True)

        ans = data_oct.join(other=self.X, on='Cell', how='left')
        ans = ans[ans[('geometry', '')].notna()]

        gpd_ans = gpd.GeoDataFrame(ans, geometry=ans[('geometry', '')])

        d_streets = gpd.GeoDataFrame.from_file(
            "../Data/Streets/streets.shp")
        d_streets.to_crs(epsg=3857, inplace=True)

        fig, ax = plt.subplots(figsize=(20, 15))

        d_streets.plot(ax=ax,
                       alpha=0.4,
                       color="dimgrey",
                       zorder=2,
                       label="Streets")

        gpd_ans.plot(
            ax=ax,
            markersize=10,
            color='red',
            marker='o',
            zorder=3,
            label="Joined Incidents"
        )

        handles = [
            Line2D([], [],
                   marker='o',
                   color='red',
                   label='Joined Incidents',
                   linestyle='None'),
        ]

        plt.legend(loc="best",
                   bbox_to_anchor=(0.1, 0.7),
                   frameon=False,
                   fontsize=13.5,
                   handles=handles)

        legends = ax.get_legend()
        for text in legends.get_texts():
            text.set_color('white')

        # Background
        ax.set_axis_off()
        fig.set_facecolor('black')
        plt.show()
        plt.close()


class ProMap:

    def __init__(self, bw, i_df=None, n_datos=3600, read_density=False,
                 hx=100, hy=100,
                 radio=None, ventana_dias=7, tiempo_entrenamiento=None,
                 month=10,
                 km2=1_000, name='Promap', shps=None):

        # DATA
        self.data = i_df
        self.n = n_datos
        self.month = month
        self.X, self.Y = None, None
        self.shps = shps

        # MAP
        self.hx, self.hy, self.km2 = hx, hy, km2
        self.x_min, self.y_min, self.x_max, self.y_max = self.shps[
            'streets'].total_bounds
        self.bw_x, self.bw_y = bw[0], bw[1]
        self.bw_t = bw[2] if not tiempo_entrenamiento else tiempo_entrenamiento
        self.radio = radio
        self.bins_x = int(round(abs(self.x_max - self.x_min) / self.hx))
        self.bins_y = int(round(abs(self.y_max - self.y_min) / self.hy))

        # MODEL
        self.name = name
        self.ventana_dias = ventana_dias
        self.matriz_con_densidades = np.zeros((self.bins_x, self.bins_y))
        self.training_matrix = np.zeros((self.bins_x, self.bins_y))
        self.testing_matrix = np.zeros((self.bins_x, self.bins_y))
        self.hr, self.pai, self.ap = None, None, None

        print('-' * 100)
        print('\t\t', self.name)
        print(af.print_mes(self.month, self.month + 1, self.ventana_dias))

        self.generar_df()

        if read_density:
            self.matriz_con_densidades = np.load(
                'predictivehp/data/density_matrix.npy')
        else:
            self.predict()
        print('-' * 100)

    def generar_df(self):

        """
        Genera un dataframe en base a los x{min, max} y{min, max}.
        Recordar que cada nodo del dataframe representa el centro de cada
        celda en la malla del mapa.

        """

        if len(self.data) >= self.n:
            print(f'\nEligiendo {self.n} datos...')
            self.data = self.data.sample(n=self.n,
                                         replace=False,
                                         random_state=250499)
            self.data.sort_values(by=['date'], inplace=True)
            self.data.reset_index(drop=True, inplace=True)

        print("\nGenerando dataframe...\n")

        geometry = [Point(xy) for xy in zip(
            np.array(self.data[['x']]),
            np.array(self.data[['y']]))
                    ]

        geo_data = gpd.GeoDataFrame(self.data,  # gdf de incidentes
                                    crs=2276,
                                    geometry=geometry)

        geo_data.to_crs(epsg=3857, inplace=True)

        self.data['x_point'] = geo_data['geometry'].x
        self.data['y_point'] = geo_data['geometry'].y

        # División en training y testing data

        self.X = self.data[self.data["date"].apply(lambda x: x.month) <= \
                           self.month]
        self.Y = self.data[self.data["date"].apply(lambda x: x.month) > \
                           self.month]

        print(f'\thx: {self.hx} mts, hy: {self.hy} mts')
        print(
            f'\tbw.x: {self.bw_x} mts, bw.y: {self.bw_y} mts, bw.t: {self.bw_t} dias')

        self.xx, self.yy = np.mgrid[
                           self.x_min + self.hx / 2:self.x_max - self.hx / 2:self.bins_x * 1j,
                           self.y_min + self.hy / 2:self.y_max - self.hy / 2:self.bins_y * 1j
                           ]

        self.total_dias_training = self.X['y_day'].max()

    def predict(self):

        """""
        Calcula los scores de la malla en base a los delitos del self.data
        :return np.mgrid

        """""

        print('\nEstimando densidades...')
        print(
            f'\n\tNº de datos para entrenar el modelo: {len(self.X)}')
        print(
            f'\tNº de días usados para entrenar el modelo: {self.total_dias_training}')
        print(
            f'\tNº de datos para testear el modelo: {len(self.Y)}')

        if not self.radio:
            ancho_x = af.radio_pintar(self.hx, self.bw_x)
            ancho_y = af.radio_pintar(self.hy, self.bw_y)
        else:
            ancho_x = af.radio_pintar(self.hx, self.radio)
            ancho_y = af.radio_pintar(self.hy, self.radio)

        for k in range(len(self.X)):
            x, y, t = self.X['x_point'][k], self.X['y_point'][k], \
                      self.X['y_day'][k]
            x_in_matrix, y_in_matrix = af.find_position(self.xx, self.yy, x, y,
                                                        self.hx, self.hy)
            x_left, x_right = af.limites_x(ancho_x, x_in_matrix, self.xx)
            y_abajo, y_up = af.limites_y(ancho_y, y_in_matrix, self.yy)

            for i in range(x_left, x_right + 1):
                for j in range(y_abajo, y_up):
                    elem_x = self.xx[i][0]
                    elem_y = self.yy[0][j]
                    time_weight = 1 / af.n_semanas(self.total_dias_training, t)

                    if af.linear_distance(elem_x, x) > self.bw_x or \
                            af.linear_distance(elem_y, y) > self.bw_y:
                        cell_weight = 0

                    else:
                        cell_weight = 1 / af.cells_distance(x, y, elem_x,
                                                            elem_y,
                                                            self.hx,
                                                            self.hy)

                    self.matriz_con_densidades[i][
                        j] += time_weight * cell_weight

        self.matriz_con_densidades = self.matriz_con_densidades / self.matriz_con_densidades.max()

        print('\nGuardando datos...')
        np.save('predictivehp/data/density_matrix.npy',
                self.matriz_con_densidades)

    # borrar
    def load_train_matrix(self):

        """
        Ubica los delitos en la matriz de training

        :param None
        :return None
        """

        delitos_agregados = 0

        for index, row in self.X.iterrows():
            x, y, t = row['x_point'], row['y_point'], row['y_day']

            if t >= (self.total_dias_training - self.bw_t):
                x_pos, y_pos = af.find_position(self.xx, self.yy, x, y,
                                                self.hx,
                                                self.hy)
                self.training_matrix[x_pos][y_pos] += 1
                delitos_agregados += 1

    def load_test_matrix(self, ventana_dias):

        """
        Ubica los delitos en la matriz de testeo

        :param None
        :return: None
        """
        delitos_agregados = 0

        for index, row in self.Y.iterrows():
            x, y, t = row['x_point'], row['y_point'], row['y_day']

            if t <= (self.total_dias_training + ventana_dias):
                x_pos, y_pos = af.find_position(self.xx, self.yy, x, y,
                                                self.hx,
                                                self.hy)
                self.testing_matrix[x_pos][y_pos] += 1
                delitos_agregados += 1
            else:
                break

    def calculate_hr(self, c):

        # 1. Solo considera las celdas que son mayor a un K
        #     Esto me entrega una matriz con True/False (Matriz A)
        # 2. La matriz de True/False la multiplico con una matriz que tiene la
        # cantidad de delitos que hay por cada celda (Matriz B)
        # 3. Al multiplicar A * B obtengo una matriz C donde todos los False en
        # A son 0 en B. Todos los True en A mantienen su valor en B
        # 4. Contar cuantos delitos quedaron luego de haber pasado este proceso.
        #
        # Se espera que los valores de la lista vayan disminuyendo a medida que el valor de K aumenta

        self.load_train_matrix()
        self.load_test_matrix(self.ventana_dias)

        hits_n = []

        for i in range(c.size):
            hits_n.append(
                np.sum(
                    (self.matriz_con_densidades >= c[
                        i]) * self.testing_matrix))

        # """
        # 1. Solo considera las celdas que son mayor a un K
        #     Esto me entrega una matriz con True/False (Matriz A)
        # 2. Contar cuantos celdas quedaron luego de haber pasado este proceso.
        #
        # Se espera que los valores de la lista vayan disminuyendo a medida que el valor de K aumenta
        # """

        area_hits = []

        for i in range(c.size):
            area_hits.append(
                np.count_nonzero(self.matriz_con_densidades >= c[i]))

        n_delitos_testing = np.sum(self.testing_matrix)

        self.hr = [i / n_delitos_testing for i in hits_n]

        n_celdas = af.calcular_celdas(self.hx, self.hy, self.km2)

        self.ap = [1 if j > 1 else j for j in [i / n_celdas for
                                               i in area_hits]]

    def calculate_pai(self, c):

        if not self.hr:
            self.calculate_hr(c)

        self.pai = [
            0 if float(self.ap[i]) == 0
            else float(self.hr[i]) / float(self.ap[i])
            for i in range(len(self.ap))]

    def heatmap(self, c=0,
                nombre_grafico='Predictive Crime Map - Dallas (Method: Promap)'):

        """
        Mostrar un heatmap de una matriz de riesgo.

        :param matriz: np.mgrid
        :return None
        """

        matriz = np.where(self.matriz_con_densidades >= c,
                          self.matriz_con_densidades, 0)

        dallas = gpd.read_file('predictivehp/data/streets.shp')
        dallas.crs = 2276
        dallas.to_crs(epsg=3857, inplace=True)

        fig, ax = plt.subplots(figsize=(15, 12))
        ax.set_facecolor('xkcd:black')

        plt.title(nombre_grafico)
        plt.imshow(np.flipud(matriz.T),
                   extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                   cmap='jet',
                   # vmin=0, vmax=1
                   )

        dallas.plot(ax=ax,
                    alpha=.3,  # Ancho de las calles
                    color="gray")

        plt.colorbar()
        plt.show()

    def plot_incident(self, matriz, nombre_grafico):

        """
        Mostrar un heatmap de una matriz de riesgo.

        :param matriz: np.mgrid
        :return None
        """

        dallas = gpd.read_file('predictivehp/data/streets.shp')
        dallas.crs = 2276
        dallas.to_crs(epsg=3857, inplace=True)

        fig, ax = plt.subplots(figsize=(15, 12))
        ax.set_facecolor('xkcd:black')

        plt.title(nombre_grafico)
        plt.imshow(np.flipud(matriz.T),
                   extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                   cmap='gist_heat',
                   vmin=0, vmax=1)

        dallas.plot(ax=ax,
                    alpha=.1,  # Ancho de las calles
                    # color="gray")
                    )

        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    pass
