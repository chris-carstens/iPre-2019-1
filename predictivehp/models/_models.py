from calendar import month_name
from datetime import date, timedelta, datetime

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.nonparametric.kernel_density as kd
from matplotlib.lines import Line2D
from pyevtk.hl import gridToVTK
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor

import predictivehp.utils._aux_functions as af

settings = kd.EstimatorSettings(efficient=True, n_jobs=8)


class MyKDEMultivariate(kd.KDEMultivariate):
    def resample(self, size, shp):
        """

        Parameters
        ----------
        size : int

        Returns
        -------
        np.hstack

        """
        # print("\nResampling...", end=" ")

        n, d = self.data.shape
        indices = np.random.randint(0, n, size)

        cov = np.diag(self.bw) ** 2
        means = self.data[indices, :]
        norm = np.random.multivariate_normal(np.zeros(d), cov, size)

        # simulated and checked points
        s_points = np.transpose(means + norm)
        c_points = af.checked_points(s_points, shp)

        # print(f"\n{size - c_points.shape[1]} invalid points found")

        if size == c_points.shape[1]:
            # print("\nfinished!")
            return s_points

        a_points = self.resample(size - c_points.shape[1], shp)

        return np.hstack((c_points, a_points))


class STKDE:
    def __init__(self,
                 shps=None, bw=None, sample_number=3600,
                 start_prediction=date(2017, 11, 1),
                 length_prediction=7, name="STKDE"):
        """
        Parameters
        ----------
        year: string
          Database year
        bw: np.array
          bandwidth for x, y, t
        sample_number: int
          Número de muestras de la base de datos
        number_of_groups:
          Número de grupos distintos a generar para predecir por separado.
          Si no se desea dividir en grupos, por defecto es 1.
        start_prediction : date
          Fecha de comienzo en la ventana temporal a predecir
        length_prediction : int
          Número de días de ventana a predecir
        name : str
          Nombre predeterminado del modelo
        shps : gpd.GeoDataFrame
          GeoDataFrame que contiene información sobre la ciudad de Dallas
        """

        self.name, self.sn, self.bw = name, sample_number, bw
        self.shps = shps
        self.start_prediction = start_prediction
        self.lp = length_prediction

        self.hr, self.ap, self.pai = None, None, None
        self.f_delitos, self.f_nodos = None, None
        self.df = None

        self.x_min, self.y_min, self.x_max, self.y_max = self.shps[
            'streets'].total_bounds
        # training data 3000
        # testing data  600
        # print('-' * 30)
        # print('\t\tSTKDE')
        # print(af.print_mes(self.X_months, self.X_months + 1, self.wd))

        # print('-' * 30)

    def set_parameters(self, bw):
        """

        Parameters
        ----------
        bw: np.array
            Bandwith for x,y,t

        Returns
        -------

        """
        self.bw = bw
        # Reentrenamos el modelo con nuevo bw
        if self.df is not None:
            self.fit(self.df, self.X_train, self.y, self.pg)

    def print_parameters(self):
        """

        Returns
        -------

        """
        print('STKDE Hyperparameters')
        if self.bw is not None:
            print(f'bandwith x: {self.bw[0]} mts.')
            print(f'bandwith y: {self.bw[1]} mts.')
            print(f'bandwith t: {self.bw[2]} days\n')
        else:
            print(
                "No bandwith set. The model will automatically calculate bandwith after fit.\n")

    def fit(self, X, X_t):
        """
        Parameters
        ----------
        X : pd.DataFrame
          Training data.
        X_t : pd.DataFrame
          Test data.
        predict_groups: list
          Lista con los datos separados por grupos y sus correspondientes
          ventanas.
        """
        self.X_train, self.X_test = X, X_t

        # print("\nBuilding KDE...")
        self.kde = MyKDEMultivariate(
            [np.array(self.X_train[['x']]),
             np.array(self.X_train[['y']]),
             np.array(self.X_train[['y_day']])],
            'ccc', bw=self.bw)

        self.bw = self.kde.bw

    def predict(self):
        """
        Returns
        -------
        f_delitos_by_group : dict
          Diccionario con los valores de la función densidad evaluada en
          los puntos a predecir con llave el grupo
        f_nodos_by_group : dict
          Diccionario con los valores de la función densidad evaluada en
          la malla original con llave el grupo
        """
        if self.f_delitos is not None:
            return self.f_delitos, self.f_nodos
        stkde = self.kde

        # Data base, para primer grupo
        x_training = pd.Series(
            self.X_train["x"]).tolist()
        y_training = pd.Series(
            self.X_train["y"]).tolist()
        t_training = pd.Series(
            self.X_train["y_day"]).tolist()

        self.predicted_sim = stkde.resample(len(pd.Series(
            self.X_test["x"]).tolist()), self.shps["councils"])

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
            np.array([x.flatten(), y.flatten(), t.flatten()]),
            self.shps['councils']))

        x, y, t = \
            np.array(self.X_test['x']), \
            np.array(self.X_test['y']), \
            np.array(self.X_test['y_day'])
        ti = np.repeat(max(t_training), x.size)
        f_delitos = stkde.pdf(af.checked_points(
            np.array([x.flatten(), y.flatten(), ti.flatten()]),
            self.shps["councils"]))

        f_max = max([f_nodos.max(), f_delitos.max()])

        # normalizar
        f_delitos = f_delitos / f_max
        f_nodos = f_nodos / f_max

        self.f_delitos, self.f_nodos = f_delitos, f_nodos
        return self.f_delitos, self.f_nodos

    def score(self, x, y, t):
        """

        Parameters
        ----------
        x : float
        y : float
        t : float

        Returns
        -------
        score_pdf : float
                    Valor de la función densidad de
                    la predicción evaluada en (x,y,t)
        """
        score_pdf = self.kde.pdf(np.array([x, y, t]))
        # print(f"STKDE pdf score: {score_pdf}\n")
        return score_pdf

    def heatmap(self, c=None, show_score=True, incidences=False,
                savefig=False, fname='', bins=100, ti=100, **kwargs):
        """
        Parameters
        ----------
        bins : int
          Número de bins para heatmap
        ti : int
          Tiempo fijo para evaluar densidad en la predicción
        """
        # print("\nPlotting Heatmap...")

        dallas = self.shps['streets']

        fig, ax = plt.subplots(figsize=[prm.f_size[0]] * 2)
        dallas.plot(ax=ax, alpha=.4, color="gray", zorder=1)

        x, y = np.mgrid[
               self.x_min:
               self.x_max:bins * 1j,
               self.y_min:
               self.y_max:bins * 1j
               ]
        x_f = x.flatten()
        y_f = y.flatten()
        z = self.kde.pdf(np.vstack([x_f, y_f,
                                    ti * np.ones(x.size)]))
        # z.reshape(x_shape)
        # Normalizar
        z = z / z.max()

        if c is None:
            z_plot = z
        elif type(c) == float or type(c) == int:
            z_plot = z > c
        elif type(c) == list or type(c) == np.ndarray:
            c = np.array(c).flatten()
            c = c[c > 0]
            c = c[c < 1]
            c = np.unique(c)
            c = np.sort(c)
            z_plot = np.zeros(z.size)
            for i in c:
                z_plot[z > i] += 1
            z_plot = z_plot / np.max(z_plot)

        heatmap = plt.pcolormesh(x, y,
                                 z_plot.reshape(x.shape),
                                 shading='gouraud',
                                 alpha=.2,
                                 cmap='jet',
                                 zorder=2,
                                 )

        if not show_score:
            cbar = plt.colorbar(heatmap,
                                ax=ax,
                                shrink=.5,
                                aspect=10)
            cbar.solids.set(alpha=1)

            # norm = mpl.colors.Normalize(vmin=0, vmax=1)
            # cmap = mpl.cm.jet
            # mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            # c_bar = fig.colorbar(mappable, ax=ax,
            #                     fraction=0.15,
            #                     shrink=0.5,
            #                     aspect=21.5)
            # c_bar.ax.set_ylabel('Danger Score')

        if incidences:
            # print("\nPlotting Spatial Pattern of incidents...", sep="\n\n")

            geometry = [Point(xy) for xy in zip(
                np.array(self.X_test[['x']]),
                np.array(self.X_test[['y']]))]
            geo_df = gpd.GeoDataFrame(self.X_test,
                                      crs=dallas.crs,
                                      geometry=geometry)

            # print("\tPlotting Incidents...", end=" ")

            geo_df.plot(ax=ax,
                        markersize=17.5,
                        color='red',
                        marker='o',
                        zorder=3,
                        label="Incidents")

            # print("finished!")

        plt.title('STKDE')
        plt.legend()

        ax.set_axis_off()
        plt.tight_layout()
        if savefig:
            plt.savefig(fname, **kwargs)

        print(z_plot)
        plt.show()

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

    def spatial_pattern(self):
        """
        Spatial pattern of incidents
        pdf: True si se desea guardar el plot en formato pdf
        """

        self.predict()
        # print("\nPlotting Spatial Pattern of incidents...", sep="\n\n")

        # print("\tReading shapefiles...", end=" ")
        dallas = self.shps['streets']
        # print("finished!")

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_facecolor('xkcd:black')

        # US Survey Foot: 0.3048 m
        # print("\n", f"EPSG: {dallas.crs['init'].split(':')[1]}")  # 2276

        geometry = [Point(xy) for xy in zip(
            np.array(self.X_test[['x']]),
            np.array(self.X_test[['y']]))]
        geo_df = gpd.GeoDataFrame(self.X_test, crs=dallas.crs,
                                  geometry=geometry)

        # print("\tPlotting Streets...", end=" ")
        dallas.plot(ax=ax,
                    alpha=0.4,
                    color="steelblue",
                    zorder=2,
                    label="Streets")
        # print("finished!")

        # print("\tPlotting Incidents...", end=" ")

        geo_df.plot(ax=ax,
                    markersize=17.5,
                    color='red',
                    marker='o',
                    zorder=3,
                    label="Incidents")

        # print("finished!")

        geometry = [Point(xy) for xy in zip(self.predicted_sim[0],
                                            self.predicted_sim[1])]

        geo_df = gpd.GeoDataFrame(self.X_test,
                                  crs=dallas.crs,
                                  geometry=geometry)

        # print("\tPlotting Simulated Incidents...", end=" ")

        geo_df.plot(ax=ax,
                    markersize=17.5,
                    color='blue',
                    marker='o',
                    zorder=3,
                    label="Simulated Incidents")

        # print("finished!")

        plt.title(f"Dallas Incidents - Spatial Pattern\n",
                  fontdict={'fontsize': 20},
                  pad=25)

        plt.legend(loc="lower right",
                   frameon=False,
                   fontsize=13.5)

        # ax.set_axis_off()
        plt.show()

    def contour_plot(self, bins: int, ti: int, pdf: bool = False):
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

    def generate_grid(self, bins: int = 100):
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

    # @af.timer
    # def plot_4d(self,
    #             jpg: bool = False,
    #             interactive: bool = False):
    #     """
    #     Docstring
    #     """
    #
    #     print("\nCreating 4D plot...")
    #
    #     # get the material library
    #     materialLibrary1 = GetMaterialLibrary()
    #
    #     # Create a new 'Render View'
    #     renderView1 = CreateView('RenderView')
    #     # renderView1.GetRenderWindow().SetFullScreen(True)
    #     renderView1.ViewSize = [1080, 720]
    #     renderView1.AxesGrid = 'GridAxes3DActor'
    #     renderView1.CenterOfRotation = [2505251.4078454673, 6981929.658190809,
    #                                     336000.0]
    #     renderView1.StereoType = 0
    #     renderView1.CameraPosition = [2672787.133709079, 6691303.18204621,
    #                                   510212.0148339354]
    #     renderView1.CameraFocalPoint = [2505211.707979299, 6981538.82059641,
    #                                     335389.1971413433]
    #     renderView1.CameraViewUp = [-0.23106610676072656, 0.40064135354217617,
    #                                 0.8866199637603102]
    #     renderView1.CameraParallelScale = 97832.66331727587
    #     renderView1.Background = [0.0, 0.0, 0.0]
    #     renderView1.OSPRayMaterialLibrary = materialLibrary1
    #
    #     # init the 'GridAxes3DActor' selected for 'AxesGrid'
    #     renderView1.AxesGrid.XTitle = 'X_train'
    #     renderView1.AxesGrid.YTitle = 'Y'
    #     renderView1.AxesGrid.ZTitle = 'T'
    #     renderView1.AxesGrid.XTitleFontFile = ''
    #     renderView1.AxesGrid.YTitleFontFile = ''
    #     renderView1.AxesGrid.ZTitleFontFile = ''
    #     renderView1.AxesGrid.XLabelFontFile = ''
    #     renderView1.AxesGrid.YLabelFontFile = ''
    #     renderView1.AxesGrid.ZLabelFontFile = ''
    #
    #     # ----------------------------------------------------------------
    #     # restore active view
    #     SetActiveView(renderView1)
    #     # ----------------------------------------------------------------
    #
    #     # ----------------------------------------------------------------
    #     # setup the data processing pipelines
    #     # ----------------------------------------------------------------
    #
    #     # Si no existe STKDE_grid.vts, crear malla (INCOMPLETO)
    #
    #     # create a new 'XML Structured Grid Reader'
    #
    #     print("\n\tLoading 3D grid...", end=" ")
    #
    #     densities = XMLStructuredGridReader(FileName=[
    #         '/Users/msmendozaelguera/Desktop/iPre/Modeling/Python/STKDE '
    #         'grid.vts'])
    #     densities.PointArrayStatus = ['density', 'y_day']
    #
    #     print("finished!")
    #
    #     # create a new 'GDAL Vector Reader'
    #     print("\tLoading Dallas Street shapefile...", end=" ")
    #
    #     dallasMap = GDALVectorReader(
    #         FileName='/Users/msmendozaelguera/Desktop/iPre/Modeling/Data'
    #                  '/shapefiles'
    #                  '/streets.shp')
    #
    #     print("finished!")
    #
    #     print("\tPlotting Contours...", end=" ")
    #
    #     # create a new 'Contour'
    #     aboveSurfaceContour = Contour(Input=densities)
    #     aboveSurfaceContour.ContourBy = ['POINTS', 'y_day']
    #     aboveSurfaceContour.Isosurfaces = [360.0]
    #     aboveSurfaceContour.PointMergeMethod = 'Uniform Binning'
    #
    #     # create a new 'Contour'
    #     aboveContour = Contour(Input=aboveSurfaceContour)
    #     aboveContour.ContourBy = ['POINTS', 'density']
    #     aboveContour.ComputeScalars = 1
    #     aboveContour.Isosurfaces = [1.2217417871681798e-13]
    #     aboveContour.PointMergeMethod = 'Uniform Binning'
    #
    #     # create a new 'Contour'
    #     belowSurfaceContour = Contour(Input=densities)
    #     belowSurfaceContour.ContourBy = ['POINTS', 'y_day']
    #     belowSurfaceContour.Isosurfaces = [307.0]
    #     belowSurfaceContour.PointMergeMethod = 'Uniform Binning'
    #
    #     # create a new 'Contour'
    #     belowContour = Contour(Input=belowSurfaceContour)
    #     belowContour.ContourBy = ['POINTS', 'density']
    #     belowContour.ComputeScalars = 1
    #     belowContour.Isosurfaces = [1.8379040711040815e-12]
    #     belowContour.PointMergeMethod = 'Uniform Binning'
    #
    #     print("finished!")
    #
    #     # ----------------------------------------------------------------
    #     # setup the visualization in view 'renderView1'
    #     # ----------------------------------------------------------------
    #
    #     print("\tRendering Volume...", end=" ")
    #
    #     # show data from densities
    #     densitiesDisplay = Show(densities, renderView1)
    #
    #     # get color transfer function/color map for 'density'
    #     densityLUT = GetColorTransferFunction('density')
    #     densityLUT.RGBPoints = [0.0, 0.278431372549, 0.278431372549,
    #                             0.858823529412,
    #                             5.418939399286293e-13, 0.0, 0.0,
    #                             0.360784313725,
    #                             1.0799984117458697e-12, 0.0, 1.0, 1.0,
    #                             1.625681819785888e-12, 0.0, 0.501960784314,
    #                             0.0,
    #                             2.1637862916031283e-12, 1.0, 1.0, 0.0,
    #                             2.7056802315317578e-12, 1.0, 0.380392156863,
    #                             0.0,
    #                             3.247574171460387e-12, 0.419607843137, 0.0,
    #                             0.0,
    #                             3.7894681113890166e-12, 0.878431372549,
    #                             0.301960784314,
    #                             0.301960784314]
    #     densityLUT.ColorSpace = 'RGB'
    #     densityLUT.ScalarRangeInitialized = 1.0
    #
    #     # get opacity transfer function/opacity map for 'density'
    #     densityPWF = GetOpacityTransferFunction('density')
    #     densityPWF.Points = [0.0, 0.0, 0.5, 0.0, 3.7894681113890166e-12, 1.0,
    #                          0.5, 0.0]
    #     densityPWF.ScalarRangeInitialized = 1
    #
    #     # trace defaults for the display properties.
    #     densitiesDisplay.Representation = 'Volume'
    #     densitiesDisplay.ColorArrayName = ['POINTS', 'density']
    #     densitiesDisplay.LookupTable = densityLUT
    #     densitiesDisplay.Scale = [1.0, 1.0, 1000.0]
    #     densitiesDisplay.OSPRayScaleArray = 'density'
    #     densitiesDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    #     densitiesDisplay.SelectOrientationVectors = 'None'
    #     densitiesDisplay.ScaleFactor = 13997.517379119061
    #     densitiesDisplay.SelectScaleArray = 'None'
    #     densitiesDisplay.GlyphType = 'Arrow'
    #     densitiesDisplay.GlyphTableIndexArray = 'None'
    #     densitiesDisplay.GaussianRadius = 699.875868955953
    #     densitiesDisplay.SetScaleArray = ['POINTS', 'density']
    #     densitiesDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    #     densitiesDisplay.OpacityArray = ['POINTS', 'density']
    #     densitiesDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    #     densitiesDisplay.DataAxesGrid = 'GridAxesRepresentation'
    #     densitiesDisplay.SelectionCellLabelFontFile = ''
    #     densitiesDisplay.SelectionPointLabelFontFile = ''
    #     densitiesDisplay.PolarAxes = 'PolarAxesRepresentation'
    #     densitiesDisplay.ScalarOpacityFunction = densityPWF
    #     densitiesDisplay.ScalarOpacityUnitDistance = 2272.3875215933476
    #
    #     # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
    #     densitiesDisplay.DataAxesGrid.XTitleFontFile = ''
    #     densitiesDisplay.DataAxesGrid.YTitleFontFile = ''
    #     densitiesDisplay.DataAxesGrid.ZTitleFontFile = ''
    #     densitiesDisplay.DataAxesGrid.XLabelFontFile = ''
    #     densitiesDisplay.DataAxesGrid.YLabelFontFile = ''
    #     densitiesDisplay.DataAxesGrid.ZLabelFontFile = ''
    #
    #     # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
    #     densitiesDisplay.PolarAxes.Scale = [1.0, 1.0, 1000.0]
    #     densitiesDisplay.PolarAxes.PolarAxisTitleFontFile = ''
    #     densitiesDisplay.PolarAxes.PolarAxisLabelFontFile = ''
    #     densitiesDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
    #     densitiesDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''
    #
    #     print("finished!")
    #
    #     print("\tMaking some adjustments...", end=" ")
    #
    #     # show data from belowContour
    #     belowContourDisplay = Show(belowContour, renderView1)
    #
    #     # trace defaults for the display properties.
    #     belowContourDisplay.Representation = 'Wireframe'
    #     belowContourDisplay.ColorArrayName = ['POINTS', 'density']
    #     belowContourDisplay.LookupTable = densityLUT
    #     belowContourDisplay.LineWidth = 2.0
    #     belowContourDisplay.Scale = [1.0, 1.0, 1000.0]
    #     belowContourDisplay.OSPRayScaleArray = 'Normals'
    #     belowContourDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    #     belowContourDisplay.SelectOrientationVectors = 'None'
    #     belowContourDisplay.ScaleFactor = 7461.155641368963
    #     belowContourDisplay.SelectScaleArray = 'None'
    #     belowContourDisplay.GlyphType = 'Arrow'
    #     belowContourDisplay.GlyphTableIndexArray = 'None'
    #     belowContourDisplay.GaussianRadius = 373.05778206844815
    #     belowContourDisplay.SetScaleArray = ['POINTS', 'Normals']
    #     belowContourDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    #     belowContourDisplay.OpacityArray = ['POINTS', 'Normals']
    #     belowContourDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    #     belowContourDisplay.DataAxesGrid = 'GridAxesRepresentation'
    #     belowContourDisplay.SelectionCellLabelFontFile = ''
    #     belowContourDisplay.SelectionPointLabelFontFile = ''
    #     belowContourDisplay.PolarAxes = 'PolarAxesRepresentation'
    #
    #     # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
    #     belowContourDisplay.DataAxesGrid.XTitleFontFile = ''
    #     belowContourDisplay.DataAxesGrid.YTitleFontFile = ''
    #     belowContourDisplay.DataAxesGrid.ZTitleFontFile = ''
    #     belowContourDisplay.DataAxesGrid.XLabelFontFile = ''
    #     belowContourDisplay.DataAxesGrid.YLabelFontFile = ''
    #     belowContourDisplay.DataAxesGrid.ZLabelFontFile = ''
    #
    #     # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
    #     belowContourDisplay.PolarAxes.Scale = [1.0, 1.0, 1000.0]
    #     belowContourDisplay.PolarAxes.PolarAxisTitleFontFile = ''
    #     belowContourDisplay.PolarAxes.PolarAxisLabelFontFile = ''
    #     belowContourDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
    #     belowContourDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''
    #
    #     # show data from aboveContour
    #     aboveContourDisplay = Show(aboveContour, renderView1)
    #
    #     # trace defaults for the display properties.
    #     aboveContourDisplay.Representation = 'Wireframe'
    #     aboveContourDisplay.ColorArrayName = ['POINTS', 'density']
    #     aboveContourDisplay.LookupTable = densityLUT
    #     aboveContourDisplay.LineWidth = 2.0
    #     aboveContourDisplay.Scale = [1.0, 1.0, 1000.0]
    #     aboveContourDisplay.OSPRayScaleArray = 'Normals'
    #     aboveContourDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    #     aboveContourDisplay.SelectOrientationVectors = 'None'
    #     aboveContourDisplay.ScaleFactor = 12060.679544949253
    #     aboveContourDisplay.SelectScaleArray = 'None'
    #     aboveContourDisplay.GlyphType = 'Arrow'
    #     aboveContourDisplay.GlyphTableIndexArray = 'None'
    #     aboveContourDisplay.GaussianRadius = 603.0339772474626
    #     aboveContourDisplay.SetScaleArray = ['POINTS', 'Normals']
    #     aboveContourDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    #     aboveContourDisplay.OpacityArray = ['POINTS', 'Normals']
    #     aboveContourDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    #     aboveContourDisplay.DataAxesGrid = 'GridAxesRepresentation'
    #     aboveContourDisplay.SelectionCellLabelFontFile = ''
    #     aboveContourDisplay.SelectionPointLabelFontFile = ''
    #     aboveContourDisplay.PolarAxes = 'PolarAxesRepresentation'
    #
    #     # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
    #     aboveContourDisplay.DataAxesGrid.XTitleFontFile = ''
    #     aboveContourDisplay.DataAxesGrid.YTitleFontFile = ''
    #     aboveContourDisplay.DataAxesGrid.ZTitleFontFile = ''
    #     aboveContourDisplay.DataAxesGrid.XLabelFontFile = ''
    #     aboveContourDisplay.DataAxesGrid.YLabelFontFile = ''
    #     aboveContourDisplay.DataAxesGrid.ZLabelFontFile = ''
    #
    #     # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
    #     aboveContourDisplay.PolarAxes.Scale = [1.0, 1.0, 1000.0]
    #     aboveContourDisplay.PolarAxes.PolarAxisTitleFontFile = ''
    #     aboveContourDisplay.PolarAxes.PolarAxisLabelFontFile = ''
    #     aboveContourDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
    #     aboveContourDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''
    #
    #     # show data from dallasMap
    #     dallasMapDisplay = Show(dallasMap, renderView1)
    #
    #     # trace defaults for the display properties.
    #     dallasMapDisplay.Representation = 'Wireframe'
    #     dallasMapDisplay.AmbientColor = [0.5019607843137255,
    #                                      0.5019607843137255,
    #                                      0.5019607843137255]
    #     dallasMapDisplay.ColorArrayName = ['POINTS', '']
    #     dallasMapDisplay.DiffuseColor = [0.7137254901960784,
    #                                      0.7137254901960784,
    #                                      0.7137254901960784]
    #     dallasMapDisplay.MapScalars = 0
    #     dallasMapDisplay.InterpolateScalarsBeforeMapping = 0
    #     dallasMapDisplay.PointSize = 0.5
    #     dallasMapDisplay.LineWidth = 0.5
    #     dallasMapDisplay.Position = [0.0, 0.0, 305000.0]
    #     dallasMapDisplay.Scale = [1.0, 1.0, 1000.0]
    #     dallasMapDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    #     dallasMapDisplay.SelectOrientationVectors = 'None'
    #     dallasMapDisplay.ScaleFactor = 17341.236314833164
    #     dallasMapDisplay.SelectScaleArray = 'None'
    #     dallasMapDisplay.GlyphType = 'Arrow'
    #     dallasMapDisplay.GlyphTableIndexArray = 'None'
    #     dallasMapDisplay.GaussianRadius = 867.0618157416583
    #     dallasMapDisplay.SetScaleArray = [None, '']
    #     dallasMapDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    #     dallasMapDisplay.OpacityArray = [None, '']
    #     dallasMapDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    #     dallasMapDisplay.DataAxesGrid = 'GridAxesRepresentation'
    #     dallasMapDisplay.SelectionCellLabelFontFile = ''
    #     dallasMapDisplay.SelectionPointLabelFontFile = ''
    #     dallasMapDisplay.PolarAxes = 'PolarAxesRepresentation'
    #
    #     # init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
    #     dallasMapDisplay.DataAxesGrid.XTitleFontFile = ''
    #     dallasMapDisplay.DataAxesGrid.YTitleFontFile = ''
    #     dallasMapDisplay.DataAxesGrid.ZTitleFontFile = ''
    #     dallasMapDisplay.DataAxesGrid.XLabelFontFile = ''
    #     dallasMapDisplay.DataAxesGrid.YLabelFontFile = ''
    #     dallasMapDisplay.DataAxesGrid.ZLabelFontFile = ''
    #
    #     # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
    #     dallasMapDisplay.PolarAxes.Translation = [0.0, 0.0, 305000.0]
    #     dallasMapDisplay.PolarAxes.Scale = [1.0, 1.0, 1000.0]
    #     dallasMapDisplay.PolarAxes.PolarAxisTitleFontFile = ''
    #     dallasMapDisplay.PolarAxes.PolarAxisLabelFontFile = ''
    #     dallasMapDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
    #     dallasMapDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''
    #
    #     # setup the color label parameters for each label in this view
    #
    #     # get color label/bar for densityLUT in view renderView1
    #     densityLUTColorBar = GetScalarBar(densityLUT, renderView1)
    #     densityLUTColorBar.WindowLocation = 'AnyLocation'
    #     densityLUTColorBar.Position = [0.031037827352085365,
    #                                    0.6636363636363637]
    #     densityLUTColorBar.Title = 'density'
    #     densityLUTColorBar.ComponentTitle = ''
    #     densityLUTColorBar.TitleFontFile = ''
    #     densityLUTColorBar.LabelFontFile = ''
    #     densityLUTColorBar.ScalarBarLength = 0.3038636363636361
    #
    #     # set color bar visibility
    #     densityLUTColorBar.Visibility = 1
    #
    #     # show color label
    #     densitiesDisplay.SetScalarBarVisibility(renderView1, True)
    #
    #     # show color label
    #     belowContourDisplay.SetScalarBarVisibility(renderView1, True)
    #
    #     # show color label
    #     aboveContourDisplay.SetScalarBarVisibility(renderView1, True)
    #
    #     # ----------------------------------------------------------------
    #     # setup color maps and opacity mapes used in the visualization
    #     # note: the Get..() functions create a new object, if needed
    #     # ----------------------------------------------------------------
    #
    #     # ----------------------------------------------------------------
    #     # finally, restore active source
    #     SetActiveSource(None)
    #     # ----------------------------------------------------------------
    #
    #     print("finished!")
    #
    #     if jpg:
    #         print("\tSaving .jpg file...", end=" ")
    #
    #         SaveScreenshot('STKDE_4D.jpg', ImageResolution=(1080, 720))
    #
    #         plt.subplots(figsize=(10.80, 7.2))
    #
    #         img = mpimg.imread('STKDE_4D.png')
    #         plt.imshow(img)
    #
    #         plt.axis('off')
    #         plt.show()
    #
    #         print("finished!")
    #     if interactive:
    #         print("\tInteractive Window Mode ON...", end=" ")
    #         Interact()
    #         print("finished!")

    def calculate_hr(self, c=None):
        """
        Parameters
        ----------
        c : np.linspace
            Threshold de confianza para
            filtrar hotspots
        Returns
        -------
        hr_by_group: list
                    Lista con los valores del HR
                    para cada grupo
        ap_by_group: list
                    Lista con los valores del
                    Area Percentage para cada grupo
        """
        if self.f_delitos is None:
            self.predict()
        f_delitos, f_nodos = self.f_delitos, self.f_nodos
        # c = np.linspace(0, f_nodos.max(), 100)
        hits = [np.sum(f_delitos >= c[i]) for i in range(c.size)]
        area_h = [np.sum(f_nodos >= c[i]) for i in range(c.size)]
        HR = [i / len(f_delitos) for i in hits]
        area_percentaje = [i / len(f_nodos) for i in area_h]
        self.hr, self.ap = HR, area_percentaje
        return self.hr, self.ap

    def calculate_pai(self, c=None):
        """
        Parameters
        ----------
        c : np.linspace
            Threshold de confianza para
            filtrar hotspots
        Returns
        -------
        pai_by_group : list
                    Lista con los valores del PAI
                    para cada grupo
        hr_by_group: list
                    Lista con los valores del HR
                    para cada grupo
        ap_by_group: list
                    Lista con los valores del
                    Area Percentage para cada grupo
        """
        if not self.hr:
            self.calculate_hr(c)
        PAI = [float(self.hr[i]) / float(self.ap[i]) if
               self.ap[i] else 0 for i in range(len(self.hr))]
        self.pai = PAI
        return self.pai, self.hr, self.ap

    def validate(self, c=0):
        if type(c != list):
            hits = self.f_delitos[self.f_delitos > c]
        else:
            hits = self.f_delitos[self.f_delitos < max(c)]
            hits = self.hits[hits > min(c)]
        self.d_incidents = hits.size


class RForestRegressor(object):
    def __init__(self, data_0=None, shps=None,
                 xc_size=100, yc_size=100, n_layers=7,
                 t_history=4, start_prediction=date(2017, 11, 1),
                 read_data=False, w_data=False,
                 read_X=False, w_X=False,
                 name='RForestRegressor'):
        """ Regressor modificado de Scipy usado para predecir delitos.

        Parameters
        ----------
        data_0 : pd.DataFrame
          Corresponde a los datos extraídos en primera instancia desde
          la Socrata API.
        shps : gpd.GeoDataFrame
          GeoDataFrame que contiene información sobre la ciudad de Dallas
        xc_size : int
          Tamaño de las celdas en x [metros]
        yc_size : int
          Tamaño de las celdas en y [metros]
        n_layers : int
          Cantidad de capas para contabilizar incidentes de celdas vecinas
        t_history : int
          Ventanas temporales consideradas 'hacia atrás' desde la fecha de
          término de entrenamiento, que serán usadas para entrenar el
          regressor
        start_prediction : date
          Fecha de comienzo en la ventana temporal a predecir
        read_data : bool
          True para leer el DataFrame con los incidentes de la Socrata
          de un archivo .pkl previamente guardado.
        read_X : bool
          True para leer el DataFrame con la información de las celdas
          en Dallas de un archivo .pkl previamente guardado.
        name : str
          Nombre especial para el regressor que aparece en los plots,
          estadísticas, etc.

        Returns
        -------
        RForestRegressor
        """
        self.name = name
        self.shps = shps
        self.xc_size, self.yc_size = xc_size, yc_size
        self.n_layers = n_layers
        self.nx, self.ny, self.hx, self.hy = [None] * 4
        self.t_history = t_history
        self.start_prediction = start_prediction
        self.weeks = []
        self.l_weights = None

        self.rfr = RandomForestRegressor(n_jobs=8)
        self.ap, self.hr, self.pai = [None] * 3

        start_prediction = self.start_prediction
        # current date, que corresponde al último día en la ventana
        # temporal de entrenamiento, desde donde se comenzarán a
        # armar los grupos de entrenamiento
        c_date = self.start_prediction - timedelta(days=1)
        for _ in range(self.t_history):
            c_date -= timedelta(days=7)
            self.weeks.append(c_date + timedelta(days=1))
        self.weeks.reverse()
        self.weeks.append(start_prediction)

        self.data_0 = data_0
        self.data = self.data_0
        self.X = None
        self.read_data, self.read_X = read_data, read_X
        self.w_data, self.w_X = w_data, w_X
        self.X = None

        self.d_incidents = 0  # Detected incidents
        self.h_area = 0  # Hotspot area
        if self.read_X:
            self.X = pd.read_pickle('predictivehp/data/X.pkl')
        if self.read_data:
            self.data = pd.read_pickle('predictivehp/data/data.pkl')

    def set_parameters(self, t_history,
                       xc_size, yc_size, n_layers,
                       label_weights=None,
                       read_data=False, w_data=False,
                       read_X=False, w_X=False):
        """
        Setea los hiperparámetros del modelo

        Parameters
        ----------
        t_history : int
          Número de semanas hacia atrás a partir del 31-10-2017 que
          serán usadas para entrenar el modelo
        xc_size : int
          Tamaño en metros de las celdas en x de la malla
        yc_size : int
          Tamaño en metros de las celdas en y de la malla
        n_layers : int
          Número de capas para considerar en el conteo de delitos de
          celdas vecinas
        label_weights : np.ndarray
        read_data : bool
        read_X : bool
        w_data : bool
        w_X : bool
        """
        self.t_history = t_history
        self.xc_size = xc_size
        self.yc_size = yc_size
        self.n_layers = n_layers
        self.l_weights = label_weights
        self.read_data = read_data
        self.read_X = read_X
        self.w_data = w_data
        self.w_X = w_X

    def print_parameters(self):
        print('RFR Hyperparameters')
        print(f'{"Training History:":<20s}{self.t_history} weeks')
        print(f'{"xc_size:":<20s}{self.xc_size} m')
        print(f'{"yc_size:":<20s}{self.yc_size} m')
        print(f'{"n_layers:":<20s}{self.n_layers}')
        print(f'{"l_weights:":<20s}{self.l_weights}')
        print()

    def generate_X(self):
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
        # print("\tCreating mgrid...")
        x_min, y_min, x_max, y_max = self.shps['streets'].total_bounds
        x_bins = abs(x_max - x_min) / self.xc_size
        y_bins = abs(y_max - y_min) / self.yc_size
        x, y = np.mgrid[x_min: x_max: x_bins * 1j, y_min: y_max: y_bins * 1j, ]

        # Creación del esqueleto del dataframe
        print("\tCreating dataframe columns...")

        X_cols = pd.MultiIndex.from_product(
            [[f"Incidents_{i}" for i in range(self.n_layers + 1)], self.weeks]
        )
        X = pd.DataFrame(columns=X_cols)

        # Creación de los parámetros para el cálculo de los índices
        print("\tFilling data...")
        self.nx = x.shape[0] - 1
        self.ny = y.shape[1] - 1
        self.hx = (x.max() - x.min()) / self.nx
        self.hy = (y.max() - y.min()) / self.ny

        if self.read_data:
            self.data = pd.read_pickle('predictivehp/data/data.pkl')
        else:
            # en caso que no se tenga un data.pkl de antes, se recibe el
            # dado por la PreProcessing Class, mientras que self.X es llenado
            # al llamar self.generate_X dentro de self.fit()
            self.data = self.data_0
            # Manejo de los puntos de incidentes para poder trabajar en (x, y)
            geometry = [Point(xy) for xy in zip(np.array(self.data[['x']]),
                                                np.array(self.data[['y']]))]
            self.data = gpd.GeoDataFrame(self.data, crs=2276,
                                         geometry=geometry)
            self.data.to_crs(epsg=3857, inplace=True)
            self.data['Cell'] = None
            self.assign_cells()

            if self.w_data:
                self.to_pickle('data.pkl')

        # Nro. incidentes en la i-ésima capa de la celda (i, j)
        for week in self.weeks:
            print(f"\t\t{week}... ", end=' ')
            wi_date = week
            wf_date = week + timedelta(days=6)
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
                X.loc[:, (f"Incidents_{i}", f"{week}")] = \
                    af.to_df_col(D) if i == 0 \
                        else af.to_df_col(af.il_neighbors(D, i))
            print('finished!')

        # Adición de las columnas 'geometry' e 'in_dallas' al data
        print("\tPreparing data for filtering...")
        X[('geometry', '')] = [Point(i) for i in
                               zip(x[:-1, :-1].flatten(),
                                   y[:-1, :-1].flatten())]
        X[('in_dallas', '')] = 0

        # Filtrado de celdas (llenado de la columna 'in_dallas')
        X = af.filter_cells(df=X, shp=self.shps['councils'])
        X.drop(columns=[('in_dallas', '')], inplace=True)

        self.X = X

    def to_pickle(self, file_name):
        """Genera un pickle de self.data o self.data dependiendo el nombre
        dado (data.pkl o X.pkl).

        OBS.

        >>> self.data  # es guardado en self.generate_X()
        >>> self.X  # es guardado en self.predict()

        luego, si self.read_X = True, no es necesario realizar un
        self.fit() o self.predict()

        Parameters
        ----------
        file_name : str
          Nombre del pickle a generar en predictivehp/data/file_name
        """
        # print("\nPickling dataframe...", end=" ")
        if file_name == "X.pkl":
            self.X.to_pickle(f"predictivehp/data/{file_name}")
        if file_name == "data.pkl":
            self.data.to_pickle(f"predictivehp/data/{file_name}")

    def assign_cells(self):
        """Rellena la columna 'Cell' de self.data. Asigna el número de
        celda asociado a cada incidente.

        Parameters
        ----------
        week : date
          Día en el cual comienza el periodo de predicción
        """
        # print("\nAssigning cells...\n")
        x_min, y_min, x_max, y_max = self.shps['streets'].total_bounds

        x_bins = abs(x_max - x_min) / self.xc_size
        y_bins = abs(y_max - y_min) / self.yc_size

        x, y = np.mgrid[x_min:x_max:x_bins * 1j, y_min:y_max:y_bins * 1j, ]

        nx = x.shape[0] - 1
        ny = y.shape[1] - 1
        hx = (x.max() - x.min()) / nx
        hy = (y.max() - y.min()) / ny
        for idx, inc in self.data.iterrows():
            xi, yi = inc.geometry.x, inc.geometry.y
            nx_i = af.n_i(xi, x.min(), hx)
            ny_i = af.n_i(yi, y.min(), hy)
            cell_idx = ny_i + ny * nx_i

            self.data.loc[idx, 'Cell'] = cell_idx

        # Dejamos la asociación inc-cell en el index de self.data
        self.data.set_index('Cell', drop=True, inplace=True)
        self.data.to_pickle('predictivehp/data/data.pkl')

    def fit(self, X, y):
        """Entrena el modelo

        Parameters
        ----------
        X : pd.DataFrame
            X_train
        y : pd.DataFrame
            y_train

        Returns
        -------
        self : object
        """
        # print("\tFitting Model...")
        self.rfr.fit(X, y.to_numpy().ravel())
        # Sirven para determinar celdas con TP/FN
        self.X[('Dangerous', '')] = y
        self.X.to_pickle('predictivehp/data/X.pkl')
        return self

    def predict(self, X):
        """Predice el score de peligrosidad en cada una de las celdas
        en la malla de Dallas.

        Parameters
        ----------
        X : pd.DataFrame
          X_test for prediction

        Returns
        -------
        y : np.ndarray
          Vector de predicción que indica el score de peligrocidad en
          una celda de la malla de Dallas
        """
        # print("\tMaking predictions...")
        y_pred = self.rfr.predict(X)
        self.X[('Dangerous_pred', '')] = y_pred / y_pred.max()
        self.X.index.name = 'Cell'
        if self.w_X:
            self.to_pickle('X.pkl')
        return y_pred

    def score(self):
        return self.X[('Dangerous_pred', '')]
        # y_pred = self.predict(X_test)
        #
        # score = self.rfr.score(X_test, y_test)
        # precision = precision_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        #
        # print(f"{'Score:':<10s}{score:1.5f}")
        # print(f"{'Precision:':<10s}{precision:1.5f}")
        # print(f"{'Recall:':<10s}{recall:1.5f}")

    def validate(self, c=None):
        """

        Parameters
        ----------
        c

        Returns
        -------

        """
        cells = self.X[[('geometry', ''), ('Dangerous_pred', '')]]
        cells = gpd.GeoDataFrame(cells)

        if type(c) == list or type(c) == tuple:
            d_cells = cells[
                (c <= cells[('Dangerous_pred', '')]) &
                (cells[('Dangerous_pred', '')] <= c)
                ]
            cells['Hit'] = np.where(
                (c[0] <= cells[('Dangerous_pred', '')])
                & (cells[('Dangerous_pred', '')] <= c[1]),
                1, 0)

        else:
            d_cells = cells[c <= cells[('Dangerous_pred', '')]]
            cells['Hit'] = np.where(
                cells[('Dangerous_pred', '')] >= c, 1, 0
            )

        data_nov = pd.DataFrame(self.data[
                                    (date(2017, 11, 1) <= self.data.date) &
                                    (self.data.date <= date(2017, 11, 7))
                                    ])
        data_nov.columns = pd.MultiIndex.from_product([data_nov.columns, ['']])
        cells.drop(columns='geometry', inplace=True)
        join_ = data_nov.join(cells)

        hits = gpd.GeoDataFrame(join_[join_['Hit'] == 1])

        d_incidents = hits.shape[0]
        h_area = d_cells.shape[0] * self.xc_size * self.yc_size * (10 ** -6)

        self.d_incidents = d_incidents
        self.h_area = h_area

    def calculate_hr(self, c=0.9):
        """

        Parameters
        ----------
        c : {float, np.ndarray}
          Threshold de confianza para filtrar hotspots
        """
        if c.size == 1:
            data_nov = pd.DataFrame(
                self.data[(date(2017, 11, 1) <= self.data.date) &
                          (self.data.date <= date(2017, 11, 7))]
            )  # 62 Incidentes
            data_nov.drop(columns='geometry', inplace=True)
            data_nov.columns = pd.MultiIndex.from_product(
                [data_nov.columns, ['']]
            )
            ans = data_nov.join(self.X)
            incidentsh = ans[ans[('Dangerous_pred', '')] >= c[0]]
            hr = incidentsh.shape[0] / data_nov.shape[0]
            return hr
        else:
            A = self.X.shape[0]

            def a(X, c):
                return X[X[('Dangerous_pred', '')] >= c].shape[0]

            c_arr = c
            hr_l = []
            ap_l = []
            for c in c_arr:
                hr_l.append(self.calculate_hr(c=np.array([c])))
                ap_l.append(a(self.X, c) / A)
            self.hr = np.array(hr_l)
            self.ap = np.array(ap_l)

    def calculate_pai(self, c=0.9):
        """
        Calcula el Predictive Accuracy Index (PAI)

        :param [float, np.ndarray] c:
        :return:
        """

        # data_oct = pd.DataFrame(self.data[self.data.month1 == 'October'])
        # data_oct.drop(columns='geometry', inplace=True)

        # ans = data_oct.join(other=fwork.data, on='Cell', how='left')
        # ans = self.data[self.data[('geometry', '')].notna()]

        # a = self.data[self.data[('Dangerous_pred_Oct', '')] == 1].shape[0]
        # a = self.data[self.data[('Dangerous_pred_Oct_rfr', '')] >= c].shape[0]
        def a(x, c):
            return x[x[('Dangerous_pred', '')] >= c].shape[0]

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
                hr_l.append(hr), ap_l.append(ap)
                if ap == 0:
                    pai_l.append(0)
                    continue
                pai_l.append(hr / ap)
            self.hr = np.array(hr_l)
            self.ap = np.array(ap_l)
            self.pai = np.array(pai_l)

    def heatmap(self, c=0, show_score=True, incidences=False,
                savefig=False, fname='', **kwargs):
        """

        Parameters
        ----------
        c : {int, list, tuple}
        show_score
        incidences
        savefig
        fname
        kwargs

        Returns
        -------

        """
        cells = self.X[[('geometry', ''), ('Dangerous_pred', '')]]
        cells = gpd.GeoDataFrame(cells)
        d_streets = self.shps['streets']

        fig, ax = plt.subplots(figsize=[prm.f_size[0]] * 2)
        #     fig.dpi = 300
        d_streets.plot(ax=ax, alpha=0.2, lw=0.3, color="w", label="Streets")

        if type(c) == list or type(c) == tuple:
            d_cells = cells[
                (cells[('Dangerous_pred', '')] <= c[1]) &
                (c[0] <= cells[('Dangerous_pred')])
                ]
            cells['Hit'] = np.where(
                (c[0] <= cells[('Dangerous_pred', '')]) &
                (cells[('Dangerous_pred', '')] <= c[1]),
                1, 0)
        elif c is not None and c >= 0:
            d_cells = cells[cells[('Dangerous_pred', '')] >= c]
            cells['Hit'] = np.where(
                cells[('Dangerous_pred', '')] >= c, 1, 0
            )
        else:
            d_cells = cells[cells[('Dangerous_pred', '')] >= c]
            cells['Hit'] = np.where(
                cells[('Dangerous_pred', '')] >= c, 1, 0
            )

        if c is None:
            d_cells.plot(ax=ax, column=('Dangerous_pred', ''), cmap='jet',
                         marker=',', markersize=0.1)
        else:
            d_cells.plot(ax=ax, marker=',', markersize=0.5, color='darkblue',
                         label='Hotspot')

        if incidences:  # Se plotean los incidentes
            data_nov = pd.DataFrame(self.data[
                                        (date(2017, 11, 1) <= self.data.date) &
                                        (self.data.date <= date(2017, 11, 7))])
            data_nov.columns = pd.MultiIndex.from_product(
                [data_nov.columns, ['']])
            cells.drop(columns='geometry', inplace=True)
            join_ = data_nov.join(cells)

            hits = gpd.GeoDataFrame(join_[join_['Hit'] == 1])
            misses = gpd.GeoDataFrame(join_[join_['Hit'] == 0])
            if not hits.empty:
                hits.plot(ax=ax, marker='x', markersize=0.25, color='lime',
                          label="Hits")
            if not misses.empty:
                misses.plot(ax=ax, marker='x', markersize=0.25, color='r',
                            label="Misses")

        ax.set_axis_off()
        plt.title('RForestRegressor')
        plt.legend()

        if show_score:
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            cmap = mpl.cm.jet
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            # divider = make_axes_locatable(ax)
            c_bar = fig.colorbar(mappable, ax=ax,
                                 fraction=0.15,
                                 shrink=0.5,
                                 aspect=21.5)
            c_bar.ax.set_ylabel('Danger Score')

        plt.tight_layout()
        if savefig:
            plt.savefig(fname, **kwargs)
        print('hihi')
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

    # def plot_incidents(self, i_type="real", month="October"):
    #     """
    #     Plotea los incidentes almacenados en self.data en el mes dado.
    #     Asegurarse que al momento de realizar el ploteo, ya se haya
    #     hecho un llamado al método ml_algorithm() para identificar los
    #     incidentes TP y FN
    #
    #     :param str i_type: Tipo de incidente a plotear (e.g. TP, FN, TP & FN)
    #     :param str month: String con el nombre del mes que se predijo
    #         con ml_algorithm()
    #     :return:
    #     """
    #
    #     print(f"\nPlotting {month} Incidents...")
    #     print("\tFiltering incidents...")
    #
    #     tp_data, fn_data, data = None, None, None
    #
    #     if i_type == "TP & FN":
    #         data = gpd.GeoDataFrame(self.X)
    #         tp_data = data[self.X.TP == 1]
    #         fn_data = data[self.X.FN == 1]
    #     if i_type == "TP":
    #         data = gpd.GeoDataFrame(self.X)
    #         tp_data = self.X[self.X.TP == 1]
    #     if i_type == "FN":
    #         data = gpd.GeoDataFrame(self.X)
    #         fn_data = self.X[self.X.FN == 1]
    #     if i_type == "real":
    #         data = self.data[self.data.month1 == month]
    #         n_incidents = data.shape[0]
    #         print(f"\tNumber of Incidents in {month}: {n_incidents}")
    #     if i_type == "pred":
    #         data = gpd.GeoDataFrame(self.X)
    #         all_hp = data[self.X[('Dangerous_pred_Oct', '')] == 1]
    #
    #     print("\tReading shapefile...")
    #     d_streets = gpd.GeoDataFrame.from_file(
    #         "../Data/Streets/streets.shp")
    #     d_streets.to_crs(epsg=3857, inplace=True)
    #
    #     print("\tRendering Plot...")
    #     fig, ax = plt.subplots(figsize=(20, 15))
    #
    #     d_streets.plot(ax=ax,
    #                    alpha=0.4,
    #                    color="dimgrey",
    #                    zorder=2,
    #                    label="Streets")
    #
    #     if i_type == 'pred':
    #         all_hp.plot(
    #             ax=ax,
    #             markersize=2.5,
    #             color='y',
    #             marker='o',
    #             zorder=3,
    #             label="TP Incidents"
    #         )
    #     if i_type == "real":
    #         data.plot(
    #             ax=ax,
    #             markersize=10,
    #             color='darkorange',
    #             marker='o',
    #             zorder=3,
    #             label="TP Incidents"
    #         )
    #     if i_type == "TP":
    #         tp_data.plot(
    #             ax=ax,
    #             markersize=2.5,
    #             color='red',
    #             marker='o',
    #             zorder=3,
    #             label="TP Incidents"
    #         )
    #     if i_type == "FN":
    #         fn_data.plot(
    #             ax=ax,
    #             markersize=2.5,
    #             color='blue',
    #             marker='o',
    #             zorder=3,
    #             label="FN Incidents"
    #         )
    #     if i_type == "TP & FN":
    #         tp_data.plot(
    #             ax=ax,
    #             markersize=2.5,
    #             color='red',
    #             marker='o',
    #             zorder=3,
    #             label="TP Incidents"
    #         )
    #         fn_data.plot(
    #             ax=ax,
    #             markersize=2.5,
    #             color='blue',
    #             marker='o',
    #             zorder=3,
    #             label="FN Incidents"
    #         )
    #
    #     # Legends
    #     handles = [
    #         Line2D([], [],
    #                marker='o',
    #                color='darkorange',
    #                label='Incident',
    #                linestyle='None'),
    #         Line2D([], [],
    #                marker='o',
    #                color='red',
    #                label='TP Incident',
    #                linestyle='None'),
    #         Line2D([], [],
    #                marker='o',
    #                color="blue",
    #                label="FN Incident",
    #                linestyle='None'),
    #         Line2D([], [],
    #                marker='o',
    #                color='y',
    #                label='Predicted Incidents',
    #                linestyle='None')
    #     ]
    #
    #     plt.legend(loc="best",
    #                bbox_to_anchor=(0.1, 0.7),
    #                frameon=False,
    #                fontsize=13.5,
    #                handles=handles)
    #
    #     legends = ax.get_legend()
    #     for text in legends.get_texts():
    #         text.set_color('white')
    #
    #         hits = gpd.GeoDataFrame(join_[join_['Hit'] == 1])
    #         misses = gpd.GeoDataFrame(join_[join_['Hit'] == 0])
    #         if not hits.empty:
    #             hits.plot(ax=ax, marker='x', markersize=0.25, color='lime',
    #                       label="Hits")
    #         if not misses.empty:
    #             misses.plot(ax=ax, marker='x', markersize=0.25, color='r',
    #                         label="Misses")
    #
    #     ax.set_axis_off()
    #     fig.set_facecolor('black')
    #     plt.show()
    #     plt.close()

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

        # Creamos el data para los datos reales (1) y predichos (2).
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

    # def plot_joined_cells(self):
    #     """
    #
    #     :return:
    #     """
    #
    #     data_oct = pd.DataFrame(self.data[self.data.month1 == 'October'])
    #     data_oct.drop(columns='geometry', inplace=True)
    #
    #     ans = data_oct.join(other=self.X, on='Cell', how='left')
    #     ans = ans[ans[('geometry', '')].notna()]
    #
    #     gpd_ans = gpd.GeoDataFrame(ans, geometry=ans[('geometry', '')])
    #
    #     d_streets = gpd.GeoDataFrame.from_file(
    #         "../Data/Streets/streets.shp")
    #     d_streets.to_crs(epsg=3857, inplace=True)
    #
    #     fig, ax = plt.subplots(figsize=(20, 15))
    #
    #     d_streets.plot(ax=ax,
    #                    alpha=0.4,
    #                    color="dimgrey",
    #                    zorder=2,
    #                    label="Streets")
    #
    #     gpd_ans.plot(
    #         ax=ax,
    #         markersize=10,
    #         color='red',
    #         marker='o',
    #         zorder=3,
    #         label="Joined Incidents"
    #     )
    #
    #     handles = [
    #         Line2D([], [],
    #                marker='o',
    #                color='red',
    #                label='Joined Incidents',
    #                linestyle='None'),
    #     ]
    #
    #     plt.legend(loc="best",
    #                bbox_to_anchor=(0.1, 0.7),
    #                frameon=False,
    #                fontsize=13.5,
    #                handles=handles)
    #
    #     legends = ax.get_legend()
    #     for text in legends.get_texts():
    #         text.set_color('white')
    #
    #         hits = gpd.GeoDataFrame(join_[join_['Hit'] == 1])
    #         misses = gpd.GeoDataFrame(join_[join_['Hit'] == 0])
    #         if not hits.empty:
    #             hits.plot(ax=ax, marker='x', markersize=0.25, color='lime',
    #                       label="Hits")
    #         if not misses.empty:
    #             misses.plot(ax=ax, marker='x', markersize=0.25, color='r',
    #                         label="Misses")
    #
    #     ax.set_axis_off()
    #     fig.set_facecolor('black')
    #     plt.show()
    #     plt.close()


class ProMap:
    def __init__(self, n_datos=3600, read_density=False,
                 hx=100, hy=100,
                 bw_x=400, bw_y=400, bw_t=7, length_prediction=7,
                 tiempo_entrenamiento=None,
                 start_prediction=date(2017, 11, 1),
                 km2=1_000, name='ProMap', shps=None):

        """
        Modelo Promap
        Parameters
        ----------
        n_datos: int
            indica el nº de datos que se usarán para entrenar el modelo
        read_density: bool
            True si se va a leer una matriz de densidades
        hx: int
            Ancho en x de las celdas en metros
        hy: int
            Ancho en y de las celdas en metros
        bw_x: float
            Ancho de banda en x
        bw_y: float
            Ancho de banda en y
        bw_t: float
            Ancho de banda en t
        length_prediction: int
            Número de días que se quiere predecir
        tiempo_entrenamiento: int
            Número de días que se quiern entrenar al modelo.
            Si es None, se usa bw_t
        start_prediction: datetime
            Fecha desde donde comienza la predicción
        name: str
            Nombre del modelo
        shps: gpd.GeoDataFrame
            GeoDataFrame que contiene información sobre la ciudad de Dallas
        """
        # DATA
        self.n = n_datos
        self.start_prediction = start_prediction
        self.X, self.y = None, None
        self.shps = shps
        self.read_density = read_density

        # MAP
        self.hx, self.hy, self.km2 = hx, hy, km2
        self.bw_x, self.bw_y, self.bw_t = bw_x, bw_y, bw_t
        self.x_min, self.y_min, self.x_max, self.y_max = self.shps[
            'streets'].total_bounds
        self.bins_x = int(round(abs(self.x_max - self.x_min) / self.hx))
        self.bins_y = int(round(abs(self.y_max - self.y_min) / self.hy))

        # MODEL
        self.name = name
        self.lp = length_prediction
        self.prediction = np.zeros((self.bins_x, self.bins_y))
        self.training_matrix = np.zeros((self.bins_x, self.bins_y))
        self.testing_matrix = np.zeros((self.bins_x, self.bins_y))
        self.hr, self.pai, self.ap = None, None, None

        # print('-' * 100)
        # print('\t\t', self.name)

        # print('-' * 100)

    def set_parameters(self, bw, hx=100, hy=100, read_density=False):
        """
        Setea los hiperparámetros del modelo Promap
        Parameters
        ----------
        bw: list or np.array
            anchos de banda en x, y, t
        hx: int
            ancho de la celda en metros, en x
        hy: int
            ancho de la celda en metros, en y
        -------
        """

        self.bw_x, self.bw_y, self.bw_t = bw
        self.hx, self.hy = hx, hy
        self.read_density = read_density
        # se debe actualizar la malla

    def print_parameters(self):
        """
        Imprime los parametros del modelo
        """

        print('ProMap Hyperparameters')
        print(f'bandwith x: {self.bw_x} mts')
        print(f'bandwith y: {self.bw_y} mts')
        print(f'bandwith t: {self.bw_t} days')
        print(f'hx: {self.hx} mts')
        print(f'hy: {self.hy} mts')
        print()

    def create_grid(self):

        """
        Genera una malla en base a los x{min, max} y{min, max}.
        Recordar que cada nodo de la malla representa el centro de cada
        celda en la malla del mapa.

        """

        delta_x = self.hx / 2
        delta_y = self.hy / 2

        self.xx, self.yy = np.mgrid[
                           self.x_min + delta_x:self.x_max - delta_x:self.bins_x * 1j,
                           self.y_min + delta_y:self.y_max - delta_y:self.bins_y * 1j
                           ]

    def fit(self):

        """
        Se encarga de generar la malla en base a los datos del modelo.
        También se encarga de filtrar los puntos que no están dentro de dallas.
        """

        # print('Fitting...')

        self.create_grid()

        # points = np.array([self.xx.flatten(), self.yy.flatten()])
        # self.cells_in_map = af.checked_points_pm(points)  # 141337
        self.cells_in_map = 100_000

    def predict(self, X, y):

        """
        Calula el score en cada celda de la malla de densidades.
        Parameters
        ----------
        X: pd.dataframe
            Son los datos de entrenamiento (x_point, y_point)
        y: pd.dataframe
            Son los datos para el testeo (x_point, y_point)
        """

        self.X = X
        self.y = y
        self.dias_train = self.X['y_day'].max()

        if self.read_density:
            self.prediction = np.load(
                'predictivehp/data/prediction.npy')


        else:
            # print('\nEstimando densidades...')
            # print(
            #     f'\n\tNº de datos para entrenar el modelo: {len(self.X)}')
            # print(
            #     f'\tNº de días usados para entrenar el modelo: {self.dias_train}')
            # print(
            #     f'\tNº de datos para testear el modelo: {len(self.y)}')

            ancho_x = af.radio_pintar(self.hx, self.bw_x)
            ancho_y = af.radio_pintar(self.hy, self.bw_y)

            for k in range(len(self.X)):
                x, y, t = self.X['x_point'][k], self.X['y_point'][k], \
                          self.X['y_day'][k]
                x_in_matrix, y_in_matrix = af.find_position(self.xx, self.yy,
                                                            x, y,
                                                            self.hx, self.hy)
                x_left, x_right = af.limites_x(ancho_x, x_in_matrix, self.xx)
                y_abajo, y_up = af.limites_y(ancho_y, y_in_matrix, self.yy)

                for i in range(x_left, x_right + 1):
                    for j in range(y_abajo, y_up):
                        elem_x = self.xx[i][0]
                        elem_y = self.yy[0][j]
                        time_weight = 1 / af.n_semanas(self.dias_train, t)

                        if af.linear_distance(elem_x, x) > self.bw_x or \
                                af.linear_distance(elem_y, y) > self.bw_y:
                            cell_weight = 0

                        else:
                            cell_weight = 1 / af.cells_distance(x, y, elem_x,
                                                                elem_y,
                                                                self.hx,
                                                                self.hy)

                        self.prediction[i][j] += time_weight * cell_weight

            self.prediction = self.prediction / self.prediction.max()

            # print('\nGuardando datos...')
            np.save('predictivehp/data/prediction.npy', self.prediction)

    def load_train_matrix(self):

        """
        Ubica los delitos en la matriz de entrenamiento.
        """

        for index, row in self.X.iterrows():
            x, y, t = row['x_point'], row['y_point'], row['y_day']

            if t >= (self.dias_train - self.bw_t):
                x_pos, y_pos = af.find_position(self.xx, self.yy, x, y,
                                                self.hx,
                                                self.hy)
                self.training_matrix[x_pos][y_pos] += 1

    def load_test_matrix(self, ventana_dias):

        """
        Ubica los delitos en la matriz de testeo

        Parameters
        ----------
        ventana_dias: int
            Cantidad de días que se quieren predecir.
        """

        for index, row in self.y.iterrows():
            x, y, t = row['x_point'], row['y_point'], row['y_day']

            if t <= (self.dias_train + ventana_dias):
                x_pos, y_pos = af.find_position(self.xx, self.yy, x, y,
                                                self.hx,
                                                self.hy)
                self.testing_matrix[x_pos][y_pos] += 1

            else:
                break

    def calculate_hr(self, c):

        """
        Calcula el hr (n/N)
        Parameters
        ----------
        c: np.linespace(1, n, 100)
            Vector que sirve para analizar el mapa en cada punto
        """

        self.load_train_matrix()
        self.load_test_matrix(self.lp)

        # 1. Solo considera las celdas que son mayor a un K
        #     Esto me entrega una matriz con True/False (Matriz A)
        # 2. La matriz de True/False la multiplico con una matriz que tiene la
        # cantidad de delitos que hay por cada celda (Matriz B)
        # 3. Al multiplicar A * B obtengo una matriz C donde todos los False en
        # A son 0 en B. Todos los True en A mantienen su valor en B
        # 4. Contar cuantos delitos quedaron luego de haber pasado este proceso.

        # Se espera que los valores de la lista vayan disminuyendo a medida que el valor de K aumenta

        hits_n = []

        for i in range(c.size):
            hits_n.append(
                np.sum(
                    (self.prediction >= c[
                        i]) * self.testing_matrix))

        # 1. Solo considera las celdas que son mayor a un K
        #     Esto me entrega una matriz con True/False (Matriz A)
        # 2. Contar cuantos celdas quedaron luego de haber pasado este proceso.

        # Se espera que los valores de la lista vayan disminuyendo a medida que el valor de K aumenta

        area_hits = []

        for i in range(c.size):
            area_hits.append(
                np.count_nonzero(self.prediction >= c[i]))

        n_delitos_testing = np.sum(self.testing_matrix)

        self.hr = [i / n_delitos_testing for i in hits_n]

        self.ap = [1 if j > 1 else j for j in [i / self.cells_in_map for
                                               i in area_hits]]

    def calculate_pai(self, c):

        """
        Calcula el PAI (n/N) / (a/A)
        Parameters
        ----------
        c: np.linespace(1, n, 100)
            Vector que sirve para analizar el mapa en cada punto
        """

        if not self.hr:
            self.calculate_hr(c)

        self.pai = [
            0 if float(self.ap[i]) == 0
            else float(self.hr[i]) / float(self.ap[i])
            for i in range(len(self.ap))]

    def heatmap(self, c=0, show_score=True, incidences=False,
                savefig=False, fname='', **kwargs):
        """
        Mostrar un heatmap de una matriz de riesgo.

        Parameters
        ----------
        c: float
            Representa el valor desde donde quiero ver los scores en el mapa
        nombre_grafico: str
            nombre del gráfico
        incidentes: bool
            True para mostrar cuntos incidentes se han capturado
        -------

        """

        if type(c) == list or type(c) == tuple:
            c1 = min(c)
            c2 = max(c)
            matriz = np.where(self.prediction >= c1,
                              self.prediction, 0)
            matriz = np.where(matriz <= c2, matriz, 0)

        else:

            matriz = np.where(self.prediction >= c,
                              self.prediction, 0)

        dallas = gpd.read_file('predictivehp/data/streets.shp')
        dallas.crs = 2276
        dallas.to_crs(epsg=3857, inplace=True)

        fig, ax = plt.subplots(figsize=[prm.f_size[0]] * 2)
        ax.set_facecolor('xkcd:black')

        plt.title(self.name)
        plt.imshow(np.flipud(matriz.T),
                   extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                   cmap='jet',
                   # vmin=0, vmax=1
                   )

        dallas.plot(ax=ax, alpha=.3, color="gray")

        if incidences:
            geometry = [Point(xy) for xy in zip(
                np.array(self.y[['x_point']]),
                np.array(self.y[['y_point']]))
                        ]
            geo_df = gpd.GeoDataFrame(self.y,
                                      crs=dallas.crs,
                                      geometry=geometry)
            geo_df.plot(ax=ax,
                        markersize=17.5,
                        color='red',
                        marker='o',
                        zorder=3,
                        label="Incidents")

        ax.set_axis_off()
        plt.title('ProMap')
        plt.legend()

        if show_score:
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            cmap = mpl.cm.jet
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            c_bar = fig.colorbar(mappable, ax=ax,
                                 fraction=0.15,
                                 shrink=0.5,
                                 aspect=21.5)
            c_bar.ax.set_ylabel('Danger Score')

        plt.tight_layout()
        if savefig:
            plt.savefig(fname, **kwargs)
        plt.show()

    def plot_incident(self, matriz, nombre_grafico):

        """
        Parameters
        ----------
        matriz: np.mgrid
            Matriz donde cada celda representa un punto del mapa.
            Los valores representan los Nº de delitos en ese punto.
        nombre_grafico: str
            nombre del grafico
        -------
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

    def score(self):

        """
        Entrega la matriz de riesgo.
        Returns: np.mgrid
        -------

        """
        return self.prediction

    def validate(self, c=0):
        self.load_test_matrix(self.lp)
        if type(c) == list or type(c) == tuple:
            c1 = min(c)
            c2 = max(c)
            aux = self.prediction[self.prediction >= c1]
            aux = aux <= c2
            self.d_incidents = np.sum(aux * self.testing_matrix)
            self.h_area = np.count_nonzero(
                aux) * self.hx * self.hy / 10 ** -6  # km^2

        else:
            self.d_incidents = np.sum(
                (self.prediction >= c) * self.testing_matrix)
            self.h_area = np.count_nonzero(self.prediction >= c) * self.hx * \
                          self.hy / 10 ** -6  # km^2


class Model:
    def __init__(self, models=None):
        """Supraclase Model"""
        self.models = []

    def prepare_stkde(self):
        """

        Returns
        -------

        """
        data = self.data
        geometry = [Point(xy) for xy in zip(np.array(data[['x']]),
                                            np.array(data[['y']]))]
        data = gpd.GeoDataFrame(data, crs=2276, geometry=geometry)
        data.to_crs(epsg=3857, inplace=True)

        data['x'] = data['geometry'].apply(lambda x: x.x)
        data['y'] = data['geometry'].apply(lambda x: x.y)

        data = data.sample(n=self.stkde.sn, replace=False, random_state=0)
        data.sort_values(by=['date'], inplace=True)
        data.reset_index(drop=True, inplace=True)

        # División en training data (X_train) y testing data (y)
        X_train = data[data["date"] <= self.stkde.start_prediction]
        X_test = data[data["date"] > self.stkde.start_prediction]
        X_test = X_test[
            X_test["date"] < self.stkde.start_prediction + datetime.timedelta(
                days=self.stkde.lp)]

        return X_train, X_test

    def prepare_promap(self):

        df = self.data

        if len(df) >= self.promap.n:
            # print(f'\nEligiendo {self.promap.n} datos...')
            df = df.sample(n=self.promap.n,
                           replace=False,
                           random_state=250499)
            df.sort_values(by=['date'], inplace=True)
            df.reset_index(drop=True, inplace=True)

        # print("\nGenerando dataframe...")

        geometry = [Point(xy) for xy in zip(
            np.array(df[['x']]),
            np.array(df[['y']]))
                    ]

        geo_data = gpd.GeoDataFrame(df,  # gdf de incidentes
                                    crs=2276,
                                    geometry=geometry)

        geo_data.to_crs(epsg=3857, inplace=True)

        df['x_point'] = geo_data['geometry'].x
        df['y_point'] = geo_data['geometry'].y

        # División en training y testing data

        X = df[df["date"] <= self.promap.start_prediction]
        y = df[df["date"] > self.promap.start_prediction]
        y = y[y["date"] < self.promap.start_prediction + datetime.timedelta(
            days=self.promap.lp)]

        return X, y

    def prepare_rfr(self, mode='train', label='default'):
        """Prepara el set de datos correspondiente para entrenar RFR y
        predecir para un set dado

        Parameters
        ----------
        mode : str
            Tipo de X, y a retornar. Elegir entre {'train', 'test'}
        label : str
            Establece la forma en la que se generará la label:
            {'default', 'weighted'}. En el caso de 'weighted', se usa
            el atributo .l_weights de la clase para ponderar las
            labels asociadas
        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
        """
        # y en 'label_weights' es una sola columna que corresponde a la
        # suma ponderada de las columnas (considerar division por número
        # de celdas en las capas [o distancia])
        # [('Incidents_i', self.model.weeks[-2])] for i in range(8)
        if self.rfr.X is None:  # Sin las labels generadas
            self.rfr.generate_X()

        if mode == 'train':
            # print("\nPreparing Training Data for RFR...")
            # First three weeks of October
            X = self.rfr.X.loc[
                :,
                reduce(lambda a, b: a + b,
                       [[(f'Incidents_{i}', week)]
                        for i in range(self.rfr.n_layers)
                        for week in self.rfr.weeks[:-2]]
                       )
                ]
            # Last week of October
            # y = self.model.X_train.loc[:, [('Incidents_0', self.model.weeks[-2])]]
            y = self.rfr.X.loc[
                :, [(f'Incidents_{i}', self.rfr.weeks[-2])
                    for i in range(self.rfr.n_layers)]
                ]
            if label == 'default':
                # Cualquier valor != 0 en la fila produce que la celda sea
                # 'Dangerous' = 1
                y[('Dangerous', '')] = y.T.any().astype(int)
            else:
                if self.rfr.l_weights is not None:
                    w = self.rfr.l_weights
                else:
                    w = np.array([1 / (l + 1)
                                  for l in range(self.rfr.n_layers)])
                y[('Dangerous', '')] = y.dot(w)  # Ponderación con los pesos
            y = y[('Dangerous', '')]  # Hace el .drop() del resto de las cols

        else:
            # print("Preparing Testing Data for RFR...")
            # Nos movemos una semana adelante
            X = self.rfr.X.loc[
                :,
                reduce(lambda a, b: a + b,
                       [[(f'Incidents_{i}', week)]
                        for i in range(self.rfr.n_layers)
                        for week in self.rfr.weeks[1:-1]]
                       )
                ]
            y = self.rfr.X.loc[
                :, [(f'Incidents_{i}', self.rfr.weeks[-1])
                    for i in range(self.rfr.n_layers)]
                ]
            if label == 'default':
                y[('Dangerous', '')] = y.T.any().astype(int)
            else:
                if self.rfr.l_weights is not None:
                    w = self.rfr.l_weights
                else:
                    w = np.array([1 / (l + 1)
                                  for l in range(self.rfr.n_layers)])
                y[('Dangerous', '')] = y.dot(w)
            y = y[('Dangerous', '')]
        return X, y

    def prepare_data(self):
        pass

    def add_model(self, m):
        """Añade el modelo a self.models

        Parameters
        ----------
        m : {STKDE, RForestRegressor, ProMap}
        """
        self.models.append(m)

    def print_parameters(self):
        """
        Printea los hiperparámetros de cada uno de los modelos activos
        en self.stkde, self.promap, self.rfr

        """
        if self.stkde:
            self.stkde.print_parameters()
        if self.promap:
            self.promap.print_parameters()
        if self.rfr:
            self.rfr.print_parameters()

    def fit(self):
        if self.stkde:
            self.stkde.fit(*self.pp.preparing_data('STKDE'))
        if self.rfr:
            if self.rfr.read_X:
                self.rfr.X = pd.read_pickle('predictivehp/data/X.pkl')
            if self.rfr.read_data:
                self.rfr.data = pd.read_pickle('predictivehp/data/data.pkl')

            self.rfr.fit(*self.pp.preparing_data(
                'RForestRegressor', mode='train', label='default'
            )
                         )
        if self.promap:
            self.promap.fit()

    def predict(self):
        if self.stkde:
            self.stkde.predict()
        if self.promap:
            self.promap.predict(*self.pp.preparing_data('ProMap'))
        if self.rfr:
            X_test, _ = self.pp.preparing_data(
                'RForestRegressor', mode='test', label='default'
            )
            self.rfr.predict(X_test)

    def validate(self, c=None):
        """
        Calcula la cantidad de incidentes detectados para los hotspots
        afines.

        nro incidentes, area x cada c

        Parameters
        ----------
        c : {None, float, list}
          Umbral de score

        Returns
        -------
        int
          ji
        """
        for m in self.models:
            m.validate(c)

    def detected_incidences(self):
        for m in self.models:
            print(f"{m.name}: {m.d_incidents}")

    def hotspot_area(self):
        for m in self.models:
            if m.name == 'STKDE':
                continue
            print(f"{m.name}: {m.h_area}")

    def plot_hr(self):
        pass

    def plot_pai(self):
        pass

    def store(self, file_name='model.data'):
        pass

    # def plot_heatmap(self, c=0, show_score=True, incidences=False,
    #                  savefig=False, fname='', **kwargs):
    #     """
    #
    #     Parameters
    #     ----------
    #     c : {int, float, list}
    #       Si un solo elemento es especificado, luego celdas con un
    #       score superior a c son consideradas parte del hotspot. El
    #       resto quedan de color transparente.
    #
    #       Si es una lista, tenemos hotpots con scores entre c[0] y c[1],
    #       el resto de las celdas quedan transparentes.
    #     incidences : bool
    #       Determina si se plotean o no los incidentes en el heatmap.
    #
    #       Green marker para hits, Red marker para misses.
    #     """
    #     for m in self.models:
    #         m.heatmap(c=c, show_score=show_score, incidences=incidences,
    #                   savefig=savefig, fname=fname, **kwargs)


def create_model(data=None, shps=None,
                 start_prediction=date(2017, 11, 1), length_prediction=7,
                 use_stkde=False, use_promap=False, use_rfr=False):
    """

    Parameters
    ----------
    data
    shps
    start_prediction
    length_prediction
    use_stkde
    use_promap
    use_rfr

    Returns
    -------
    Model
    """
    m = Model()
    if use_stkde:
        stkde = STKDE(shps=shps, start_prediction=start_prediction,
                      length_prediction=length_prediction)
        m.add_model(stkde)
    if use_promap:
        promap = ProMap(shps=shps, start_prediction=start_prediction,
                        length_prediction=length_prediction)
        m.add_model(promap)
    if use_rfr:
        rfr = RForestRegressor(data_0=data, shps=shps,
                               start_prediction=start_prediction)
        m.add_model(rfr)
    return m


if __name__ == '__main__':
    pass