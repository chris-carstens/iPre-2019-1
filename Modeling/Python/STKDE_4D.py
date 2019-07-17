# state file generated using paraview version 5.6.1

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# trace generated using paraview version 5.6.1
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

# import the simple module from the paraview
from paraview.simple import *

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.GetRenderWindow().SetFullScreen(True)  # Para la p. completa
renderView1.ViewSize = [784, 1100]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [2505251.4078454673, 6981929.658190809, 336000.0]
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

# create a new 'XML Structured Grid Reader'
densities = XMLStructuredGridReader(FileName=[
    '/Users/msmendozaelguera/Desktop/iPre/Modeling/Python/STKDE grid.vts'])
densities.PointArrayStatus = ['density', 'y_day']

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

# create a new 'GDAL Vector Reader'
dallasMap = GDALVectorReader(
        FileName='/Users/msmendozaelguera/Desktop/iPre/Modeling/Data/shapefiles'
                 '/STREETS.shp')

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

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from densities
densitiesDisplay = Show(densities, renderView1)

# get color transfer function/color map for 'density'
densityLUT = GetColorTransferFunction('density')
densityLUT.RGBPoints = [0.0, 0.278431372549, 0.278431372549, 0.858823529412,
                        5.418939399286293e-13, 0.0, 0.0, 0.360784313725,
                        1.0799984117458697e-12, 0.0, 1.0, 1.0,
                        1.625681819785888e-12, 0.0, 0.501960784314, 0.0,
                        2.1637862916031283e-12, 1.0, 1.0, 0.0,
                        2.7056802315317578e-12, 1.0, 0.380392156863, 0.0,
                        3.247574171460387e-12, 0.419607843137, 0.0, 0.0,
                        3.7894681113890166e-12, 0.878431372549, 0.301960784314,
                        0.301960784314]
densityLUT.ColorSpace = 'RGB'
densityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'density'
densityPWF = GetOpacityTransferFunction('density')
densityPWF.Points = [0.0, 0.0, 0.5, 0.0, 3.7894681113890166e-12, 1.0, 0.5, 0.0]
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
dallasMapDisplay.AmbientColor = [0.5019607843137255, 0.5019607843137255,
                                 0.5019607843137255]
dallasMapDisplay.ColorArrayName = ['POINTS', '']
dallasMapDisplay.DiffuseColor = [0.7137254901960784, 0.7137254901960784,
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
densityLUTColorBar.Position = [0.031037827352085365, 0.6636363636363637]
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

Interact()
