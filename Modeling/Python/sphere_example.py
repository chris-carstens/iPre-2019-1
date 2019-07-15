# state file generated using paraview version 5.6.1

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

from paraview.simple import *

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
# renderView1.ViewSize = [1031, 902]
renderView1.GetRenderWindow().SetFullScreen(True)
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.StereoType = 0
renderView1.CameraPosition = [2.5690645513219206, 1.9685550853787432,
                              0.5926729858858314]
renderView1.CameraViewUp = [-0.14297548298243398, -0.10955548546443827,
                            0.9836440448000441]
renderView1.CameraParallelScale = 0.8516115354228021
renderView1.Background = [0, 0, 0]
renderView1.OSPRayMaterialLibrary = materialLibrary1

# init the 'GridAxes3DActor' selected for 'AxesGrid'
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

# create a new 'Sphere'
sphere = Sphere()

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from sphere
sphereDisplay = Show(sphere, renderView1)

# get color transfer function/color map for 'Normals'
normalsLUT = GetColorTransferFunction('Normals')
normalsLUT.RGBPoints = [0.9999999771868981, 0.0, 0.0, 0.5625,
                        0.9999999811692579, 0.0, 0.0, 1.0, 0.9999999902718126,
                        0.0, 1.0, 1.0, 0.999999994823081, 0.5, 1.0, 0.5,
                        0.9999999993743494, 1.0, 1.0, 0.0, 1.000000008476904,
                        1.0, 0.0, 0.0, 1.0000000130281723, 0.5, 0.0, 0.0]
normalsLUT.ColorSpace = 'RGB'
normalsLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
sphereDisplay.Representation = 'Point Gaussian'
sphereDisplay.ColorArrayName = ['POINTS', 'Normals']
sphereDisplay.LookupTable = normalsLUT
sphereDisplay.OSPRayScaleArray = 'Normals'
sphereDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
sphereDisplay.SelectOrientationVectors = 'None'
sphereDisplay.ScaleFactor = 0.1
sphereDisplay.SelectScaleArray = 'None'
sphereDisplay.GlyphType = 'Arrow'
sphereDisplay.GlyphTableIndexArray = 'None'
sphereDisplay.GaussianRadius = 0.005
sphereDisplay.ShaderPreset = 'Gaussian Blur'
sphereDisplay.SetScaleArray = ['POINTS', 'Normals']
sphereDisplay.ScaleTransferFunction = 'PiecewiseFunction'
sphereDisplay.OpacityArray = ['POINTS', 'Normals']
sphereDisplay.OpacityTransferFunction = 'PiecewiseFunction'
sphereDisplay.DataAxesGrid = 'GridAxesRepresentation'
sphereDisplay.SelectionCellLabelFontFile = ''
sphereDisplay.SelectionPointLabelFontFile = ''
sphereDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
sphereDisplay.DataAxesGrid.XTitleFontFile = ''
sphereDisplay.DataAxesGrid.YTitleFontFile = ''
sphereDisplay.DataAxesGrid.ZTitleFontFile = ''
sphereDisplay.DataAxesGrid.XLabelFontFile = ''
sphereDisplay.DataAxesGrid.YLabelFontFile = ''
sphereDisplay.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
sphereDisplay.PolarAxes.PolarAxisTitleFontFile = ''
sphereDisplay.PolarAxes.PolarAxisLabelFontFile = ''
sphereDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
sphereDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# setup the color legend parameters for each legend in this view

# get color legend/bar for normalsLUT in view renderView1
normalsLUTColorBar = GetScalarBar(normalsLUT, renderView1)
normalsLUTColorBar.WindowLocation = 'AnyLocation'
normalsLUTColorBar.Position = [0.07468477206595536, 0.6031042128603105]
normalsLUTColorBar.Title = 'Normals'
normalsLUTColorBar.ComponentTitle = 'Magnitude'
normalsLUTColorBar.TitleFontFile = ''
normalsLUTColorBar.LabelFontFile = ''
normalsLUTColorBar.ScalarBarLength = 0.32999999999999996

# set color bar visibility
normalsLUTColorBar.Visibility = 1

# show color legend
sphereDisplay.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'Normals'
normalsPWF = GetOpacityTransferFunction('Normals')
normalsPWF.Points = [0.9999999771868981, 0.0, 0.5, 0.0, 1.0000000130281723, 1.0,
                     0.5, 0.0]
normalsPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# finally, restore active source
SetActiveSource(sphere)
Interact()
# ----------------------------------------------------------------
