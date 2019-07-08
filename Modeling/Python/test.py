"""Test Module"""

from paraview.simple import *

sphere = Sphere(ThetaResolution=16, PhiResolution=32)

shrink = Shrink(sphere)

Show(shrink)
Render()
