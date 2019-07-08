"""pv_script"""

from paraview.simple import *

sphere = Sphere(ThetaResolution=32,
                PhiResolution=32)
#shrink = Shrink()

show = Show()
interact = Interact()
