import os
import json
import PIL.Image
import tempfile
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
from vedo import Mesh, Plotter
import vedo
import tqdm
from typing import Union

# functions ----


def vedo_show(plotter: vedo.Plotter) -> None:
   """
   Show the plotter in an non-interactive window,
  
   Args:
       plotter (vedo.Plotter): The vedo Plotter object to display.


   Returns:
       None


   Details:
       avoids issues of failure to display repeatedly
   """
   with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
       plotter.screenshot(tmp.name)
       img = PIL.Image.open(tmp.name)
       plt.imshow(img)
       plt.show(block=False)


def set_camera_init_position(
       plotter: vedo.Plotter,
       mesh: vedo.Mesh,
       angle: Union[int, float]
   ) -> tuple:
   bounds = mesh.bounds()


   # mesh center and size
   center = (
       np.mean(bounds[0:2]),
       np.mean(bounds[2:4]),
       np.mean(bounds[4:6])
   )
   size = (
       bounds[1] - bounds[0],
       bounds[3] - bounds[2],
       bounds[5] - bounds[4],
   )


   eye_level = .85 * size[1] + bounds[2]
   orbit_radius = 2.2 * max(size)


   angle_rad = np.radians(angle)


   focal_point = (center[0], center[1], center[2])
   camera_position = (
       center[0] + orbit_radius * np.cos(angle_rad),
       eye_level,
       center[2] + orbit_radius * np.sin(angle_rad)
   )


   # set camera ----
   cam = plotter.camera
   cam.position = camera_position
   cam.focal_point = focal_point


   plotter.render()