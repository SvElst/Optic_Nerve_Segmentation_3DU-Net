#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find angle and rotation matrix between two points
"""

import numpy as np
import math
import optparse
from scipy.spatial.transform import Rotation as R


parser = optparse.OptionParser()
parser.add_option('-s','--pOS',action="store",type="float", dest="pOS", nargs=2)        # Centroid left eye
parser.add_option('-d','--pOD',action="store",type="float", dest="pOD", nargs=2)        # Centroid right eye
parser.add_option('-p','--path',action="store",type="string", dest="path") 
options, args = parser.parse_args()

def get_angle(p1,p2):
    """Get the angle of this line with the horizontal axis."""
    dy = (p2[1] - p1[1])
    dx = p2[0] - p1[0]
    theta = math.atan2(dy, dx)
    angle = math.degrees(theta)  # angle is in (-180, 180]
    return angle

angle = get_angle(options.pOS, options.pOD)

"Creat rotation matrix"
r = R.from_euler('z', angle, degrees=True)
mat=r.as_matrix()
ver = np.array(([[0,0,0]]))
hor = np.array([[0],[0],[0],[1]])
rotm = np.hstack([np.vstack([mat, ver]), hor])

# Save rotation matrix and rotation angle
np.savetxt(options.path +'/rotm.mat', rotm)
f=open(options.path+'/angle.txt', "w")
f.write(str(abs(angle)))
f.close