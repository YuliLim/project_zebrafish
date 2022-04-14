"""
This module contains function to visualization of obtained angles in hsv format.
"""
import numpy as np
import math
from skimage.color import rgb2hsv, hsv2rgb


def compute_curve_angles(shape,x,y,derivatives,ind):
    """
    For each pixel of the image compute the tangent in the closest point of the curve - the angle that will be substracted to obtain corrected angles. 
    INPUT: tuple - shape of the image, two 1-d arrays corresponding to x and y coordinates of the curve, 2d-array (derivatives of the curve),
            2d-array (distance transform).
    OUTPUT: 2d-array of curve angles.
    """
        # 2d-array (of the same shape as the image) of the curve derivatives.
    curve_derivatives = np.zeros(shape, dtype=object)
    for i in range(len(x)):
        curve_derivatives[int(y[i]),int(x[i])] = [derivatives[i][0],derivatives[i][1]]

    angles_alpha = np.zeros(shape)
    # loop over y coordinate
    for i in range(shape[0]):
        #loop over x coordinate
        for j in range(shape[1]):

            x_coord = ind[1][i][j]
            y_coord = ind[0][i][j]
            (dx,dy) = curve_derivatives[y_coord][x_coord]

                #get angle
            v_curve = [dx,dy]
            v_curve_u = v_curve / np.linalg.norm(v_curve)

            angle_curve = math.degrees(np.arcsin(-v_curve_u[1]))

            angles_alpha[i,j] = angle_curve
    return angles_alpha

def myfunc(z):
    return np.uint8(255*z)
myfunc_vec = np.vectorize(myfunc)

def convert(shape,angles_zstack,z_layer):
    """
    Map angles to the interval [0,180] for hsv representation.
    INPUT: shape of the image, 3d-array (z-stack of corrected angles), index of chosen for visualization z_layer 
    OUTPUT: 2d-array of angles in [0,180].
    """
    size = shape[0]*shape[1]
    if (angles_zstack.ndim == 2):                                               #case of hsv diagram of curve angles "plot_hsv_diagram()"
        angles = np.reshape(np.nan_to_num(angles_zstack),(size))
    if (angles_zstack.ndim == 3):
        angles = np.reshape(np.nan_to_num(angles_zstack[:,:,z_layer]),size)
    converter = lambda x : (x + 180) if x < 0 else x
    angles_180 = np.array([converter(x) for x in angles])
    angles_180 = np.reshape(angles_180,shape[:2])
    return angles_180

def to_hsv(shape,angles_zstack):
    """
    Represent corrected angles as an HSV image with H = angle/180, S=1, V=1.
    INPUT: shape of the rgb image, 3d-array (z-stack of corrected angles), index of chosen for visualization z_layer 
    OUTPUT: 3d-array - HSV image of corrected angles, 2d-array of curve angles.
    """
    shape = list(shape)
    shape[1] = angles_zstack.shape[1]
    shape = tuple(shape)
   
    z_layer = 4

    angles_180 = convert(shape,angles_zstack,z_layer)

    hsv = np.zeros(shape)

    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            if not np.isnan(angles_zstack[i,j,z_layer]): 
                hsv[i,j,0] = angles_180[i,j]/180
                hsv[i,j,1] = 1
                hsv[i,j,2] = 1

        # python can represent only rgb format          
    rgb_img = hsv2rgb(hsv)
    return myfunc_vec(rgb_img), angles_180

def plot_hsv_diagram(shape,delta,x,y,derivatives,ind):
    """
    Represent curve angles for each pixel as an HSV image with H = angle/180, S=1, V=1. 
    For each pixels it gives the value of angle to substract from the initial angle to obtain the corrected angle.
    INPUT: shape of the rgb image, width of the cutted bords, 3d-array (z-stack of corrected angles), index of chosen for visualization z_layer 
    OUTPUT: 3d-array - HSV image of curve angles.
    """
    angles_alpha = compute_curve_angles(shape[:2],x,y,derivatives,ind)
        # crop the image, delta = 0 corresponds to the original image
    shape = list(shape)
    shape[1] -= 2*delta 
    shape = tuple(shape)
        
    angles_alpha_cropped = np.empty(shape[:2])
    x_end = angles_alpha.shape[1]-delta
    angles_alpha_cropped[:,:] = angles_alpha[:,delta:x_end]
        # map angles to [0,180] for HSV representation
    angles_alpha_180 = convert(shape[:2],angles_alpha_cropped,0)
        # each pixel has some value
    hsv = np.ones(shape)
    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            hsv[i,j,0] = angles_alpha_180[i,j]/180

        # python can represent only rgb format
    rgb_img = hsv2rgb(hsv)
    return myfunc_vec(rgb_img)

