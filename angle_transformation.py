"""
This module contains functions performing the main purpose of the project - correction of the angles of the orientations images. 
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import math
from skimage.morphology import medial_axis
from skimage.filters import gaussian


"""
x - first coordinate of the curve
y - second coordinate of the curve
"""
def distance_transformation(shape,x,y):
    """
    For each pixel of the image the distance transform from ndimage returns the distance from this pixel to the skeleton 
        and the cordinates of the corresponding closest pixel of the skeleton.
    INPUT: shape of the image, 1-d arrays corresponding to x and y coordinates of the curve
    OUTPUT: skeleton created from the curve (for visualization), 2-d array of distances, 
            3d-array (array of two 2-d arrays) of coordiantes of the closest points on the curve.
    """
    curve_skeleton = np.zeros(shape,dtype=bool)
        # create skeleton from the curve coordiantes
    for i in range(len(x)):
        curve_skeleton[int(y[i]),int(x[i])] = True

    curve_skeleton_inversed = curve_skeleton==False

    dist, ind = ndimage.distance_transform_edt(curve_skeleton_inversed, return_indices=True)
  
    return curve_skeleton, dist, ind

def cut_corn(segmentation,dist):
    """
    On some images there are "corns" of important size. 
    Those corns are not alligned with the principal curve and introduce errors in calculation of the statistical characteristics of the corrected images.
    For this reason, we want to "cut" them and to proceed further angle transformations and evaluation of the obtained results for the images without corns.

    This function calculates the width of the segmented shape ("width" returned by "medial axis"). 
    Using distance transform (2d-array of distances "dist"), we can exclude from segmented image the parts that are too far from the "skeleton". 
    Decision criterion "dist[i][j] > max_dist-5" was choosen manually.

    INPUT: 2d boolean array (segmented image), 2d-array (distance transform)
    OUTPUT: 2d boolean array (segmented image after cutting the corn)

    Note: Most images are already homogenous enough and will not be changed by this function. 

    """
    segmentation_smooth = gaussian(segmentation, sigma=0.8)
    skel, width = medial_axis(segmentation_smooth, return_distance=True)

    max_dist = np.max(width)
    segm_sans_corne = segmentation.copy()

    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[1]):
            if segmentation[i][j] == True:
                if dist[i][j] > max_dist-5:
                    segm_sans_corne[i][j] = False
    return segm_sans_corne

def angle_transformation(ind_max, mat, segmentation, derivatives, dist, ind,y,x):
    """
    The correction of the angles are performed in several steps:
    - Initial orientations for the chosen index are collected from external array (mat['PHICH03'])
    - z-array of several layers of images is initialized (the layer with max. intensity +- 4 layers)
    - For all layers in z-array:
        * R2 filter is applied: only pixels with intensities > 0.5 will be proceeded
        * Angle of the tangent is calculated (in degrees) for each point of the curve as -arcsin(y) (as y-axis is inverted). 
        * Obtain corrected angles by substracting the tangent of the curve from the initial angles.
    - Map the corrected angles to the interval [-90,90].
    INPUT: index of the layer with max. intensity (from segmentation), external information array, 2d boolean array (segmented image), 
            2d-array (derivatives of the curve), 2d-array (distance transform), 
            3d-array (array of two 2-d arrays) of coordiantes of the closest points on the curve, 
            two 1-d arrays corresponding to x and y coordinates of the curve.
    OUTPUT: 3d-array (z-stack of initial angles), 3d-array (z-stack of corrected angles)

    """
        # z-layer with maximal intensity
    z_max = ind_max

        # initial orientations extraction
    angles_z = mat['PHICH03'][:,:,z_max]

        # R2 filter mask 
    mask = [mat['R2CH03'][:,:,z_max]>0.5][0]
    angles_filtered = angles_z.copy()
    angles_filtered[mask==False] = np.nan

        # create array of possible z-layers
    z_array = [z_max]
    for i in range(1,5):
        if z_max-i > 0: 
            z_array.append(z_max-i)
        if z_max+i < mat['R2CH03'].shape[2]:
            z_array.append(z_max+i)
    z_array.sort()
    print("Z-layers to correct ", z_array)

        # z-stack of new angles
    shape = mat['R2CH03'].shape[:2]
    shape = shape + (len(z_array),)

        # array of new (corrected) angles
    new_angles_zstack = np.empty(shape)
    new_angles_zstack[:,:,:] = np.nan

        # array of initial angles
    old_angles_zstack = np.empty(shape)
    old_angles_zstack[:,:,:] = np.nan 

        # image-shape array of curve derivatives in corresponding (curve) points
    curve_derivatives = np.zeros(segmentation.shape, dtype=object)
    for i in range(len(x)):
        curve_derivatives[int(y[i]),int(x[i])] = [derivatives[i][0],derivatives[i][1]]

        # cut corn
    segmentation_sans_corne = cut_corn(segmentation,dist)

        #loop over all layers in z_array
    for k, z_i in enumerate(z_array):

        angles_z = mat['PHICH03'][:,:,z_i]

        mask = [mat['R2CH03'][:,:,z_i]>0.5][0]

        angles_filtered = angles_z.copy()
        angles_filtered[mask==False] = np.nan   
        #print("Number of pixels on layer", z_i, "with known angle: ", np.count_nonzero(~np.isnan(angles_filtered)))

        # vector representation of the bord and tangents (curve derivatives)
        v_curve = derivatives[0][0:2]

        # for each pixel find the closest point on the curve, corresponding angle, and correct the angle

        # loop over y coordinate
        for i in range(angles_z.shape[0]):
            #loop over x coordinate
            for j in range(angles_z.shape[1]):

                if segmentation_sans_corne[i][j] == False: 
                    continue

                x_coord = ind[1][i][j]
                y_coord = ind[0][i][j]
                (dx,dy) = curve_derivatives[y_coord][x_coord]

                    #get angle
                v_curve = [dx,dy]
                v_curve_u = v_curve / np.linalg.norm(v_curve)
                angle_curve = math.degrees(np.arcsin(-v_curve_u[1]))
                    #change angle on the image
                new_angles_zstack[i,j,k] = angles_filtered[i][j] - angle_curve
                old_angles_zstack[i,j,k] = angles_filtered[i][j]

    converter = lambda x : (x - 180) if x > 90 else x + 180 if x<-90 else x

    angles_90 =  np.array([converter(x) for x in new_angles_zstack.flatten()])
    new_angles_90 = np.reshape(angles_90,shape)

    angles_90 =  np.array([converter(x) for x in old_angles_zstack.flatten()])
    old_angles_90 = np.reshape(angles_90,shape)

    return old_angles_90, new_angles_90

def crop_image(angles_zstack, delta):
    """
    Due to not perfect segmentation and skeletonization, there are still some problems on the image extremities along the x-axis. 
    To overcome this unfortunate complication, the bords (on x extremities) of the images are cropped.

    INPUT: 3d-array (z-stack of corrected angles), delta - width of the cutted parts.
    OUTPUT: 3d-array (cropped z-stack of corrected angles)
    """
    shape = list(angles_zstack.shape)
    shape[1] -= 2*delta
    shape = tuple(shape)

    angles_zstack_cropped = np.empty(shape)

    x_end = angles_zstack.shape[1]-delta

    for z_i in range(angles_zstack.shape[2]):
        cropped_ang = angles_zstack[:,:,z_i][:,delta:x_end]
        angles_zstack_cropped[:,:,z_i] = cropped_ang
        
    return angles_zstack_cropped
    