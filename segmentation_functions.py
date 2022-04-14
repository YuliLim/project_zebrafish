"""
This module contains functions for segmentation of the principal form (the cranial vault). 
"""
from skimage import data
from skimage.exposure import histogram
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import os
from skimage import io
from skimage import data_dir
from skimage.morphology import skeletonize
from skimage.io import imread_collection
import math

from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.morphology import square, opening,closing
from skimage.morphology import medial_axis


# proceed segmentation for image in intensities
def image_segmentation_i(z_stack):
    """
    This function procceds segmentation in several steps: 
    - An image with maximal intensity is chosen from the z_stack of images
    - The image is donoised with gaussian filter
    - Segmentation with watershed algorithm from skimage.segmentation library is proceeded (needs markers and elevation map(topography)),
         markers are chosen manualy as a fraction of mean intensity of the chosen image.
    - Use closing from skimage.morphology to connect close separated parts
    - Clean segmented image from small disconnected parts, usually corresponding to noise
    - Smoose the segmentation with gaussian filter
from skimage.morphology import skeletonize
    - Perform 2 types of skeletonization (for some cases one works better than another): with "skeletonize" and with "medial_axis" from skimage.morphology 
    - Use "distance" from "mdeial_axis" to estimate the width of the segmented form. If the image is classsified as "wide shape case", another segmentation function will be called 

    INPUT: n-array of 2d images
    OUTPUT: skeletonization with medial axis, skeletonization with skeletonize, segmented image, index of the image with max. intensity chosen from z-stack

    """
        #choose image with max. intensity
    sum_intensity = 0
    ind_max = -1
    for i in range(len(z_stack)):
        if sum_intensity < z_stack[i].sum():
            sum_intensity = z_stack[i].sum()
            ind_max = i
            
    mean_intensity = z_stack[ind_max].mean()
    raw_image = z_stack[ind_max]    
    #print('The brightest image', ind_max)

    fig, ax = plt.subplots(nrows = 4, ncols = 2, figsize=(10,15)) 
    
    ax[0][0].set_title('image in intensities')
    ax[0][0].imshow(raw_image)
    
        #denoise
    im_denoise = gaussian(raw_image, sigma=3)  
    ax[0][1].set_title('denoised image')
    ax[0][1].imshow(im_denoise)

        #choose markers
    markers_i = np.zeros_like(im_denoise)
    UB = mean_intensity*0.01
    LB = mean_intensity*0.003
    markers_i[im_denoise < LB] = 1
    markers_i[im_denoise > UB] = 2
    ax[1][0].set_title('Markers')
    ax[1][0].imshow(markers_i)

        #elevation map
    elevation_map_i = sobel(im_denoise)
    #ax[3].imshow(elevation_map_i)
    #ax[3].set_title('Elevation map')

        #watershed
    segmentation = watershed(elevation_map_i, markers_i)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    segmentation_cl = closing(segmentation)

    ax[1][1].set_title('Segmentation')
    ax[1][1].imshow(segmentation_cl)

        # remove small objects
    label_objects, nb_labels = ndi.label(segmentation_cl)
    sizes = np.bincount(label_objects.ravel())
    #print(sizes)
    ax[2][0].set_title('Label objects')
    ax[2][0].imshow(label_objects)
    mask_sizes = sizes > 600
    mask_sizes[0] = 0
    segmentation_cleaned = mask_sizes[label_objects]
    #plt.title("Segmentation cleaned")
    #plt.imshow(segmentation_cleaned)
    segmentation = segmentation_cleaned

        # apply gaussian filter
    segmentation_smooth = gaussian(segmentation_cleaned, sigma=0.8)  # mild smoothing
    ax[2][1].set_title('Segmentation smoothed')
    ax[2][1].imshow(segmentation_smooth)
        
        # perform skeletonization
    skeleton = skeletonize(segmentation_smooth)
    ax[3][0].set_title('Skeleton')
    ax[3][0].imshow(skeleton)
    
        # perform another skeletonization        
    skel, distance = medial_axis(segmentation_smooth, return_distance=True)
    max_dist = np.max(distance)
    mean_dist = np.mean(distance)

    ax[3][1].set_title('New Skeleton')
    ax[3][1].imshow(skel)
    
    fig.tight_layout()   

    #print("Max width = ", max_dist)
    #print("Mean width = ", mean_dist)
    fig.tight_layout()
    
    if (max_dist>40):
        print("Wide shape case")
        skel, skeleton, segmentation, ind_max = image_segmentation_wide(z_stack) 
    
    return skel, skeleton, segmentation, ind_max

# proceed segmentation for image in intensities in case of a wide shape
def image_segmentation_wide(z_stack):
    """
     This function is called if in the "image_segmentation_i" the segmented shape is considered to be wide. 
     It proceeds the same steps as image_segmentation_i, but the markers for the watershed algorithm are 10 times smaller (wide shape usually corresponds to images with high mean intensity)
    """
    sum_intensity = 0
    ind_max = -1
    for i in range(len(z_stack)):
        if sum_intensity < z_stack[i].sum():
            sum_intensity = z_stack[i].sum()
            ind_max = i
            
    mean_intensity = z_stack[ind_max].mean()
    print("Mean intensity",mean_intensity)
    raw_image = z_stack[ind_max]    
    print('The brightest image', ind_max)

    fig, ax = plt.subplots(nrows = 4, ncols = 2, figsize=(10,15)) 
    
    ax[0][0].set_title('image in intensities')
    ax[0][0].imshow(raw_image)
    
    im_denoise = gaussian(raw_image, sigma=3)  
    ax[0][1].set_title('denoised image')
    ax[0][1].imshow(im_denoise)

        #choose markers
    markers_i = np.zeros_like(im_denoise)
    UB = mean_intensity*0.001
    LB = mean_intensity*0.0003
    markers_i[im_denoise < LB] = 1
    markers_i[im_denoise > UB] = 2
    ax[1][0].set_title('Markers')
    ax[1][0].imshow(markers_i)

        #elevation map
    elevation_map_i = sobel(im_denoise)
    #ax[3].imshow(elevation_map_i)
    #ax[3].set_title('Elevation map')

        #watershed
    segmentation = watershed(elevation_map_i, markers_i)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    segmentation_cl = closing(segmentation)

    ax[1][1].set_title('Segmentation')
    ax[1][1].imshow(segmentation_cl)

        # remove small objects
    label_objects, nb_labels = ndi.label(segmentation_cl)
    sizes = np.bincount(label_objects.ravel())
    print(sizes)
    ax[2][0].set_title('Label objects')
    ax[2][0].imshow(label_objects)

    mask_sizes = sizes > 600
    mask_sizes[0] = 0
    segmentation_cleaned = mask_sizes[label_objects]
    #plt.title("Segmentation cleaned")
    #plt.imshow(segmentation_cleaned)
        
        # apply gaussian filter
    segmentation_smooth = gaussian(segmentation_cleaned, sigma=0.8)  # mild smoothing
    ax[2][1].set_title('Segmentation smoothed')
    ax[2][1].imshow(segmentation_smooth)
        
        # perform skeletonization
    skeleton = skeletonize(segmentation_smooth)
    ax[3][0].set_title('Skeleton')
    ax[3][0].imshow(skeleton)
    
        # perform another skeletonization        
    skel, distance = medial_axis(segmentation_smooth, return_distance=True)
    max_dist = np.max(distance)
    mean_dist = np.mean(distance)

    ax[3][1].set_title('New Skeleton')
    ax[3][1].imshow(skel)
    
    fig.tight_layout()   

    print("Max width = ", max_dist)
    print("Mean width = ", mean_dist)
    fig.tight_layout()
    
    segmentation = segmentation_cleaned
    return skel, skeleton, segmentation, ind_max


# proceed segmentation for image in intensities taken R2-filtered image
def image_segmentation_i_filtered(z_stack):
    """
    This function proceeds segmentation for R2-filtered images (the second half of image stack).
    It is not used in the final program, because R2-filtered images have lower intensities and thus are less adapted for segmentation.
    """
    sum_intensity = 0
    ind_max = -1
    for i in range(int(len(z_stack)/2), len(z_stack)):
        if sum_intensity < z_stack[i].sum():
            sum_intensity = z_stack[i].sum()
            ind_max = i
   
    raw_image = z_stack[ind_max]    
    print('Taken R2 filtered image', ind_max)

    fig, ax = plt.subplots(nrows = 7, ncols = 1, figsize=(8,15)) 
    ax[0].set_title('image in intensities')
    ax[0].imshow(raw_image)
    
    im_denoise = gaussian(raw_image, sigma=3)  
    ax[1].set_title('denoised image')
    ax[1].imshow(im_denoise)

        #choose markers
    markers_i = np.zeros_like(im_denoise)
    markers_i[im_denoise <= 0.001] = 1
    markers_i[im_denoise > 0.009] = 2
    ax[2].set_title('Markers')
    ax[2].imshow(markers_i)

        #elevation map
    elevation_map_i = sobel(im_denoise)
    #ax[3].imshow(elevation_map_i)
    #ax[3].set_title('Elevation map')

        #watershed
    segmentation = watershed(elevation_map_i, markers_i)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    segmentation_cl = closing(segmentation)

    ax[3].set_title('Segmentation')
    ax[3].imshow(segmentation_cl)

        # remove small objects
    label_objects, nb_labels = ndi.label(segmentation_cl)
    sizes = np.bincount(label_objects.ravel())
    print(sizes)
    ax[4].set_title('Label objects')
    ax[4].imshow(label_objects)

    mask_sizes = sizes > 600
    mask_sizes[0] = 0
    segmentation_cleaned = mask_sizes[label_objects]
    #plt.title("Segmentation cleaned")
    #plt.imshow(segmentation_cleaned)
        
        # apply gaussian filter
    segmentation_smooth = gaussian(segmentation_cleaned, sigma=0.8)  # mild smoothing
    ax[5].set_title('Segmentation smoothed')
    ax[5].imshow(segmentation_smooth)
        
        # perform skeletonization
    skeleton = skeletonize(segmentation_smooth)
    ax[6].set_title('Skeleton')
    ax[6].imshow(skeleton)
        
    fig.tight_layout()    
    
    return skeleton, segmentation, ind_max


#proceed segmentation for image in orientations 
def image_segmentation_phi(raw_image):
    """
    This function proceeds segmentation for images in orientations.
    It is not used in the final program, because images in orientations have lower intensities and thus are less adapted for segmentation.
    """
    fig, ax = plt.subplots(nrows = 5, ncols = 1, figsize=(10,15))
    
    ax[0].set_title('image in orientations')
    ax[0].imshow(raw_image)
        #convert to the gray scale and smooth(denoise)
    im_gray = rgb2gray(raw_image)
    im_denoise = gaussian(im_gray, sigma=0.5)  
    ax[1].set_title('denoised image')
    ax[1].imshow(im_denoise)

        #choose markers
    markers_phi = np.zeros_like(im_denoise)
    markers_phi[im_denoise < 0.1] = 1
    markers_phi[im_denoise > 0.1] = 2
    ax[2].set_title('Markers')
    ax[2].imshow(markers_phi)

        #elevation map
    elevation_map_phi = sobel(im_denoise)
    #ax[3].imshow(elevation_map_phi)
    #ax[3].set_title('Elevation map')

        #watershed
    segmentation = watershed(elevation_map_phi, markers_phi)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    segmentation_op = opening(segmentation)

    ax[3].set_title('Segmentation')
    ax[3].imshow(segmentation_op)
        
        # apply gaussian filter
    segmentation_smooth = gaussian(segmentation_op, sigma=0.8)  # mild smoothing
    #ax[5].set_title('Segmentation smoothed')
    #ax[5].imshow(segmentation_smooth)
        
        # perform skeletonization
    skeleton = skeletonize(segmentation_smooth)
    ax[4].set_title('Skeleton')
    ax[4].imshow(skeleton)
        
    fig.tight_layout()  
    
    return skeleton, segmentation