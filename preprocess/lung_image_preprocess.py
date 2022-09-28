import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import glob
from PIL import Image
from datetime import date
import matplotlib.pyplot as plt
from pylab import rcParams
import SimpleITK as sitk
from PIL import Image
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import cv2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--window_level', type=int) 
args= parser.parse_args()

def proc_slice(image, wl, ww,slope,intercept):
    #ds = pydicom.read_file(image, force=True)
    #image = ds.pixel_array # CT value 512x512
 
    #slice_n = ds.InstanceNumber
    #lungslide
    
    image = slope * image + intercept #HU
    minbound = wl - ww/2
    maxbound = wl + ww/2
    image = (image - minbound) / (maxbound - minbound)
    image[image>1] = 1.
    image[image<0] = 0.
    image *= 255 #512x512
    image=image.astype('uint8')
    return image



npz_list=glob.glob('/raid/data/yanglab/ILD/MSH_voxel/*npz*')
name_list=[i.split('/')[-1] for i in npz_list]
name_list=[i.replace('.npz','') for i in name_list]



WL_lung = args.window_level
WW_lung = 1500.0
for k in range(len(npz_list)):
    sample = np.load(npz_list[k])
    sample_x= sample['x'] #[a, b] a is slope, b is intercept
    sample_y = sample['y'] #ct values, raw data , 1) lungwindow = y *slope + intercept 2) apply lung window filters
    slope1=sample_x[0][0]
    intercept1=sample_x[0][1]

    image1=sample_y*slope1+intercept1
    image2=sample_y*slope1+intercept1

    
    s=int(len(sample_y)/2)
    binary=image1[s]< WL_lung
    
    cleared = clear_border(binary)

# Label the image
    label_image = label(cleared)

    # Keep the labels with 2 largest areas
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
             if region.area < areas[-2]:
                    for coordinates in region.coords:                
                        label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    # Closure operation with disk of radius 12
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    
    selem = disk(10)
    binary = binary_closing(binary, selem)
    
    # Fill in the small holes inside the lungs
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)

    # Superimpose the mask on the input image
    get_high_vals = binary == 0
    image1[s][get_high_vals] = 0
    mean1=np.sum(image1[s])

    for i in range(sample_y.shape[0]):

        binary=image2[i]< WL_lung
        cleared = clear_border(binary)

# Label the image
        label_image = label(cleared)

    # Keep the labels with 2 largest areas
        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:                
                        label_image[coordinates[0], coordinates[1]] = 0
        binary = label_image > 0

    # Closure operation with disk of radius 12
        selem = disk(2)
        binary = binary_erosion(binary, selem)
    
        selem = disk(10)
        binary = binary_closing(binary, selem)
    
    # Fill in the small holes inside the lungs
        edges = roberts(binary)
        binary = ndi.binary_fill_holes(edges)

    # Superimpose the mask on the input image
        get_high_vals = binary == 0
        image2[i][get_high_vals] = 0
    
        lung_tissue=np.sum(image2[i])
    
        #print(lung_tissue, mean1)
        if lung_tissue>0.4*mean1:
            pass
        else:
            minbound = WL_lung - WW_lung/2
            maxbound = WL_lung + WW_lung/2
            image2[i] = (image2[i] - minbound) / (maxbound - minbound)
            image2[i][image2[i]>1] = 1.
            image2[i][image2[i]<0] = 0.
            image2[i] *= 255 #512x512
            image2[i]=image2[i].astype('uint8')
            #image2[i][image2[i] == 195]=0
            path='/raid/data/yanglab/ILD/MSH_preprocessed_250/'+name_list[k]+'/'
            if not os.path.exists(path):
                os.makedirs(path)
            filename=os.path.join(path , f"{i:04d}.png")
            #print(filename)
            cv2.imwrite(filename,image2[i])
