import numpy as np
from PIL import Image
from minieigen import MatrixX,Vector3
from .datatypes import *

def convert_array_to_colorimg(inp):
    """Returns the img as PIL images"""
    image_arr = inp.copy()
    if image_arr.dtype == np.float32:
        image_arr += 0.5
        image_arr *= 255
        image_arr = image_arr.astype(np.uint8)
    image_arr = image_arr[0:3,:,:]
    image_arr = np.rollaxis(image_arr,0,3)
    return Image.fromarray(image_arr)


def convert_array_to_grayimg(inp):
    """Convert single channel array to grayscale PIL image. """
    arr = inp.copy()
    arr[np.isinf(arr)] = np.nan
    norm_factor = [np.nanmin(arr), np.nanmax(arr)-np.nanmin(arr)]
    
    if norm_factor[1] == 0:
        raise RuntimeError('Cannot convert array.')
    else:
        arr -= norm_factor[0]
        arr /= norm_factor[1]
        arr *= 255
        arr[np.isnan(arr)] = 0
        arr[np.isinf(arr)] = 0
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr)
