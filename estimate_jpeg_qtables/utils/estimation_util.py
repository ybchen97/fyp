import io
import numpy as np


def crop_leave4(im):
    return im.crop((4, 4, im.size[0]-4, im.size[1]-4))

def chi2_hist_distance(h1, h2):
    distance = 0
    for b in range(len(h1)):
        if h1[b] == 0 and h2[b] == 0:
            continue
        distance += (h1[b] - h2[b])**2 / (h1[b] + h2[b])
 
    return distance

def get_closest_histogram(reference, histogram_list):
    smallest_dist = np.inf
    smallest_idx = -1
    closest_hist = None
    
    for idx, h in enumerate(histogram_list):
        dist = chi2_hist_distance(h, reference)
        
        if dist < smallest_dist:
            smallest_dist = dist
            smallest_idx = idx
            closest_hist = h
    
    return closest_hist, smallest_idx

def jpeg_compress_to_buffer(im, q_table):
    """
    Compresses a PIL image `im` using quantization table `q_table` and saves the result
    in memory to a buffer.
    Args:
        im: PIL image.
        q_table: A list of 64 integers in normal array order, not zigzag fashion.
    Returns:
        The buffer where the image is saved
    """    
    # This compression step quantizes the image
    buffer = io.BytesIO()
    im.save(buffer, format='jpeg', qtables=[q_table], subsampling=0) # no chroma-subsampling
    return buffer

