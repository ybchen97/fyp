import jpeglib
import numpy as np
from PIL import Image
from fast_histogram import histogram1d
from utils import *


def compute_dct_coefficient_histogram(im_path, bin_range=50):
    im = jpeglib.read_dct(im_path)
    q_table = im.qt[0]  # read the luminance quantization table
    dct_coef = im.Y.reshape(-1, 8, 8).transpose(0, 2, 1) * q_table[None,:]  # tranpose because jpeglib reads wrongly
    k_factor_list = dct_coef.reshape(-1, 64).transpose()
        
    histograms = []
    for k in k_factor_list:
        h = histogram1d(k, bins=bin_range*2+1, range=(-bin_range,bin_range+1))
        histograms.append(h)

    return np.array(histograms)

def estimate_q_table(im_path, n, bin_range):
    """
    Args:
        im_path: Path to m-compressed image. Assume image is grayscale only.
        n: Greatest value assumed by quantization factors
    """
    # Step 1: Compute reference k DCT histograms for m-compressed image
    k_hists_ref = compute_dct_coefficient_histogram(im_path, bin_range=bin_range)
    
    # Step 2: Extract mth q-table of image
    im = Image.open(im_path)
    mth_q_table_lumi = im.quantization[0]  # already in flattened 64 integers
    
    # Step 3: Crop out a patch, leaving 4 pixels on each side to break JPEG block alignment
    patch = crop_leave4(im)
    
    # Step 4: Create n constant matrices and do compression
    k_hists_compare = []
    for i in range(1, n+1):
        # Create constant matrix with element i
        # M_i is just a length 64 constant array since jpeg compress takes in a 1d array
        M_i = np.full((64), i)
        
        # Compress patch using M_i
        C_p_buffer = jpeg_compress_to_buffer(patch, M_i)
        
        # Compress again using mth_q_table
        C_p = Image.open(C_p_buffer)
        C_pp_buffer = jpeg_compress_to_buffer(C_p, mth_q_table_lumi)
        
        with open("C_pp.jpg", "wb") as f:
            f.write(C_pp_buffer.getbuffer())
        
        # Read DCT coefficients of doubly compressed patch and compute histograms
        k_hists = compute_dct_coefficient_histogram('C_pp.jpg', bin_range=bin_range)
        k_hists_compare.append(k_hists)
    
    k_hists_compare = np.array(k_hists_compare).transpose(1,0,2)
    
    # Step 5: Compute closest histogram using chi-square histogram distance
    estimation = np.zeros(64)
    for i in range(64):
        _, idx = get_closest_histogram(k_hists_ref[i], k_hists_compare[i])
        best_n = idx + 1  # since n starts from 1
        estimation[i] = best_n
    
    return estimation.reshape((8,8))

