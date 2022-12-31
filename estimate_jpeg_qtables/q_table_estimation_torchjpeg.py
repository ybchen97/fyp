import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
import torchjpeg.codec
from fast_histogram import histogram1d
from utils import *


def compute_dct_coefficient_histogram(dct_blocks, bin_range=50):
    k_factor_list = dct_blocks.reshape(-1, 64).transpose()
    
    histograms = []
    for k in k_factor_list:
        h = histogram1d(k, bins=bin_range*2+1, range=(-bin_range,bin_range+1))
        histograms.append(h)

    return np.array(histograms)

def compression_simulation(im, n, mth_q_table, bin_range=100):
    """
    1. Cropping image need to break JPEG blocks. Hence, it needs to be done in
    the spatial domain. PIL read image, then crop.
    2. Spatial domain, blockify, level shift, do DCT transform to yield DCT
    coefficients
    3. Simulate compressions with n constant matrices using DCT coefficients
        1. Perform quantization with chosen constant quantization table. This
        procedure is lossy. 
        2. Dequantize coefficients by multiplying with the same constant
        quantization table.
        3. Repeat steps 2.C.a and 2.C.b with m-th quantization table
        4. After the "simulated" compressions (steps 2.C.a-b), compute the DCT
        coefficient histogram.
    
    Args:
        im: PIL image.
        n: Number of constant matrices to try.
        mth_q_table: Quantization table used in the m-th compression.
    Returns:
        List of list of histograms, of shape (64, n, histogram_size). 
    """
    ############################################
    # Step 1: Crop image to break JPEG block alignment
    ############################################
    patch = crop_leave4(im)

    ############################################
    # Step 2: Blockify, level shift and DCT transform
    ############################################
    pixels = (to_tensor(patch) * 255) - 128  # convert to range [0-255] and level shift
    pixel_blocks = torchjpeg.dct.blockify(pixels.unsqueeze(0), 8)
    dct_blocks = torchjpeg.dct.block_dct(pixel_blocks)
    
    ############################################
    # Step 3: Create n constant matrices and do compression (quantize and dequantize)
    ############################################
    k_hists_compare = []
    
    for i in range(1, n+1):
        # Create constant matrix with element i
        # M_i is just a length 64 constant array since jpeg compress takes in a 1d array
        M_i = torch.ones((8,8)) * i
        
        ############################################
        # Step 3.1 & 3.2: "Compress" using M_i
        ############################################
        quantized_dct_blocks = torch.round(dct_blocks / M_i)  # lossy step
        dequantized_dct_blocks = quantized_dct_blocks * M_i
        
        ############################################
        # Step 3.3: "Compress" again using mth_q_table
        ############################################
        quantized_dct_blocks = torch.round(dequantized_dct_blocks / mth_q_table)  # lossy step
        dequantized_dct_blocks = quantized_dct_blocks * mth_q_table
        
        ############################################
        # Step 3.4: Compute 64 histograms from dequantized dct coefficients
        ############################################
        k_hists = compute_dct_coefficient_histogram(np.array(dequantized_dct_blocks.squeeze()), bin_range=bin_range)
        k_hists_compare.append(k_hists)
        
    k_hists_compare = np.array(k_hists_compare).transpose(1,0,2)
    return k_hists_compare

def estimate_q_table(im_path, n, bin_range):
    """
    Args:
        im_path: Path to m-compressed image. Assume image is grayscale only.
        n: Greatest value assumed by quantization factors
    """
    # Step 1: Compute reference k DCT histograms for m-compressed image    
    _, mth_q_table, quantized_dct_blocks_ref, _ = torchjpeg.codec.read_coefficients(im_path)
    dct_blocks_ref = np.array((quantized_dct_blocks_ref * mth_q_table).squeeze()).reshape(-1, 8, 8)
    k_hists_ref = compute_dct_coefficient_histogram(dct_blocks_ref, bin_range=bin_range)
    
    # Step 2: Simulate compressions to get k factor histograms
    im = Image.open(im_path)
    k_hists_compare = compression_simulation(im, n, mth_q_table, bin_range=bin_range)
    
    # Step 3: Compute closest histogram using chi-square histogram distance
    estimation = np.zeros(64)
    for i in range(64):
        hist, idx = get_closest_histogram(k_hists_ref[i], k_hists_compare[i])
        best_n = idx + 1  # since n starts from 1
        estimation[i] = best_n
    
    return estimation.reshape((8,8))
