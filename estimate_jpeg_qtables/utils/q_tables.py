import numpy as np

STANDARD_LUMI_QT = np.array([
    [16,  11,  10,  16,  24,  40,  51,  61],
    [12,  12,  14,  19,  26,  58,  60,  55],
    [14,  13,  16,  24,  40,  57,  69,  56],
    [14,  17,  22,  29,  51,  87,  80,  62],
    [18,  22,  37,  56,  68, 109, 103,  77],
    [24,  35,  55,  64,  81, 104, 113,  92],
    [49,  64,  78,  87, 103, 121, 120, 101],
    [72,  92,  95,  98, 112, 100, 103,  99]
])

STANDARD_CHRO_QT = np.array([
    [17,  18,  24,  47,  99,  99,  99,  99],
    [18,  21,  26,  66,  99,  99,  99,  99],
    [24,  26,  56,  99,  99,  99,  99,  99],
    [47,  66,  99,  99,  99,  99,  99,  99],
    [99,  99,  99,  99,  99,  99,  99,  99],
    [99,  99,  99,  99,  99,  99,  99,  99],
    [99,  99,  99,  99,  99,  99,  99,  99],
    [99,  99,  99,  99,  99,  99,  99,  99]
])

# Quantization table that appeared the most in CASIA2.0 dataset (~90%)
CASIA_LUMI_QT = np.array([
    [ 6,  4,  4,  6,  9, 11, 12, 16],
    [ 4,  5,  5,  6,  8, 10, 12, 12],
    [ 4,  5,  5,  6, 10, 12, 12, 12],
    [ 6,  6,  6, 11, 12, 12, 12, 12],
    [ 9,  8, 10, 12, 12, 12, 12, 12],
    [11, 10, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12],
    [16, 12, 12, 12, 12, 12, 12, 12]
])

def scale_q_table(q_table, q_factor):
    if q_factor <= 0:
        q_factor = 1
    if q_factor > 100:
        q_factor = 100

    if q_factor < 50:
        scale_factor = 5000 / q_factor
    else:
        scale_factor = 200 - q_factor * 2
    
    scaled_table = (q_table * scale_factor + 50) // 100
    scaled_table[scaled_table <= 0] = 1  # prevent zero division

    return scaled_table.astype(int)
