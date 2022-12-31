# ref: https://github.com/getsanjeev/compression-DCT/blob/master/zigzag.py

# Zigzag scan of a matrix
# Argument is a two-dimensional matrix of any size,
# not strictly a square one.
# Function returns a 1-by-(m*n) array,
# where m and n are sizes of an input matrix,
# consisting of its items scanned by a zigzag method.

import numpy as np

def zigzag(input):
    #initializing the variables
    #----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]
    
    i = 0

    output = np.zeros(( vmax * hmax), dtype=int)
    #----------------------------------

    while ((v < vmax) and (h < hmax)):
        
        if ((h + v) % 2) == 0:                 # going up
            
            if (v == vmin):
                output[i] = input[v, h]        # if we got to the first line

                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        

                i = i + 1

            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                output[i] = input[v, h] 
                v = v + 1
                i = i + 1

            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                #print(3)
                output[i] = input[v, h] 
                v = v - 1
                h = h + 1
                i = i + 1
        
        else:                                    # going down

            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                output[i] = input[v, h] 
                h = h + 1
                i = i + 1
        
            elif (h == hmin):                  # if we got to the first column
                output[i] = input[v, h] 

                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1

                i = i + 1

            elif ((v < vmax -1) and (h > hmin)):     # all other cases
                output[i] = input[v, h] 
                v = v + 1
                h = h - 1
                i = i + 1

        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            output[i] = input[v, h] 
            break

    return output

def get_zigzag_indices():
    col_idx, row_idx = np.meshgrid(np.arange(8),np.arange(8))
    return zigzag(row_idx), zigzag(col_idx)

