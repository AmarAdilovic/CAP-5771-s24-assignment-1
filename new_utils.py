import numpy as np
from numpy.typing import NDArray

"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""

def normalize(data_matrix: NDArray[np.floating]):
    x = data_matrix
    x_min = np.min(data_matrix)
    x_max = np.max(data_matrix)
    return ( x - x_min ) / ( x_max - x_min ) 

# Checks if all values in the given matrix are between 0 and 1
def scale(data_matrix: NDArray[np.floating]):
    isGreaterOrEqualToZero: bool = (0 <= data_matrix).all()
    isLessThanOrEqualToOne: bool = (data_matrix <= 1).all()
    return isGreaterOrEqualToZero and isLessThanOrEqualToOne

# TODO: Remove next three lines
# col = np.arange(-1, data_matrix.shape[1] - 1).reshape(1, data_matrix.shape[1])
# data_matrix = np.append(data_matrix, col, axis=0)
# np.array(['apples', 'foobar', 'cowboy'], dtype=object)

# Need to check that every number is a float point number AND scaled between 0 and 1
def scale_data(data_matrix: NDArray[np.floating]):
    # will need to normalize the data, if the type isn't correct will just print to console
    # type conversions can be done but would be inconsistent
    data_matrix = enforce_matrix_type(data_matrix, np.floating, float)
    if not scale(data_matrix):
        return normalize(data_matrix)

    return data_matrix

def check_matrix_type(matrix: NDArray, dtype):    
    try:
        # Attempt to convert
        copiedMatrix = matrix.astype(dtype)
        if not np.array_equal(matrix, copiedMatrix):
            return False
    except Exception as e:
        return False
    
    return True

def enforce_matrix_type(matrix: NDArray, matrix_dtype, conversion_dtype):    
    if not check_matrix_type(matrix, np.floating):
        try:
            matrix: NDArray[matrix_dtype] = matrix.astype(conversion_dtype)
        except Exception as e:
            print(f"Error converting matrix to dtype: {e}")
    return matrix


# np.random.rand 120,120
# replace X = , y= and n_train 
# should NOT call prepare data where X and Y are the arguments