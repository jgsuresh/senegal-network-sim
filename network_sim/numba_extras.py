import numpy as np
from numba import njit


# def find_unique_rows(matrix):
#     # Sort the matrix and find unique rows in pure Python mode
#     sorted_matrix = np.sort(matrix, axis=0)
#     unique_rows = np.unique(sorted_matrix, axis=0)
#
#     # Pass the unique rows to the Numba function
#     return find_unique_rows_numba(unique_rows)
#
# @njit
# def find_unique_rows_numba(unique_rows):
#     # Create a boolean mask to identify the indices of the unique rows in the original matrix
#     mask = np.zeros(unique_rows.shape[0], dtype=bool)
#     for row in unique_rows:
#         mask |= np.all(unique_rows == row, axis=1)
#
#     # Use the mask to extract the unique rows from the original matrix
#     return unique_rows[mask]


@njit
def find_unique_rows(matrix):
    # List to hold indices of unique rows
    unique_indices = []

    # Iterate over each row
    for i in range(matrix.shape[0]):
        unique = True
        # Check this row against all previous rows deemed unique
        for j in unique_indices:
            if np.array_equal(matrix[i], matrix[j]):
                unique = False
                break
        if unique:
            unique_indices.append(i)

    # Convert unique_indices to a NumPy array
    unique_indices = np.array(unique_indices)

    # Extract the unique rows using the collected indices
    return matrix[unique_indices]