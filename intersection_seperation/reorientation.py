import numpy as np
import itertools as it

def closure_approximation(ori_tensor2, dim=1, zero_comp=2):
    """
    Computes the closure approximation of a 4th-order orientation tensor based on the given
    2nd-order tensor.

    Parameters:
    -----------
    ori_tensor2 : array, shape (3, 3)
        Given 2nd-order tensor for all the inclusions.
    dim: int, optional, default=1
        Dimension specifier (1, 2, or other).
    zero_comp: int, optional, default=2
        Index for zero-component adjustment.

    Returns:
    --------
    closure_ori_tensor4 : array, shape (3x3x3x3)
        Closure_approximation of 4th-order orientation tensor of all the inclusions.
    """
    # If dim == 1, compute the closure using a tensor outer product.
    if dim == 1:
        return np.einsum('ij,kl->ijkl', ori_tensor2, ori_tensor2)

    # If dim == 2, perform a specific approximation for a fourth-order tensor.
    elif dim == 2:
        # Define coefficients alpha and beta, and the identity matrix i2.
        alpha, beta = - 1 / 24, 1 / 6
        i2 = np.eye(2)
        # Remove the row and column corresponding to zero_comp from ori_tensor2.
        ori_tensor2 = np.delete(ori_tensor2, zero_comp, axis=0)
        ori_tensor2 = np.delete(ori_tensor2, zero_comp, axis=1)
        # Adjusted second-order tensor temp_a.
        temp_a = ori_tensor2 + i2

        # Compute components of the fourth-order tensor.
        temp_1 = alpha * np.einsum('ij,kl->ijkl', i2, i2)
        temp_2 = (2.0 * (alpha - beta) * 0.5) * (np.einsum('ik,jl->ijkl', i2, i2) + np.einsum('il,jk->ijkl', i2, i2))
        temp_3 = beta * (np.einsum('ij,kl->ijkl', i2, ori_tensor2) + np.einsum('ij,kl->ijkl', ori_tensor2, i2))
        temp_41 = 0.5 * (np.einsum('ik,jl->ijkl', temp_a, temp_a) + np.einsum('il,jk->ijkl', temp_a, temp_a))
        temp_42 = 0.5 * (np.einsum('ik,jl->ijkl', ori_tensor2, ori_tensor2) + np.einsum('il,jk->ijkl', ori_tensor2, ori_tensor2))
        temp_4 = 2.0 * beta * (temp_41 - temp_42)
        temp_upa = temp_1 + temp_2 + temp_3 + temp_4

        # Initialize the fourth-order tensor to store the final result.
        ori_tensor4 = np.zeros((3, 3, 3, 3))
        for i, j, k, l in it.product(range(3), repeat=4):
            if i != zero_comp and j != zero_comp and k != zero_comp and l != zero_comp:
                i1 = i - 1 if i > zero_comp else i
                j1 = j - 1 if j > zero_comp else j
                k1 = k - 1 if k > zero_comp else k
                l1 = l - 1 if l > zero_comp else l
                ori_tensor4[i, j, k, l] = temp_upa[i1, j1, k1, l1]
        return ori_tensor4

    # For other dimensions, compute the closure using a default method.
    else:
        i2 = np.eye(3)
        alpha, beta = -1 / 35.0, 1 / 7
        temp_a = ori_tensor2 + i2
        temp_1 = alpha * np.einsum('ij,kl->ijkl', i2, i2)
        temp_2 = (2 * (alpha - beta) * 0.5) * (np.einsum('ik,jl->ijkl', i2, i2) + np.einsum('il,jk->ijkl', i2, i2))
        temp_3 = beta * (np.einsum('ij,kl->ijkl', i2, ori_tensor2) + np.einsum('ij,kl->ijkl', ori_tensor2, i2))
        temp_41 = 0.5 * (np.einsum('ik,jl->ijkl', temp_a, temp_a) + np.einsum('il,jk->ijkl', temp_a, temp_a))
        temp_42 = 0.5 * (np.einsum('ik,jl->ijkl', ori_tensor2, ori_tensor2) + np.einsum('il,jk->ijkl', ori_tensor2, ori_tensor2))
        temp_4 = 2 * beta * (temp_41 - temp_42)
        return temp_1 + temp_2 + temp_3 + temp_4

def triangle_area(p1, p2, p3):
    """
    Calculate the area of a triangle given its three vertices.

    Parameters:
    -----------
    p1, p2, p3 : tuple or list
        The coordinates of the three vertices of the triangle, each as a 2D point (x, y).

    Returns:
    --------
    float
        The area of the triangle.

    Formula:
    --------
    The area is computed using the determinant method:
        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    """
    return abs(p1[0] * (p2[1] - p3[1]) +
               p2[0] * (p3[1] - p1[1]) +
               p3[0] * (p1[1] - p2[1])) * 0.5


def find_coefficients(pn):
    """
    Calculate the barycentric coordinates (weights) for a point within a triangle.

    Parameters:
    -----------
    pn : np.ndarray
        A 2D point (x, y) for which barycentric coordinates are to be calculated.

    Returns:
    --------
    np.ndarray
        A 1D array containing the barycentric coordinates (w1, w2, w3),
        where each weight corresponds to one vertex of the triangle.

    Explanation:
    ------------
    The function computes the barycentric coordinates of a point `pn` relative to
    the vertices of a triangle using the following method:
      - Compute the area of the entire triangle (S0).
      - Compute the area of the sub-triangles formed by the point `pn` and
        two of the triangle's vertices (S1, S2, S3).
      - Normalize these areas by dividing each sub-area by the total area (S0).
    """
    # Define the vertices of the triangle
    p1 = np.array([1, 0])
    p2 = np.array([0.5, 0.5])
    p3 = np.array([1 / 3, 1 / 3])

    # Calculate the area of the full triangle
    s0 = triangle_area(p1, p2, p3)

    # Calculate the areas of the sub-triangles
    s1 = triangle_area(pn, p2, p3)
    s2 = triangle_area(p1, pn, p3)
    s3 = triangle_area(p1, p2, pn)

    # Calculate barycentric coordinates
    return np.array([s1 / s0, s2 / s0, s3 / s0])

# Function: trans_matrix
# Purpose: Computes eigenvalues and eigenvectors of a tensor and sorts them in descending order.
def trans_matrix(ori_tensor2):
    """
    Compute the eigenvalues and eigenvectors of a tensor and sort them in descending order.

    Parameters:
    -----------
    ori_tensor2 : np.ndarray
        A 2x2 or 3x3 symmetric matrix (tensor) for which eigenvalues and eigenvectors are computed.

    Returns:
    --------
    tuple:
        - eig_values : np.ndarray
            A 1D array containing eigenvalues sorted in descending order.
        - eig_vectors : np.ndarray
            A 2D array where each column corresponds to the eigenvector of the eigenvalue
            at the same index, sorted in descending order.
    """
    # Compute eigenvalues and eigenvectors
    eig_values, eig_vectors = np.linalg.eig(ori_tensor2)

    # Sort eigenvalues in descending order and adjust eigenvectors accordingly
    sorted_indices = np.argsort(eig_values)[::-1]  # Get indices to sort in descending order
    eig_values = eig_values[sorted_indices]  # Sort eigenvalues
    eig_vectors = eig_vectors[:, sorted_indices]  # Reorder eigenvectors accordingly

    return eig_values, eig_vectors

def ori_tensor4_recon(ori_tensor2):
    """
    Reconstruct the 4th-order orientation tensor based on the given 2nd-order orientation tensor.

    Parameters:
    -----------
    ori_tensor2 : np.ndarray, shape (3, 3)
        The given 2nd-order orientation tensor of all the inclusions.

    Returns:
    --------
    recon_orien_t : np.ndarray, shape (3, 3, 3, 3)
        Reconstructed 4th-order orientation tensor of all the inclusions.
    """
    # Step 1: Compute eigenvalues and eigenvectors of the 2nd-order tensor
    eig_values, eig_vectors = trans_matrix(ori_tensor2)

    # Step 2: Calculate the point PN based on eigenvalues (for barycentric calculation)
    pn = np.array([eig_values[0], eig_values[1]])

    # Step 3: Compute barycentric coefficients (alpha) for closure approximation
    alpha = find_coefficients(pn)

    # Step 4: Generate basis tensors A1, A2, and A3 using closure approximations
    a1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    a2 = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0]])
    a3 = np.array([[1 / 3, 0, 0], [0, 1 / 3, 0], [0, 0, 1 / 3]])
    a1_closure = closure_approximation(a1, 1, 0)
    a2_closure = closure_approximation(a2, 2, 2)
    a3_closure = closure_approximation(a3, 3, 0)

    # Step 5: Compute the weighted sum of the basis tensors
    approx_tensor = alpha[0] * a1_closure + alpha[1] * a2_closure + alpha[2] * a3_closure

    # Step 6: Transform the reconstructed tensor using eigenvectors
    recon_orien_t = np.zeros((3, 3, 3, 3))
    for i, j, k, l in it.product(range(3), repeat=4):
        for i1, j1, k1, l1 in it.product(range(3), repeat=4):
            recon_orien_t[i, j, k, l] += (
                    eig_vectors[i, i1] * eig_vectors[j, j1] * eig_vectors[k, k1]
                    * eig_vectors[l, l1] * approx_tensor[i1, j1, k1, l1])

    return recon_orien_t

def current_ori_tensor4(inc_ori_vecs):
    """
    Calculate the 4th-order orientation tensor based on the inclusion orientation vectors.

    Parameters:
    -----------
    inc_ori_vecs : numpy.ndarray
        Array of shape (NumInc, 3) where NumInc is the number of inclusions,
        and each row is a 3D orientation vector for an inclusion.

    Returns:
    --------
    orientation_tensor : numpy.ndarray
        A 4th-order tensor (3x3x3x3) representing the orientation tensor.
    """
    # Initialize the 4th-order orientation tensor with zeros
    curt_ori_tensor4 = np.zeros((3, 3, 3, 3))
    # Update the 4th-order tensor: A_ijkl = v_i * v_j * v_k * v_l / N
    for i, j, k, l in it.product(range(3), repeat=4):
        for n in range(inc_ori_vecs.shape[0]):
            v = inc_ori_vecs[n]
            curt_ori_tensor4[i, j, k, l] += v[i] * v[j] * v[k] * v[l] / inc_ori_vecs.shape[0]
    # Return the computed 4th-order orientation tensor
    return curt_ori_tensor4

def orivector_optimization(inc_ori_vecs, pred_ori_tensor4, alpha=1.0, max_iter=1e3, tolerance=1e-2):
    """
    Optimize orientation vectors of all the inclusions to match a predefined 4th-order orientation tensor.

    Parameters:
    -----------
    inc_ori_vecs : np.ndarray, shape (num_inc, 3)
        Array where each row is a 3D orientation vector for `num_inc` inclusions.

    pred_ori_tensor4 : np.ndarray, shape (3, 3, 3, 3)
        Target 4th-order orientation tensor to optimize towards.

    alpha : float, optional, default=0.5
        Learning rate for gradient descent.

    max_iter : float, optional, default=1e5
        Maximum number of iterations for the optimization loop.

    tolerance : float, optional, default=1e-4
        Convergence tolerance. Optimization stops when the potential falls below
        `tolerance**2`.

    Returns:
    --------
    tuple:
        - np.ndarray: Optimized orientation vectors of all the inclusions, shape (num_inc, 3).
        - int: Number of iterations actually executed.
    """
    iter_count = 0
    for iter_num in range(int(max_iter)):
        # Compute the residual tensor between the predicted and current orientation tensors
        temp_ori_tensor4 = pred_ori_tensor4 - current_ori_tensor4(inc_ori_vecs)

        # Compute intermediate tensors and gradients
        temp_1 = -np.eye(3) + np.einsum('ni,nj->nij', inc_ori_vecs, inc_ori_vecs)
        temp_2 = np.einsum('ni,nj,nk->nijk', inc_ori_vecs, inc_ori_vecs, inc_ori_vecs)
        temp_3 = np.einsum('ijkl,nlkj->ni', temp_ori_tensor4, temp_2)
        gradients = np.einsum('nij,nj->ni', temp_1, temp_3)

        # Calculate the potential function
        potential = np.sum(temp_ori_tensor4**2)

        # Update orientation vectors using gradient descent
        inc_ori_vecs -= alpha * gradients

        # Normalize each orientation vector
        inc_ori_vecs /= np.linalg.norm(inc_ori_vecs, axis=1, keepdims=True)

        iter_count = iter_num + 1
        # Check for convergence
        if potential <= tolerance ** 2:
            print(f"orientation optimized")
            break

    return inc_ori_vecs, int(iter_count)

def optimized_ori_angles(inc_ori_vecs, angles):
    """
    Calculate orientation angles (theta, phi, and psi) for a set of input vectors.

    Parameters:
    -----------
    inc_ori_vecs : array, shape (NumInc, 3)
        Each row is a 3D orientation vector for NumInc inclusions.
    angles: array, shape (NumInc, 3)
        orientation angles of NumInc inclusions

    Returns:
    --------
    np.ndarray
        A 2D array of shape (N, 3), where each row contains the calculated angles:
        - Column 1: Phi (azimuthal angle, in radians)
        - Column 2: Theta (polar angle, in radians)
        - Column 3: Psi (rotation angle, in radians)
    """
    xv, yv, zv = inc_ori_vecs[:, 0], inc_ori_vecs[:, 1], inc_ori_vecs[:, 2]
    # theta, phi, psi
    theta = np.arccos(zv)
    phi = np.arctan2(yv, xv)
    phi %= np.pi * 2
    psi = angles[:, 2]
    #
    return np.column_stack((phi, theta, psi))
