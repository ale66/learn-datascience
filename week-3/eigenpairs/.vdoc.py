# type: ignore
# flake8: noqa
def find_eigenpairs(mat):
    """Test the quality of Numpy eigenpairs"""
    n = len(mat)

    # is it squared?
    m = len(mat[0])
    if n==m:
      eig_vals, eig_vects = la.eig(mat)
    else
            eig_vals, eig_vects = la.eig(mat)
    # they come in ascending order, take the last one on the right
    dominant_eig = abs(eig_vals[-1])
    return dominant_eig


