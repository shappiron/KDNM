import numpy as np

# generate degrees and coeffs matrix
def generate_degreee_matrix(N_feat, max_deg,
                            miss_deg_num, scale=5):
    coeff = 2 * np.random.random(size=(max_deg + 1)) - 1.
    m = np.ones((N_feat, max_deg + 1)) * coeff * scale
                                
    i = np.random.randint(0, N_feat, size=(miss_deg_num,))
    j = np.random.randint(0, max_deg+1, size=(miss_deg_num,))
    
    m[i, j] = 0
    return m

def l_norm(data, low, up):
    normed = []
    for i, x in enumerate(data):
        x_hat  = low[i] + (up[i] - low[i]) * x
        normed.append(x_hat)
    return np.asarray(normed)

def polynom_feature(y, degree_mtrx, noise=None):
    """
    y: [y x 1]
    degree: matrix of degrees [number_of_features x max_degree]
    # coeffs: [number_of_features x y + 1 (bias)]
    noise: vector of shape [number_of_features x y] 
    ??? amplitude of noise scaled to amp of feature ???
    """
    N_feat = degree_mtrx.shape[0]
    poly_features = np.empty((N_feat, y.shape[0]))

    for n, row in enumerate(degree_mtrx):
        y_poly = np.zeros(y.shape)
        
        for deg, coeff in enumerate(row):
            y_poly += coeff * y**deg
        
        poly_features[n] = y_poly
        
    if np.any(noise):
        # calc max value for noise
        max_val = np.max(poly_features, axis=1) * noise
        min_val = np.min(poly_features, axis=1) * noise
        # l_norm = lambda l, low, up: [low[j] + (up[j] - low) * x for x in l]
        
        eps = np.random.random(size=poly_features.shape)
        eps = l_norm(eps, min_val, max_val)
        poly_features += eps
        
    return poly_features.T