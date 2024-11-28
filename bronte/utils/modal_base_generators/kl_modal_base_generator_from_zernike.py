import numpy as np
from arte.utils.zernike_generator import ZernikeGenerator
    
class KLModesGenerator():
    '''
    Generator of Karhunen-Loève (KL) modal basis from a give pupil and
    covariance matrix (Ca) obtained from a Zernike modal base
    '''
    def __init__(self, pupil, covariance_matrix):
        
        #super().__init__(pupil)
        self._pupil = pupil
        self._check_covriance_martrix(covariance_matrix)
        self._covariance_matrix = covariance_matrix
        self._kl_modes, self._eigen_values = self._compute_kl_modal_base()
        self._zg = ZernikeGenerator(pupil)
        
    # TODO: check the hypothesis of the spectral theorem on covariance_matrix
    def _check_covriance_martrix(self, covariance_matrix):
        
        is_simmetric = (covariance_matrix == covariance_matrix.T).all()
        is_positive = (covariance_matrix >= 0).all()
        
        if is_positive is False and is_simmetric is False:
            error_message = f"Covariance matrix does not satisfy"\
            f"the theorem conditions: \n - positive: {is_positive} "\
            f"\n- simmetric: {is_simmetric} "
            raise ValueError(error_message)
          
    def _compute_kl_modal_base(self):
        '''
        Computes the KL eigen vectors (modes) and values (variances) by diagonalising
        the covariance matrix (Ca). Note: Ca must satisfy the hypothesis of the
        spectral theorem.
        '''
        eigen_values, eigen_vectors = np.linalg.eigh(self._covariance_matrix)
        # Sort eigen values and eigen vectors in descending order
        idx = np.argsort(-eigen_values)
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]
        return eigen_vectors, eigen_values
    
    def getMode(self, index):
        '''
        Returns the KL mode corresponding to the given index.
        '''
        if index < 1 or index > len(self._kl_modes):
            raise ValueError(f"Invalid KL index: {index}")
        coefficients = self._kl_modes[:, index - 1]
        kl_mode = sum(coeff * self._zg.getZernike(i + 1) for i, coeff in enumerate(coefficients))
        return kl_mode
    
    def getDerivativeX(self, index):
        '''
        Returns the X derivative of the specified KL mode.
        '''
        coefficients = self._kl_modes[:, index - 1]
        kl_dx = sum(coeff * self._zg.getDerivativeX(i + 1) for i, coeff in enumerate(coefficients))
        return kl_dx
    
    def _get_derivative_x_numerically(self, index):
        
        mode = self.getMode(index)
        #mode is a masked array
        #avoid unexpected behaviour while using gradient
        if np.ma.is_masked(mode):
            mode = mode.filled(0)
        #normalized coord, unit radius
        dx = 1/ (self._pupil.radius())
        dmode_dx = np.gradient(mode, axis=1) / dx
        #dmode_dy = np.gradient(mode, axis=0) / dy
        return dmode_dx
    
    def _get_derivative_y_numerically(self, index):
        
        mode = self.getMode(index)
        #mode is a masked array
        #avoid unexpected behaviour while using gradient
        if np.ma.is_masked(mode):
            mode = mode.filled(0)
        #normalized coord, unit radius
        dy = 1/ (self._pupil.radius())
        dmode_dy = np.gradient(mode, axis=0) / dy      
        return dmode_dy
    
    def getDerivativeY(self, index):
        '''
        Returns the Y derivative of the specified KL mode.
        '''
        coefficients = self._kl_modes[:, index - 1]
        kl_dy = sum(coeff * self._zg.getDerivativeY(i + 1) for i, coeff in enumerate(coefficients))
        return kl_dy