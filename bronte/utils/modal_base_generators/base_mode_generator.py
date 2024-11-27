import numpy as np

class BaseModesGenerator():
    
    def __init__(self, pupil, covariance_matrix = None):
        self._pupil = pupil
        self._dictDxCache = {}
        self._dictDyCache = {}
        if covariance_matrix is not None:
            self._covariance_matrix = covariance_matrix
        
    def getMode(self):
        pass

    def _computeDerivativeX(self, index):
        
        mode = self.getMode(index)
        #mode is usually a masked array
        #avoid unexpected behaviour while using gradient
        if np.ma.is_masked(mode):
            mode = mode.filled(0)
        #normalized coord, unit radius
        dx = 1/ (self._pupil.radius())
        dmode_dx = np.gradient(mode, axis=1) / dx
        return dmode_dx

    def getDerivativeX(self, index):
        if index not in self._dictDxCache:
            self._dictDxCache[index] = self._computeDerivativeX(index)
        return self._dictDxCache[index]
    
    def getDerivativeXDict(self, indexVector):
        ret = {}
        for index in indexVector:
            ret[index] = self.getDerivativeX(index)
        return ret
    
    def getDerivativeY(self, index):
        if index not in self._dictDyCache:
            self._dictDyCache[index] = self._computeDerivativeY(index)
        return self._dictDyCache[index]
    
    def _computeDerivativeY(self, index):
        
        mode = self.getMode(index)
        ##mode is usually a masked array
        #avoid unexpected behaviour while using gradient
        if np.ma.is_masked(mode):
            mode = mode.filled(0)
        #normalized coord, unit radius
        dy = 1/ (self._pupil.radius())
        dmode_dy = np.gradient(mode, axis=0) / dy      
        return dmode_dy
    
    def getDerivativeYDict(self, indexVector):
        ret = {}
        for index in indexVector:
            ret[index] = self.getDerivativeY(index)
        return ret
    
    def getModeDict(self, indexVector):
        ret = {}
        for index in indexVector:
            ret[index] = self.getMode(index)
        return ret
    