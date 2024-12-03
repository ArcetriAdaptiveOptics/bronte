import numpy as np
import time
from arte.types.zernike_coefficients import ZernikeCoefficients


class InteractionMatrixComputer:
    
    def __init__(self, deformable_mirror, wfs_camera, slope_computer, slm_rasterizer):
        self._dm = deformable_mirror
        self._wfs = wfs_camera
        self._sc = slope_computer
        self._rast = slm_rasterizer
        self._SLM_SLEEP=0.005
        
        
    def acquire(self):
        n_modes = 2
        amp = 1000e-9
        self._n_slopes = self._sc.total_number_of_subapertures()*2
        self._intmat = np.zeros((self._n_slopes, n_modes))

        self._dm.set_shape(np.zeros(1152*1920))
        time.sleep(self._SLM_SLEEP)
        for i in range(n_modes):
            zcv = np.zeros(n_modes)
            zcv[i]=amp
            zc = ZernikeCoefficients.fromNumpyArray(zcv)
            wfz=self._rast.zernike_coefficients_to_raster(zc).toNumpyArray()
            slp=self._apply_and_measure(wfz)
            slm=self._apply_and_measure(-wfz)
            self._intmat[:, i]=(slp-slm)/amp

    def _apply_and_measure(self, wfz):
        self._dm.set_shape(self._rast.reshape_map2vector(wfz))
        time.sleep(self._SLM_SLEEP)
        fr = self._wfs.getFutureFrames(1,1).toNumpyArray()
        self._sc.set_frame(fr.astype(float))
        return self._sc.slopes().reshape(self._n_slopes)


if __name__ == '__main__':
    pass