import numpy as np 
from astropy.io import fits
from bronte.package_data import shframes_folder, psf_camera_folder

class CameraMasterMeasurer():
    
    def __init__(self, camera, ftag, texp_in_ms=8, Nframes2average=20):
        
        self._cam = camera
        self._ftag = ftag
        self._texp = texp_in_ms
        self._Nframes = Nframes2average
        self._cam.setExposureTime(self._texp)
        self._master_frame = None
        
    def acquire_master(self):
        
        raw_frames = self._cam.getFutureFrames(self._Nframes).toNumpyArray()
        self._master_frame = np.median(raw_frames, axis=-1)
    
    def get_master(self):
        return self._master_frame
    
    def save_master(self, detector = 'shwfs'):
        if detector is 'shwfs':
            file_name = shframes_folder() / (self._ftag + '.fits')
        else:
            file_name = psf_camera_folder()/ (self._ftag + '.fits')
        hdr = fits.Header()
        hdr['TEXP_MS'] = self._texp
        fits.writeto(file_name, self._master_frame, hdr)
    
    @staticmethod
    def load_master(ftag, detector = 'shwfs'):
        if detector is 'shwfs':
            file_name = shframes_folder() / (ftag + '.fits')
        else:
            file_name = psf_camera_folder()/ (ftag + '.fits')
            
        header = fits.getheader(file_name)
        hduList = fits.open(file_name)
        texp = header['TEXP_MS']
        master_frame = hduList[0].data
        return master_frame, texp