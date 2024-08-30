import numpy as np

class PsfComputer():
    
    def __init__(self,
                wl = 635e-9,
                pixel_scale = (4.65e-6/250e-3)*(180*60*60/np.pi),
                camera_frame_shape_in_pixel = (1024, 1360)):
        
        self._wl = wl
        self._pupil_pixel_pitch = 9.2e-6 # Dpe/(2*571)
        self._pixel_scale = pixel_scale
        self._camera_frame_shape = camera_frame_shape_in_pixel
        self._diffracion_limited_psf = None
        #for lab application 
        #self._focal_length = 250e-3
        #self._camera_pixel_pitch = 4.65e-6
        
    
    
    def compute_psf_from_phase(self, phase, Npad = 4):
        '''
        phase is a 2D masked array where the mask mimics the pupil
        Godmann formalism:
        Ui: input electric field (set to 1)
        transmission: transmission (acts only on the phase)
        Ut: transmitted electric field 
        the trasmission acts only on the phase
        the input amplitude is set to 1
        '''
        
        #computing transmitted electric field
        Ui = 1
        transmission_amplitude = 1
        transmission = transmission_amplitude * np.exp(1j * phase)
        Ut = transmission * Ui
        
        Ut.fill_value = 0
        Ut.data[Ut.mask == True] = 0
        
        
        #padding transmitted electric field
        padded_frame_size = np.max(phase.shape) * Npad
        padded_Ut = np.zeros((padded_frame_size, padded_frame_size), dtype=complex)
        padded_Ut[0 : Ut.shape[0], 0 : Ut.shape[1]] = Ut
        
        
        if self._diffracion_limited_psf is None:
            #computing diffraction limited electric field
            piston = np.ma.array(data = np.ones(Ut.shape), mask = Ut.mask)
            piston.fill_value = 0
            piston.data[piston.mask == True] = 0
            #padding trasmitted 
            diffraction_limited_Ut = np.zeros(padded_Ut.shape, dtype=complex)
            diffraction_limited_Ut[0 : Ut.shape[0], 0 : Ut.shape[1]] = piston
                        
            self._diffracion_limited_psf = np.abs(np.fft.fftshift(np.fft.fft2(diffraction_limited_Ut))/(self._wl))**2
        
        #computing angular frequencies
        dxi = self._pupil_pixel_pitch 
        deta = dxi 
        
        #in rad
        self._x = np.fft.fftshift(np.fft.fftfreq(padded_Ut.shape[1], dxi)) * self._wl 
        self._y = np.fft.fftshift(np.fft.fftfreq(padded_Ut.shape[0], deta)) * self._wl
        
        #computing psfs
        self._psf = np.abs(np.fft.fftshift(np.fft.fft2(padded_Ut))/(self._wl))**2
        
       

    def get_psf(self):
        
        return self._psf    
        
    def display_normalized_psf(self):
        #in rad 
        dy = np.abs(self._y[-1] - self._y[-2])
        dx = np.abs(self._x[-1] - self._x[-2])
        
        #in rad 
        padded_frame_width = self._psf.shape[1]*dx
        padded_frame_height = self._psf.shape[0]*dy
        
        padded_frame_extent_in_arcsec =[
            -0.5 * (180*60*60/np.pi)*padded_frame_height, \
            0.5 * (180*60*60/np.pi)*padded_frame_height, \
            -0.5 *(180*60*60/np.pi)*padded_frame_width, \
            0.5 * (180*60*60/np.pi)*padded_frame_width
            ]
        
        normalization_value = self._diffracion_limited_psf.max()
        import matplotlib.pyplot as plt
        plt.figure()
        plt.clf()
        plt.imshow(np.log10(self._psf/normalization_value), cmap = 'jet', extent=padded_frame_extent_in_arcsec)
        plt.xlabel('FOV [arcsec]')
        plt.ylabel('FOV [arcsec]')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Normalized PSF')
        plt.xlim(-0.5*self._camera_frame_shape[1]*self._pixel_scale, 0.5*self._camera_frame_shape[1]*self._pixel_scale)
        plt.ylim(-0.5*self._camera_frame_shape[0]*self._pixel_scale, 0.5*self._camera_frame_shape[0]*self._pixel_scale)
    