import numpy as np 

class DataCubeCleaner():
    
    def __init__(self):
        pass
    
   
    def get_redCube_from_rawCube(self, raw_cube, master_dark):
        
        Nframes = raw_cube.shape[-1]
        red_cube = np.zeros(raw_cube.shape)
        
        for idx in range(Nframes):
            
            red_cube[:,:,idx] = raw_cube[:,:,idx] - master_dark
        
        red_cube[red_cube<0] = 0
        
        return red_cube

    def get_master_from_rawCube(self, raw_cube, master_dark):
        
        red_cube  = self.get_redCube_from_rawCube(raw_cube, master_dark)
        return np.median(red_cube, axis = -1)