import numpy as np 

class DataCubeCleaner():
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_redCube_from_rawCube(raw_cube, master_dark):
        '''
        raw_cube has shape (FrameSizeY,FrameSizeX,Nframes)
        master_dark has shape (FrameSizeY, FrameSizeX)
        '''
        Nframes = raw_cube.shape[-1]
        red_cube = np.zeros(raw_cube.shape)
        
        for idx in range(Nframes):
            
            red_cube[:,:,idx] = raw_cube[:,:,idx].astype(float) - master_dark.astype(float)
        
        red_cube[red_cube<0] = 0.
        
        return red_cube
    
    @staticmethod
    def get_master_from_rawCube(raw_cube, master_dark):
        '''
        raw_cube has shape (FrameSizeY,FrameSizeX,Nframes)
        master_dark has shape (FrameSizeY, FrameSizeX)
        '''
        red_cube  = DataCubeCleaner.get_redCube_from_rawCube(raw_cube, master_dark)
        return np.median(red_cube, axis = -1)
    
    @staticmethod
    def get_mean_from_rawCube(raw_cube, master_dark):
        '''
        raw_cube has shape (FrameSizeY,FrameSizeX, Nframes)
        master_dark has shape (FrameSizeY, FrameSizeX)
        '''
        red_cube  = DataCubeCleaner.get_redCube_from_rawCube(raw_cube, master_dark)
        return np.mean(red_cube, axis = -1)