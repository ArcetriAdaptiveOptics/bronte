import numpy as np 
    
class LoadVisBackupData():
    
    FDIR = "C:\\Users\\labot\\Desktop\\TP OAO\\TP OAO\\BackupDATA\\"
    
    def __init__(self):
        
        self._close_loop_fname = "ORCA_477.npy"
        self._open_loop_fname = "ORCA_478.npy"
        self._bkg_fname  = "ORCA_494.npy"
        
    
    def get_open_loop_dataCube(self):
        
        return np.load(self.FDIR + self._open_loop_fname)
    
    def get_camera_dark_dataCube(self):
        
        return np.load(self.FDIR + self._bkg_fname)
    
    def get_close_loop_dataCube(self):
        
        return np.load(self.FDIR + self._close_loop_fname)
    
class LoadIrBackUpData():
    
    FDIR = "C:\\Users\\labot\\Desktop\\TP OAO\\TP OAO\\BackupDATA\\"
    
    def __init__(self):
        self._close_loop_fname = "ID_109.npy"
        self._open_loop_fname = "ID_105.npy"
        self._bkg_fname  = "ID_110.npy"
    
    def get_open_loop_data(self):
        
        return np.load(self.FDIR + self._open_loop_fname)
    
    def get_camera_dark_data(self):
        
        return np.load(self.FDIR + self._bkg_fname)
    
    def get_close_loop_dataCube(self):
        
        return np.load(self.FDIR + self._close_loop_fname)

class LoadBackUpData():

    FDIR = "C:\\Users\\labot\\Desktop\\TP OAO\\TP OAO\\BackupDATA\\"

    def __init__(self):

        pass

    def _load_data(self, cl_fname, ol_fname, bkg_fname):

        cl_data = np.load(self.FDIR + cl_fname)
        ol_data = np.load(self.FDIR + ol_fname)
        bkg_data = np.load (self.FDIR + bkg_fname)

        return cl_data, ol_data, bkg_data

    def load_orca_vis_binary_star_data(self):

        close_loop_fname = "ORCA_477.npy"
        open_loop_fname = "ORCA_478.npy"
        bkg_fname  = "ORCA_494.npy"

        cl_data, ol_data, bkg_data = self._load_data(close_loop_fname, open_loop_fname, bkg_fname)

        return cl_data, ol_data, bkg_data


    def load_cred3_ir_single_star_data(self):

        close_loop_fname = "ID_109.npy"
        open_loop_fname = "ID_105.npy"
        bkg_fname  = "ID_110.npy"

        cl_data, ol_data, bkg_data = self._load_data(close_loop_fname, open_loop_fname, bkg_fname)

        return cl_data, ol_data, bkg_data