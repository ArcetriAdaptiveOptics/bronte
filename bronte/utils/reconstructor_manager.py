
class ReconstructorManager():
    
    def __init__(self, wfs_list):
        
        self._wfs_list = wfs_list
        self._reconstructor = None
        self._compute_combined_reconstructor_from_each_wfs()
        
    def _compute_combined_reconstructor_from_each_wfs(self):
        
        reconstructors = []
        
        for wfs in self._wfs_list():
            reconstructors.append(wfs._reconstructor)
    
    def get_combined_reconstructor(self):
        return self._reconstructor