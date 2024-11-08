
class SlopesManager():
    
    def __init__(self, wfs_list):
        
        self._wfs_list = wfs_list
        self._slopes = None
        
        self._compute_combined_slopes_from_each_wfs()
        
    def _compute_combined_slopes_from_each_wfs(self):
        
        slopes = []
        
        for wfs in self._wfs_list:
            slopes.append(wfs._slopes)
            
    def get_combined_slopes(self):
        return self._slopes