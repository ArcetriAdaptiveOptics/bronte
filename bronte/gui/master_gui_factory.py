from guietta import Gui, PG, ___, III, _, M, Ax, Quit, G
from bronte.gui.camera_gui_factory import CameraGuiFactory

class MasterGuiFactory():
    
    #timer in seconds
    WFS_TIMER = 1
    CAM_TIMER = 1
    SLM_TIMER = 1
    
    def __init__(self):
        
        self._cam_gui_factory = CameraGuiFactory()
        self._wfs_gui_factory = CameraGuiFactory()
        
        self._cam_gui = self._cam_gui_factory.getGui()
        self._wfs_gui = self._wfs_gui_factory.getGui()
        
        
        self._slm_gui = Gui([['info about slm and display']])
        
        self._master_gui = Gui(
            [ G('WFS') , G('CAM'), G('SLM')]
            )
        
        self._master_gui.WFS = self._wfs_gui
        self._master_gui.CAM = self._cam_gui
        self._master_gui.SLM = self._slm_gui
    
    def run(self, *args):
        
        self._wfs_gui.timer_start(
            self._wfs_gui_factory.update_camera_frame, self.WFS_TIMER)
        
        self._cam_gui.timer_start(
            self._cam_gui_factory.update_camera_frame, self.CAM_TIMER)
        
        self._master_gui.run()