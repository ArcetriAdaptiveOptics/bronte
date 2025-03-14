from plico_dm import deformableMirror
from guietta import Gui, PG, ___, III, _, M, Ax, G
#import pyqtgraph as pg

class SlmGuiFactory():
    
    def __init__(self):
        
        self._slm_control_gui = Gui(
            [['Set Flat']],
            [['Filter TipTipt']],
            )
        
        self._slm_control_gui.events(
            [ self._set_flat_shape],
            [       _             ],
            )
        
        self._slm_gui = Gui(
            ['prova']
            )
        
    def _create_device(self, *args):
        pass
    
    def _set_flat_shape(self, *args):
        pass

    def getGui(self):
        pass
    
    def run(self, *args):
        pass
    
    