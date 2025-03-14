from pysilico import camera
from guietta import Gui, PG, ___, III, _, M, Ax, Quit, G
#import pyqtgraph as pg
from time import sleep

class DeviceGui():
    
    def __init__(self):
        
        #self._wfs_gui = Gui( [ Quit ])
        self._wfs_gui_builder()
        
        self._cam_gui = Gui([['info about psf camera and display']])
        self._slm_gui = Gui([['info about slm and display']])
        
        self._master_gui = Gui(
            [ G('WFS') , G('CAM'), G('SLM')]
            )
        self._master_gui.WFS = self._wfs_gui
        self._master_gui.CAM = self._cam_gui
        self._master_gui.SLM = self._slm_gui
        
    
    
    def run(self, *args):
        self._wfs_gui.timer_start(self._update_wfs_frame, 1)
        self._master_gui.run()
    
    def _wfs_gui_builder(self, *args):
        
        self._wfs_control_gui = Gui(
            ['Exposure Time [ms]:' ,   'actual_texp'     ,  '__texp__'  ,       _     ,       _       ],
            ['Frame Rate [Hz]:'    ,   'actual_fps'      ,  '__fps__'   ,       _     ,       _       ],
            )
        
        self._wfs_control_gui.actual_texp = 0
        self._wfs_control_gui.actual_fps = 0
        
        self._wfs_control_gui.events(
            [ _                     , _                  ,('returnPressed',self._set_wfs_exposure_time),       _     ,       _       ],
            [ _                     , _                  ,('returnPressed',self._set_wfs_fps)          ,       _     ,       _       ],
            )
        

        self._wfs_gui = Gui(
             
            [   'Server'       ,'Host Name:'   , '__hostname__' , 'Port:'    , '__port__' , ['Connect'] , _       ,    _       ],
            [M('display')      ,     _         ,  _             ,  _         ,  _         ,  _          , _       ,     _      ],
            ['X coord:'        ,     'x'       , 'Y coord:'     ,     'y'    , 'I [ADU]:' ,     'I'     ,    _    ,     _      ], 
            [G('Controls')     ,     _         ,  _             ,     _      ,      _     , _           , _       ,     _      ],
            )
        
        self._wfs_gui.hostname = 'localhost'
        self._wfs_gui.port = 7110
        self._wfs_gui.display = None
        self._wfs_camera = None
        
        self._wfs_gui.x = 0
        self._wfs_gui.y = 0
        self._wfs_gui.I = 0
        self._wfs_gui.Controls = self._wfs_control_gui
        
        self._wfs_gui.events(
            [_, _, _,_,_,self._create_wfs_device  ,_, _],
            [_, _, _,_,_,_                        ,_, _],
            [_, _, _,_,_,_                        ,_, _],
            [_, _, _,_,_,_                        ,_, _],
            )
        
    def _create_wfs_device(self, *args):
        hostname = self._wfs_gui.hostname
        port = int(self._wfs_gui.port)
        self._wfs_camera = camera(hostname,port)
        
    def _set_wfs_exposure_time(self, *args):
        self._wfs_camera.setExposureTime(float(self._wfs_control_gui.texp))
        
    def _set_wfs_fps(self, *args):
        self._wfs_camera.setFrameRate(float(self._wfs_control_gui.fps))
    
    def _update_wfs_frame(self, *args):
        
        if self._wfs_camera is None:
            return
        
        self._wfs_control_gui.actual_texp = self._wfs_camera.exposureTime(timeoutInSec = 0.5)
        self._wfs_control_gui.actual_fps = float('%.2f' % self._wfs_camera.getFrameRate(timeoutInSec = 0.5))
        
        im = self._wfs_camera.getFutureFrames(1,1).toNumpyArray()
        with Ax(self._wfs_gui.display) as ax:
            ax.set_title('SHWFS Frame')
            imm = ax.imshow(im)
            ax.figure.colorbar(imm)