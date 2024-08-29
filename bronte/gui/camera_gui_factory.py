from pysilico import camera
from guietta import Gui, PG, ___, III, _, M, Ax, G
#import pyqtgraph as pg


class CameraGuiFactory():
    
    def __init__(self):
        
        self._camera_control_gui = Gui(
            ['Exposure Time [ms]:' ,   'actual_texp'     ,  '__texp__'  ,       _     ,       _       ],
            ['Frame Rate [Hz]:'    ,   'actual_fps'      ,  '__fps__'   ,       _     ,       _       ],
            )
        
        self._camera_control_gui.actual_texp = 0
        self._camera_control_gui.actual_fps = 0
        
        self._camera_control_gui.events(
            [ _                     , _                  ,('returnPressed',self._set_exposure_time),       _     ,       _       ],
            [ _                     , _                  ,('returnPressed',self._set_fps)          ,       _     ,       _       ],
            )
        
        self._camera_gui = Gui(
             
            [   'Server'       ,'Host Name:'   , '__hostname__' , 'Port:'    , '__port__' , ['Connect'] , ___     ,    ___     ],
            [M('display')      ,     ___       ,  ___         ,  ___        ,  ___       ,  ___        , ___     ,     ___    ],
            ['X coord:'        ,     'x'       , 'Y coord:'     ,     'y'    , 'I [ADU]:' ,     'I'     ,    _    ,     _      ], 
            [G('Controls')     ,     _         ,  _             ,     _      ,      _     , _           , _       ,     _      ],
            )
        
        self._camera_gui.hostname = 'localhost'
        self._camera_gui.port = 7110
        self._camera_gui.display = None
        self._camera = None
        
        self._camera_gui.x = 0
        self._camera_gui.y = 0
        self._camera_gui.I = 0
        self._camera_gui.Controls = self._camera_control_gui
        
        self._camera_gui.events(
            [_, _, _,_,_,self._create_device  ,_, _],
            [_, _, _,_,_,_                        ,_, _],
            [_, _, _,_,_,_                        ,_, _],
            [_, _, _,_,_,_                        ,_, _],
            )
    
    def _create_device(self, *args):
        hostname = self._camera_gui.hostname
        port = int(self._camera_gui.port)
        self._camera = camera(hostname,port)
        
    def _set_exposure_time(self, *args):
        self._camera.setExposureTime(float(self._camera_control_gui.texp))
        
    def _set_fps(self, *args):
        self._camera.setFrameRate(float(self._camera_control_gui.fps))
    
    def update_camera_frame(self, *args):
        
        if self._camera is None:
            return
        
        self._camera_control_gui.actual_texp = self._camera.exposureTime(timeoutInSec = 0.5)
        self._camera_control_gui.actual_fps = float('%.2f' % self._camera.getFrameRate(timeoutInSec = 0.5))
        
        im = self._camera.getFutureFrames(1,1).toNumpyArray()
        with Ax(self._camera_gui.display) as ax:
            #ax.set_title('SHWFS Frame')
            imm = ax.imshow(im)
            ax.figure.colorbar(imm)
            
    def getGui(self):
        return self._camera_gui
            
    def run(self, *args):
        self._camera_gui.timer_start(self.update_camera_frame, 1)
        self._camera_gui.run()
    