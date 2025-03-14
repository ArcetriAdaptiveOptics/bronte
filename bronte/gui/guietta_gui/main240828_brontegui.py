from pysilico import camera
from guietta import Gui, PG, ___, III, _, M, Ax, Quit
#import pyqtgraph as pg
from time import sleep


class BronteGui():
    
    def __init__(self):
        
        self._gui = Gui(
            ['Host name:'           ,  '__hostname__'     ,   'Port:'    ,  '__port__' ,   ['Connect']  ],
            [M('display')           ,        ___          ,      ___     ,      ___    ,      ___       ],
            [    III                ,        III          ,      III     ,      III    ,      III       ],
            [    III                ,        III          ,      III     ,      III    ,      III       ],
            [    III                ,        III          ,      III     ,      III    ,      III       ],
            ['Exposure Time [ms]:'  ,   'actual_texp'     ,  '__texp__'  ,       _     ,       _        ],
            ['Frame Rate [Hz]:'     ,   'actual_fps'      ,  '__fps__'   ,       _     ,    Quit        ]
        )
        
        self._gui.hostname = 'localhost'
        self._gui.port = 7110
        self._gui.display = None
        self._wfs_camera = None
        
        
        self._gui.events(
            [   _  ,   _    ,   _                                         ,   _  , self._create_device],
            [   _  ,   _    ,   _                                         ,   _  ,      _             ],
            [   _  ,   _    ,   _                                         ,   _  ,      _             ],
            [   _  ,   _    ,   _                                         ,   _  ,      _             ],
            [   _  ,   _    ,   _                                         ,   _  ,      _             ],
            [   _  ,   _    ,   ('returnPressed',self._set_exposure_time) ,   _  ,      _             ],
            [   _  ,   _    ,   ('returnPressed',self._set_fps)           ,   _  ,  _                 ],
        )
        
    def _create_device(self, *args):
        hostname = self._gui.hostname
        port = int(self._gui.port)
        self._wfs_camera = camera(hostname, port)
         
    def _set_exposure_time(self,*args):
        self._wfs_camera.setExposureTime(float(self._gui.texp))
        #self._gui.actual_texp = self._wfs_camera.exposureTime(timeoutInSec = 0.5)
        
    def _set_fps(self, *args):
        self._wfs_camera.setFrameRate(float(self._gui.fps))
        #self._gui.actual_fps = self._wfs_camera.getFrameRate(timeoutInSec = 0.5)
    
    
    # def update_camera_parameters(self, *args):
    #
    #     self._gui.actual_texp = self._wfs_camera.exposureTime()
    #     self._gui.actual_fps = self._wfs_camera.getFrameRate()
        
    def _update_frame(self, *args):
        
        if self._wfs_camera is None:
            return
        
        self._gui.actual_texp = self._wfs_camera.exposureTime(timeoutInSec = 0.5)
        self._gui.actual_fps = float('%.2f' % self._wfs_camera.getFrameRate(timeoutInSec = 0.5))
        
        im = self._wfs_camera.getFutureFrames(1,1).toNumpyArray()
        with Ax(self._gui.display) as ax:
            ax.set_title('SHWFS Frame')
            imm = ax.imshow(im)
            ax.figure.colorbar(imm)
        

    
    def run(self, *args):
        self._gui.timer_start(self._update_frame, 1)
        self._gui.run()
        