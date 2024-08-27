import numpy as np 
from pysilico import camera
from guietta import Gui, PG, ___, III, _, M


wfs_camera = None

gui = Gui(
    [   'HostName:'  ,   '__hostname__'    ,   'Port:'    ,   '__port__'  ,     ['Connect']],
    [   M('display') ,      ___          ,      ___     ,      ___    ,      ___       ],
    [       III      ,      III          ,      III     ,      III    ,      III       ],
    [       III      ,      III          ,      III     ,      III    ,      III       ],
    [       III      ,      III          ,      III     ,      III    ,      III       ]
    )


def connect_and_go(gui, *args):
    global wfs_camera
    hostname = gui.hostname
    port = int(gui.port)
    wfs_camera = camera(hostname, port)
    
    # im = cam.getFutureFrames(1,1).toNumpyArray()
    #
    # ax = gui.plot.ax
    # ax.clear()
    # ax.set_title('SHWFS Frame')
    # imm = ax.imshow(im)
    # ax.figure.colobar(imm)
    # ax.figure.canvas.draw()
    



@gui.timer(1)
def timer_update_sh_wfs_image(gui):
    global wfs_camera
    if wfs_camera is None:
        return
    im = wfs_camera.getFutureFrames(1,1).toNumpyArray()
    gui.display = im


    # ax = gui.display.ax
    # ax.clear()
    # ax.set_title('SHWFS Frame')
    # imm = ax.imshow(im)
    # ax.figure.colorbar(imm)
    # ax.figure.canvas.draw()


gui.events(
    [   _  ,   _    ,   _    ,   _  ,     connect_and_go],
    [   _  ,   _    ,   _    ,   _  ,     _             ],
    [   _  ,   _    ,   _    ,   _  ,     _             ],
    [   _  ,   _    ,   _    ,   _  ,     _             ],
    [   _  ,   _    ,   _    ,   _  ,     _             ],
    )
       
        
def main():
    gui.run()
    