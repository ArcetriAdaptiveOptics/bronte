import numpy as np 
from bronte import startup
from bronte.scao.open_loop_runner_old import OpenLoopRunner
from bronte.scao.telemetry.scao_telemetry_data_analyser import ScaoTelemetryDataAnalyser

def main():
    
    ff = startup.flattening_startup()
    #applying a shape flat shape on the slm
    flat = np.zeros(1920*1152)
    ff.deformable_mirror.set_shape(flat)
    #or an offset
    modal_offset = ScaoTelemetryDataAnalyser.load_modal_offset('250317_165500')
    offset_cmd = ff.slm_rasterizer.m2c(-modal_offset)
    ff.deformable_mirror.set_shape(offset_cmd)
    
    olr = OpenLoopRunner(ff)
    olr.run(500)
    olr.save_telemetry('pippo_tag')