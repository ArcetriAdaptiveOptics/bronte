from bronte import startup
from bronte.scao.flattening_runner import FlatteningRunner
import numpy as np 

def main(ftag):
    
    ff = startup.flattening_startup()
    flat = np.zeros(1920*1152)
    ff.deformable_mirror.set_shape(flat)
    gain = -0.3 * np.ones(ff.N_MODES_TO_CORRECT) 
    bad_mode_index = 187 
    gain[bad_mode_index] = -0.3*0.5
    ff.INT_GAIN = gain
    
    fr = FlatteningRunner(ff)
    Nsteps = 100
    fr.run(Nsteps)
    fr.save_telemetry(ftag)