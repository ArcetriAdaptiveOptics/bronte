
from bronte import startup
from bronte.telemetry.display_telemetry_data import DisplayTelemetryData
import matplotlib.pyplot as plt
 
def main():
    
    bf = startup.startup()
    
    modal_offset, _ = DisplayTelemetryData.load_modal_offset('250128_144000')
    
    ftag_tip = '250128_150200'
    ftag_tilt = '250128_150800'
    ftag_focus = '250128_151300'
    
    dtd = DisplayTelemetryData(ftag_tip)
    tip_delta_cmds = dtd.get_delta_modal_commands_at_step(-2).copy()
    
    dtd = DisplayTelemetryData(ftag_tilt)
    tilt_delta_cmds = dtd.get_delta_modal_commands_at_step(-2).copy()
    
    dtd = DisplayTelemetryData(ftag_focus)
    focus_delta_cmds = dtd.get_delta_modal_commands_at_step(-2).copy()
    
    plt.figure()
    plt.clf()
    plt.plot(-tip_delta_cmds, 'g.-', label='-dcms (+1um rms Z2)')
    plt.plot(modal_offset, 'm.-', alpha = 0.4, label='modal_offset')
    plt.ylabel('Zernike coefficient [m] rms wavefront')
    plt.xlabel('index')
    plt.xlim(0, 10)
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.plot(-tilt_delta_cmds, 'b.-', label='-dcms (+1um rms Z3)')
    plt.plot(modal_offset, 'm.-', alpha = 0.4, label='modal_offset')
    plt.ylabel('Zernike coefficient [m] rms wavefront')
    plt.xlabel('index')
    plt.xlim(0, 10)
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.plot(-focus_delta_cmds, 'r.-', label='-dcms (+1um rms Z4)')
    plt.plot(modal_offset, 'm.-', alpha = 0.4, label='modal_offset')
    plt.ylabel('Zernike coefficient [m] rms wavefront')
    plt.xlabel('index')
    plt.xlim(0, 10)
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.plot(-tip_delta_cmds - modal_offset, 'g.-', label='-dcms (+1um rms Z2) - modal_offset')
    plt.ylabel('Zernike coefficient [m] rms wavefront')
    plt.xlabel('index')
    plt.xlim(0, 10)
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.plot(-tilt_delta_cmds - modal_offset, 'b.-', label='-dcms (+1um rms Z3) - modal_offset')
    plt.ylabel('Zernike coefficient [m] rms wavefront')
    plt.xlabel('index')
    plt.xlim(0, 10)
    plt.legend(loc='best')
    
    plt.figure()
    plt.clf()
    plt.plot(-focus_delta_cmds - modal_offset, 'r.-', label='-dcms (+1um rms Z4) - modal_offset')
    plt.ylabel('Zernike coefficient [m] rms wavefront')
    plt.xlabel('index')
    plt.xlim(0, 10)
    plt.legend(loc='best')