from bronte.types.scao_loop_simulator import ScaoLoopSimulator
from bronte.types.atmospheric_simulator_initialiser import AtmosphericSimulatorInitialiser
from bronte.startup import startup

def main():
    
    
    atmo_sim = AtmosphericSimulatorInitialiser()
    factory = startup()
    
    # defining parameters
    seeing = 0.65
    wind_speed_list = [5.5, 5.5]
    wind_direction_list = [0, 0]
    
    gs_source_name_list = ['on_axis_source', 'lgs1_source']
    gs_polar_coords_list = [[0.0, 0.0], [45.0, 0.0]]
    gs_magnitude_list = [8, 5]
    gs_heights_list = [float('inf'), 90000]
    gs_wl_in_nm_list = [750, 589]
    
    pupil_diameter_in_meters = 40
    pupil_diameter_in_pixel = 2 * factory.slm_pupil_mask.radius()
    outer_scale_in_m = 23
    height_list = [600, 20000] # [m] layer heights at 0 zenith angle
    Cn2_list = [1 - 0.119977, 0.119977] # Cn2 weights (total must be eq 1)
    
    # setting atmo parameters
    atmo_sim.set_atmospheric_parameters(seeing,
                                        wind_speed_list,
                                        wind_direction_list)
    
    atmo_sim.set_guide_star_sources(gs_source_name_list,
                                    gs_polar_coords_list,
                                    gs_magnitude_list,
                                    gs_heights_list,
                                    gs_wl_in_nm_list)
    
    atmo_sim.set_atmospheric_evolution_and_propagation_parameters(
        pupil_diameter_in_meters,
        pupil_diameter_in_pixel,
        outer_scale_in_m,
        height_list,
        Cn2_list)
    
    scao_sim = ScaoLoopSimulator(atmo_sim)
    
    scao_sim.initialize_groups()
    scao_sim.enable_display_in_loop(True)
    scao_sim.run_simulation(factory, Nsteps = 10)