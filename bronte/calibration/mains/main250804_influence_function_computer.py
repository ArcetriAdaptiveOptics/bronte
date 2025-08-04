import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np

import os
from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.data_objects.m2c import M2C
from specula import cpuArray

def compute_and_save_influence_functions():
    """
    Compute zonal influence functions and modal basis for the SCAO tutorial
    Follows the same approach as test_modal_basis.py
    """
    # DM and pupil parameters for VLT-like telescope
    pupil_pixels = 160           # Pupil sampling resolution
    n_actuators = 41             # 41x41 = 1681 total actuators
    telescope_diameter = 8.2     # meters (VLT Unit Telescope)

    # Pupil geometry
    obsratio = 0.14              # 14% central obstruction
    diaratio = 1.0               # Full pupil diameter

    # Actuator geometry - aligned with test_modal_basis.py
    circGeom = True              # Circular geometry (better for round pupils)
    angleOffset = 0              # No rotation

    # Mechanical coupling between actuators
    doMechCoupling = False       # Enable realistic coupling
    couplingCoeffs = [0.31, 0.05] # Nearest and next-nearest neighbor coupling

    # Actuator slaving (disable edge actuators outside pupil)
    doSlaving = True             # Enable slaving (very simple slaving)
    slavingThr = 0.1             # Threshold for master actuators

    # Modal basis parameters
    r0 = 0.15                    # Fried parameter at 500nm [m]
    L0 = 25.0                    # Outer scale [m]
    zern_modes = 5               # Number of Zernike modes to include
    oversampling = 1             # No oversampling

    # Computation parameters
    dtype = specula.xp.float32   # Use current device precision

    print("Computing zonal influence functions...")
    print(f"Pupil pixels: {pupil_pixels}")
    print(f"Actuators: {n_actuators}x{n_actuators} = {n_actuators**2}")
    print(f"Telescope diameter: {telescope_diameter}m")
    print(f"Central obstruction: {obsratio*100:.1f}%")
    print(f"r0 = {r0}m, L0 = {L0}m")

    # Step 1: Generate zonal influence functions
    influence_functions, pupil_mask = compute_zonal_ifunc(
        pupil_pixels,
        n_actuators,
        circ_geom=circGeom,
        angle_offset=angleOffset,
        do_mech_coupling=doMechCoupling,
        coupling_coeffs=couplingCoeffs,
        do_slaving=doSlaving,
        slaving_thr=slavingThr,
        obsratio=obsratio,
        diaratio=diaratio,
        mask=None,
        xp=specula.xp,
        dtype=dtype,
        return_coordinates=False
    )

    # Print statistics
    n_valid_actuators = influence_functions.shape[0]
    n_pupil_pixels = specula.xp.sum(pupil_mask)

    print(f"\nZonal influence functions:")
    print(f"Valid actuators: {n_valid_actuators}/{n_actuators**2} ({n_valid_actuators/(n_actuators**2)*100:.1f}%)")
    print(f"Pupil pixels: {int(n_pupil_pixels)}/{pupil_pixels**2} ({float(n_pupil_pixels)/(pupil_pixels**2)*100:.1f}%)")
    print(f"Influence functions shape: {influence_functions.shape}")

    # Step 2: Generate modal basis (KL modes)
    print(f"\nGenerating KL modal basis...")

    kl_basis, m2c, singular_values = make_modal_base_from_ifs_fft(
        pupil_mask=pupil_mask,
        diameter=telescope_diameter,
        influence_functions=influence_functions,
        r0=r0,
        L0=L0,
        zern_modes=zern_modes,
        oversampling=oversampling,
        if_max_condition_number=None,
        xp=specula.xp,
        dtype=dtype
    )

    print(f"KL basis shape: {kl_basis.shape}")
    print(f"Number of KL modes: {kl_basis.shape[0]}")

    kl_basis_inv = np.linalg.pinv(kl_basis)

    # Step 3: Create output directory
    os.makedirs('calibration', exist_ok=True)
    os.makedirs('calibration/ifunc', exist_ok=True)
    os.makedirs('calibration/m2c', exist_ok=True)

    # Step 4: Save using SPECULA data objects
    print(f"\nSaving influence functions and modal basis...")

    # Create IFunc object and save
    ifunc_obj = IFunc(
        ifunc=influence_functions,
        mask=pupil_mask
    )
    ifunc_obj.save('calibration/ifunc/tutorial_ifunc.fits')
    print("✓ tutorial_ifunc.fits (zonal influence functions)")

    # Create M2C object for mode-to-command matrix and save
    m2c_obj = M2C(
        m2c=m2c
    )
    m2c_obj.save('calibration/m2c/tutorial_m2c.fits')
    print("✓ tutorial_m2c.fits (KL modal basis)")

    # inverse influence function object for modal analysis
    print("Saving inverse modal base...")
    ifunc_inv_obj = IFuncInv(
        ifunc_inv=kl_basis_inv,
        mask=pupil_mask
    )
    ifunc_inv_obj.save('calibration/ifunc/tutorial_base_inv.fits')
    print("✓ tutorial_base_inv.fits (inverse modal base)")

    # Step 5: Optional visualization
    try:
        import matplotlib.pyplot as plt
        
        print("\nGenerating visualization...")
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(cpuArray(singular_values['S1']), 'o-', label='IF Covariance')
        plt.semilogy(cpuArray(singular_values['S2']), 'o-', label='Turbulence Covariance')
        plt.xlabel('Mode number')
        plt.ylabel('Singular value')
        plt.title('Singular values of covariance matrices')
        plt.legend()
        plt.grid(True)
        
        # move to CPU / numpy for plotting if required
        kl_basis = cpuArray(kl_basis)
        pupil_mask = cpuArray(pupil_mask)
        
        # Plot some modes
        max_modes = min(16, kl_basis.shape[0])
        
        # Create a mask array for display
        mode_display = np.zeros((max_modes, pupil_mask.shape[0], pupil_mask.shape[1]))
        
        # Place each mode vector into the 2D pupil shape
        idx_mask = np.where(pupil_mask)
        for i in range(max_modes):
            mode_img = np.zeros(pupil_mask.shape)
            mode_img[idx_mask] = kl_basis[i]
            mode_display[i] = mode_img
        
        # Plot the reshaped modes
        n_rows = int(np.round(np.sqrt(max_modes)))
        n_cols = int(np.ceil(max_modes / n_rows))
        plt.figure(figsize=(18, 12))
        for i in range(max_modes):
            plt.subplot(n_rows, n_cols, i+1)
            plt.imshow(mode_display[i], cmap='viridis')
            plt.title(f'Mode {i+1}')
            plt.axis('off')
        plt.tight_layout()
        
        plt.show()

    except ImportError:
        print("Matplotlib not available - skipping visualization")

    print(f"\nInfluence functions and modal basis computation completed!")
    print(f"Files saved in: {os.path.abspath('calibration/')}")
    print(f"\nFiles created:")
    print(f"  tutorial_ifunc.fits  - Zonal influence functions ({n_valid_actuators} actuators)")
    print(f"  tutorial_m2c.fits    - KL modal basis ({kl_basis.shape[0]} modes)")

    # Step 6: Test loading the saved files
    print(f"\nTesting file loading...")

    try:
        # Test IFunc loading
        loaded_ifunc = IFunc.restore('calibration/ifunc/tutorial_ifunc.fits')
        assert loaded_ifunc.influence_function.shape == influence_functions.shape
        print("✓ IFunc loading test passed")

        # Test M2C loading
        loaded_m2c = M2C.restore('calibration/m2c/tutorial_m2c.fits')
        assert loaded_m2c.m2c.shape == kl_basis.shape
        print("M2C loading test passed")

    except Exception as e:
        print(f"File loading test failed: {e}")

    return ifunc_obj, m2c_obj