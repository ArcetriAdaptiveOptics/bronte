import numpy as np
import matplotlib.pyplot as plt
from bronte.oao_school.utils.package_data import set_data_root_dir, InfraredExampleData
from bronte.oao_school.utils.image_processing import make_master_image, print_roi_mean_values

EB_folder = "C:\\Users\\labot\\Desktop\\TP OAO\\oao24\\data"
set_data_root_dir(EB_folder)

def main():
    

    background_image = InfraredExampleData.get_camera_dark_data()
    #background_image = np.median(background_image)
    cl_raw_image_cube = InfraredExampleData.get_close_loop_data_cube()
        
    cl_master = make_master_image(cl_raw_image_cube, np.median(background_image))
    
    ol_raw_image_cube = InfraredExampleData.get_open_loop_data_cube()    
    ol_master = make_master_image(ol_raw_image_cube, np.median(background_image))
    
    ol_ima = ol_master[200:340, 290:430]
    cl_ima = cl_master[200:340, 290:430]
    
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))  # Adjusting figure size for better layout
    fig.suptitle('Long Exposure PSF')
    
    # Plot the first image and its colorbar (Open Loop)
    im0 = axs[0].imshow(np.log10(np.clip(ol_ima, 0, None) + 1), cmap='inferno')
    axs[0].title.set_text('Open Loop')
    fig.colorbar(im0, ax=axs[0])  # Add colorbar to the first plot
    
    # Plot the second image and its colorbar (Close Loop)
    im1 = axs[1].imshow(np.log10(np.clip(cl_ima, 0, None) + 1), cmap='inferno')
    axs[1].title.set_text('Close Loop')
    fig.colorbar(im1, ax=axs[1])  # Add colorbar to the second plot
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout so that title doesn't overlap with the subplots
    plt.show()
    
    #noise estimation
    # bkg noise estimation 
    # select rois with no source
    
    Num_of_roi = 50
    roi_size = 50
    
    roi_noise_vector = np.zeros(Num_of_roi)
    
    yr = np.random.randint(roi_size, background_image.shape[0]-roi_size, Num_of_roi)
    xr = np.random.randint(roi_size, background_image.shape[1]-roi_size, Num_of_roi)
    
    for idx in np.arange(Num_of_roi):
        
        roi = background_image[yr[idx] : yr[idx] + roi_size, xr[idx] : xr[idx] + roi_size]
        roi_noise_vector[idx] = roi.std()
    
    bkg_noise_adu = np.median(roi_noise_vector)
    
    
    
    return bkg_noise_adu, ol_ima, cl_ima
