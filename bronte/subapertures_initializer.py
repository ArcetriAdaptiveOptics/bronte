import numpy as np
import matplotlib.pyplot as plt
from bronte.wfs.slope_computer import PCSlopeComputer
from bronte.wfs.subaperture_set import ShSubapertureSet
from bronte.package_data import subaperture_set_folder, shframes_folder
#from bronte.mains.main250117subaperture_set_initialiser import ShSubapertureSet
#from arte.types.slopes import Slopes


class SubapertureGridInitialiser():

    def __init__(self, wf_reference, pixel_per_sub, Nsub, centroid_threshold=0):

        self._wf_ref = wf_reference
        self._pixel_per_sub = int(pixel_per_sub)
        self._Nsub = int(Nsub)
        self._last_grid_shiftYX = None
        self._original_subaps = None
        self._centroid_threshold = centroid_threshold

    def define_subaperture_set(self, ybll=400, xbll=350):

        frame_shape = self._wf_ref.shape

        bll = np.mgrid[ybll:ybll+self._pixel_per_sub*self._Nsub:self._pixel_per_sub,
                       xbll:xbll+self._pixel_per_sub*self._Nsub:self._pixel_per_sub].reshape((2, self._Nsub*self._Nsub)).T

        self._subaps = ShSubapertureSet.createMinimalSet(
            np.arange(self._Nsub*self._Nsub), frame_shape, self._pixel_per_sub, bll)

        self._subaps.update_fix_threshold(self._centroid_threshold)

        self._sc = PCSlopeComputer(self._subaps)
        self._sc.set_frame(self._wf_ref)

    def shift_subaperture_grid(self, grid_shiftYX=[0, 0]):

        self._subaps.shiftSubap(self._subaps.keys(), grid_shiftYX)
        self.show_subaperture_grid()
        plt.title(
            f"Grid Shift: Y, X = [{grid_shiftYX[0]} , {grid_shiftYX[1]}]")
        
    def shift_subaperture_grid_around_central_subap(self, grid_shiftYX=[0, 0], yc = 861, xc = 959):
        '''
        yc and xc are the coordinates of the central subaperture wrt the footprint
        you got with a point source
        '''
        self._subaps.shiftSubap(self._subaps.keys(), grid_shiftYX)
        self.show_subaperture_grid_in_central_roi(yc, xc)
        plt.title(
            f"Grid Shift: Y, X = [{grid_shiftYX[0]} , {grid_shiftYX[1]}]")

    def _shift_subaperture_grid_to_null_tilt(self):
        offset_x = 42
        offset_y = 42

        while offset_x != 0 or offset_y != 0:
            offset_y = round(self._sc.slopes()[
                             :, 1].mean()/2*self._pixel_per_sub)
            offset_x = round(self._sc.slopes()[
                             :, 0].mean()/2*self._pixel_per_sub)
            self.shift_subaperture_grid([offset_y, offset_x])
            self._sc._reset_all_computed_attributes()

    def update_subapertures_threshold(self, threshold):

        for i in self._subaps.values():
            i.setFixThreshold(threshold)

    def remove_low_flux_subaperturers(self, threshold=None):

        self._sc.remove_low_flux_subaps(threshold)
    
    def show_subaperture_grid_in_central_roi(self, yc = 861, xc = 959):
        size = 26*3 + 4
        hsize = int(round(size*0.5))
        grid =  self._sc.subapertures_map()[yc-hsize:yc+hsize,xc-hsize:xc+hsize] * 1000
        frame = self._sc.frame()[yc-hsize:yc+hsize,xc-hsize:xc+hsize]
        self._show_map(grid+frame)
        
    def show_subaperture_grid(self):

        self._show_map(self._sc.subapertures_map()*1000+self._sc.frame())

    def show_reference_wf(self):

        plt.figure()
        plt.clf()
        plt.imshow(self._wf_ref)
        plt.colorbar(label='ADU')
        plt.title("Reference")

    def show_subaperture_flux_histogram(self):

        sub_flux = self._sc.subapertures_flux_map().flatten()
        bins = range(0, int(sub_flux.max()), int(sub_flux.max()*0.02))
        #bins = range(0, len(sub_flux), 10)#len(sub_flux)*0.02)
        plt.figure()
        plt.clf()
        plt.hist(sub_flux, bins, fc='k', ec='k')
        plt.xlabel('Counts per Subap [ADU]')
        plt.ylabel('N')

    def show_subaperture_flux_map(self):

        self._show_map(self._sc.subapertures_flux_map())

    def show_slopes_maps(self):

        vmin = np.array([self._sc.slopes_x_map().min(),
                        self._sc.slopes_y_map().min()]).min()
        vmax = np.array([self._sc.slopes_x_map().max(),
                        self._sc.slopes_y_map().max()]).max()

        plt.subplots(1, 2, sharex=True, sharey=True)

        plt.subplot(1, 2, 1)
        plt.title("Slope X")
        plt.imshow(self._sc.slopes_x_map(), vmin=vmin, vmax=vmax)

        plt.subplot(1, 2, 2)
        plt.title("Slope Y")
        plt.imshow(self._sc.slopes_y_map(), vmin=vmin, vmax=vmax)
        plt.colorbar(label='Slopes units')
    
    def display_flux_and_grid_maps(self):
        plt.subplots(1, 2, sharex=True, sharey=True)
        plt.subplot(1,2,1)
        plt.imshow(self._sc.subapertures_flux_map())
        
        plt.subplot(1,2,2)
        plt.imshow(self._sc.subapertures_map()*1000+self._sc.frame())
    
    @property
    def get_wf_reference(self):

        return self._wf_ref

    @property
    def get_number_of_subapertures(self):

        return self._Nsub

    @property
    def get_pixel_per_subapertures(self):

        return self._pixel_per_sub

    @property
    def get_subapertures(self):

        return self._subaps

    @property
    def get_slope_computer(self):

        return self._sc

    def _show_map(self, frame):

        plt.figure()
        plt.clf()
        plt.imshow(frame)
        plt.colorbar()

    @staticmethod
    def shift_subapertures_to_null_tilt(slope_computer):
        offset_x = 42
        offset_y = 42
        pixel_per_sub = next(iter(slope_computer.subapertures.values())).size()

        while offset_x != 0 or offset_y != 0:
            offset_y = round(slope_computer.slopes()[
                :, 1].mean()/2*pixel_per_sub)
            offset_x = round(slope_computer.slopes()[
                :, 0].mean()/2*pixel_per_sub)
            slope_computer.subapertures.shiftSubap(
                slope_computer.subapertures.keys(), [offset_y, offset_x])
            slope_computer._reset_all_computed_attributes()

    def remove_subaps_beyond_radius(self, central_subap_id, radius_inNsubap):
        """
        Removes the subapertures over a certain distance from a central subaperture
        """
        radius = radius_inNsubap
        subap_set = self._subaps
        subaperture_size = self.get_pixel_per_subapertures
        frame_shape = self._wf_ref.shape
        subIdMap = np.zeros(frame_shape, dtype=int)
        for subap in subap_set.values():
            subIdMap.flat[subap.pixelList()] = subap.ID()
        
        # computes the center of each subaperture
        centers = {}
        for subap in subap_set.values():
            pixels = subap.pixelList()
            rows, cols = np.unravel_index(pixels, subIdMap.shape)
            center_row = np.mean(rows)
            center_col = np.mean(cols)
            
            grid_row = int(center_row // subaperture_size)
            grid_col = int(center_col // subaperture_size)
            centers[subap.ID()] = (grid_row, grid_col)
    
        # computes the coordiates of the central subaperture
        if central_subap_id not in centers:
            raise ValueError("Central ID not found in subapertures")
        c_row, c_col = centers[central_subap_id]
    
        # Computes the distances and the list of subapertures to be remouved
        to_remove = []
        for subap_id, (r, c) in centers.items():
            dist = np.sqrt((r - c_row)**2 + (c - c_col)**2)  
            if dist > radius:
                to_remove.append(subap_id)
    
        # remouves the subapertures
        subap_set.removeSubap(to_remove)

    def display_simple_subapID_map(self):
        
        plt.figure()
        plt.clf()
        plt.imshow(self._sc.subapertures_id_map())

    def display_subapID_map(self):

        plt.figure()
        plt.clf()
    
        id_map = self._sc.subapertures_id_map()
    
        plt.imshow(id_map, cmap='nipy_spectral')
        #plt.colorbar(label='Subaperture ID')
        plt.title("Subaperture ID map")
    
        unique_ids = np.unique(id_map)
    
        for sub_id in unique_ids:
            if sub_id == 0:
                continue  
    
            coords = np.argwhere(id_map == sub_id)
            center_y, center_x = coords.mean(axis=0)

            plt.text(center_x, center_y, str(np.int16(sub_id)),
                     color='white', fontsize=5,
                     ha='center', va='center')#,
                    #weight='bold',
                    #bbox=dict(facecolor='black', alpha=0.5, lw=0))
    
        plt.axis('off')
        plt.show()
    
    # def display_subaperture_status(self):
    #
    #     plt.subplots(1, 3, sharex=True, sharey=True)
    #     plt.subplot(1,3,1)
    #     plt.title("Flux Map")
    #     plt.imshow(self._sc.subapertures_flux_map())
    #     plt.axis('off')
    #     plt.subplot(1,3,2)
    #     plt.title("SH Frame")
    #     plt.imshow(self._sc.subapertures_map()*1000+self._sc.frame())
    #     plt.axis('off')
    #     plt.subplot(1,3,3)
    #     #plt.imshow(self._sc.subapertures_id_map())
    #     id_map = self._sc.subapertures_id_map()
    #     plt.imshow(id_map, cmap='nipy_spectral')
    #     #plt.colorbar(label='Subaperture ID')
    #     plt.title("Subaperture ID map")
    #
    #     unique_ids = np.unique(id_map)
    #
    #     for sub_id in unique_ids:
    #         if sub_id == 0:
    #             continue  
    #
    #         coords = np.argwhere(id_map == sub_id)
    #         center_y, center_x = coords.mean(axis=0)
    #
    #         plt.text(center_x, center_y, str(np.int16(sub_id)),
    #                  color='white', fontsize=5,
    #                  ha='center', va='center')#,
    #                 #weight='bold',
    #                 #bbox=dict(facecolor='black', alpha=0.5, lw=0))
    #
    #     plt.axis('off')
        


    def display_subaperture_status(self):
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    
        # --- Plot 1: Flux Map ---
        flux_map = self._sc.subapertures_flux_map()
        im1 = axes[0].imshow(flux_map)
        axes[0].set_title("Flux Map")
        axes[0].axis('off')
        cbar1 = fig.colorbar(im1, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.08, label='ADU')
        cbar1.ax.tick_params(labelsize=6)
    
        # --- Plot 2: SH Frame ---
        sh_frame = self._sc.subapertures_map() * 1000 + self._sc.frame()
        im2 = axes[1].imshow(sh_frame)
        axes[1].set_title("SH Frame")
        axes[1].axis('off')
        cbar2 = fig.colorbar(im2, ax=axes[1], orientation='horizontal', fraction=0.046, pad=0.08, label='ADU')
        cbar2.ax.tick_params(labelsize=6)
    
        # --- Plot 3: Subaperture ID map ---
        id_map = self._sc.subapertures_id_map()
        im3 = axes[2].imshow(id_map, cmap='nipy_spectral')
        axes[2].set_title("Subaperture ID map")
    
        unique_ids = np.unique(id_map)
        for sub_id in unique_ids:
            if sub_id == 0:
                continue
            coords = np.argwhere(id_map == sub_id)
            center_y, center_x = coords.mean(axis=0)
            axes[2].text(center_x, center_y, str(np.int16(sub_id)),
                         color='white', fontsize=5,
                         ha='center', va='center')
    
        axes[2].axis('off')
    
        plt.tight_layout()
        plt.show()

    
    
    def interactive_subaperture_selection(self):
        '''
        allows the user to interact with the ID subap
        to add and remove subapertures. It is possible
        to add and remove only the actual subaperture set status
        the one before this function is called
        '''
        id_map = self._sc.subapertures_id_map()
        flux_map = self._sc.subapertures_flux_map()
        subap_map = self._sc.subapertures_map()
        frame = self._sc.frame()
    
        # Copy of the initial full subaperture set
        from copy import deepcopy
        full_subaps = deepcopy(self._subaps)
        selected_ids = set()
    
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
        ax_flux, ax_combined, ax_id = axes
    
        img_flux = ax_flux.imshow(flux_map)
        img_combined = ax_combined.imshow(subap_map * 1000 + frame)
        img_id = ax_id.imshow(id_map)
    
        ax_flux.set_title("Flux Map")
        ax_combined.set_title("SH Frame")
        ax_id.set_title("Subap ID Map (click to Add/Remove Subaps)")
    
        def onclick(event):
            if event.inaxes != ax_id:
                return
    
            x, y = int(event.xdata), int(event.ydata)
    
            # search of id_map from the initial full subapset to collect the correct ID 
            temp_sc = PCSlopeComputer(full_subaps)
            temp_sc.set_frame(self._wf_ref)
            full_id_map = temp_sc.subapertures_id_map()
            subap_id = full_id_map[y, x]
    
            if subap_id == 0:
                return
    
            # toggle selection
            if subap_id in selected_ids:
                selected_ids.remove(subap_id)
            else:
                selected_ids.add(subap_id)
    
            # search the filtered subset
            filtered_subaps = {
                sid: subap for sid, subap in full_subaps.items() if sid not in selected_ids
            }
    
            # # new slope computer
            # self._subaps = deepcopy(filtered_subaps)
            # self._sc = PCSlopeComputer(self._subaps)
            # self._sc.set_frame(self._wf_ref)
    
            # crea copia modificabile e rimuovi le subaperture escluse
            self._subaps = deepcopy(full_subaps)
            self._subaps.removeSubap(list(selected_ids))

            # aggiorna slope computer
            self._sc = PCSlopeComputer(self._subaps)
            self._sc.set_frame(self._wf_ref)
            
            # updates maps
            id_map_updated = self._sc.subapertures_id_map()
            flux_map_updated = self._sc.subapertures_flux_map()
            subap_map_updated = self._sc.subapertures_map()
    
            img_flux.set_data(flux_map_updated)
            img_combined.set_data(subap_map_updated * 1000 + self._sc.frame())
            img_id.set_data(id_map_updated)
    
            fig.canvas.draw_idle()
    
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    
    def set_curret_subapset_as_backup(self):
        from copy import deepcopy
        self._original_subaps = deepcopy(self._subaps)
    
    def reset_subapertures_to_backup(self):
        """
        Restore to a backup subaperture set
        """
        from copy import deepcopy
        if self._original_subaps is None:
            raise ValueError("Subaperture set backup is not defined.")
        
        self._subaps = deepcopy(self._original_subaps)
        self._sc = PCSlopeComputer(self._subaps)
        self._sc.set_frame(self._wf_ref)


# def main(wf_ref, corner_xy=(0, 0), nsubaps=50, flux_threshold=57000):
#     pixel_per_sub = 26
#     # nsubaps = 78  # 50
#     sgi = SubapertureGridInitialiser(
#         wf_ref, pixel_per_sub, nsubaps, centroid_threshold=70)
#
#     sgi.show_reference_wf()
#     # top left coords to be fair
#     ybll = corner_xy[1]
#     xbll = corner_xy[0]
#     sgi.define_subaperture_set(ybll, xbll)
#     sgi.show_subaperture_grid()
#
#     # sgi.shift_subaperture_grid_to_null_tilt()
#
#     sgi.show_subaperture_flux_histogram()
#
#     sgi.remove_low_flux_subaperturers(flux_threshold)
#     sgi.show_subaperture_flux_map()
#     sgi.show_subaperture_grid()
#     sgi.show_slopes_maps()
#     return sgi

    #
    # @staticmethod
    # def restore(subap_tag, wf_ref):
    #     from astropy.io import fits
    #     subap_file_name = subaperture_set_folder()/(subap_tag + '.fits')
    #     subap = ShSubapertureSet.restore(subap_file_name)
    #     sc = PCSlopeComputer(subap)
    #     sc.set_frame(wf_ref)