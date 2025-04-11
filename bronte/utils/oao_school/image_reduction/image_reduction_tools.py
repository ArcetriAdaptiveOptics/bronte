import numpy as np


class ImageReduction():
    
    def __init__(self):
        pass
    @staticmethod
    def get_masked_array_above_threshold(image, threshold):
        
        masked_image = np.ma.array(image.copy(), mask = False) 
        masked_image.mask[image>=threshold] = True
        masked_image.fill_value = 0 
        return masked_image 
        