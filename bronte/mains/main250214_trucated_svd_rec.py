import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np

from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from bronte.package_data import reconstructor_folder
#import matplotlib.pyplot as plt

def load_intmat(intmat_tag):
    file_name = reconstructor_folder() / (intmat_tag + '_bronte_im.fits')
    int_mat = Intmat.restore(file_name)
    return int_mat

def main(ftag=None):
    '''
    Computes the reconstructor, using a T-SVD.
    
    parameter:
    ftag(string): tag file of the reconstructor to be saved. 
    If None the reconstructor is not saved
    '''
    intmat_tag = '250211_154500'
    specula_int_mat = load_intmat(intmat_tag)
    
    int_mat = specula_int_mat._intmat
    cond_fact = 0.05
    rec_tsvd = np.linalg.pinv(int_mat, rcond=cond_fact)
    rec = Recmat(rec_tsvd)
    rec.im_tag = specula_int_mat._norm_factor
    rec_tag = ftag + '_bronte_rec.fits'
    file_name = reconstructor_folder() / rec_tag
    rec.save(str(file_name))
    
    