import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np

from specula.data_objects.intmat import Intmat
from bronte.package_data import reconstructor_folder
import matplotlib.pyplot as plt


def main():
    
    intmat_tag_list = _get_tag_list()
    Nmat = len(intmat_tag_list)
    intmat_list = []
    eigen_values_list = []
    
    for idx in range(Nmat):
        intmat = _load_intmat(intmat_tag_list[idx])
        intmat_list.append(intmat)   
        eigen_values_list.append(_get_eigenvalues(intmat))  
    
    plt.figure()
    plt.clf()
    plt.semilogy(eigen_values_list[0], label = '250218')
    for idx in range(1, Nmat):
        plt.semilogy(eigen_values_list[idx])
    
    plt.legend(loc='best')
    plt.grid('--', alpha = 0.3)
    plt.ylabel('Eigenvalues')
    

def _load_intmat(intmat_tag):
    file_name = reconstructor_folder() / (intmat_tag + '_bronte_im.fits')
    int_mat = Intmat.restore(file_name)
    return int_mat

def _get_eigenvalues(intmat):
    im_mat = intmat._intmat
    u,s,vh = np.linalg.svd(im_mat)
    return s

def _get_normalized(unnorm_data):
    norm_data = (unnorm_data - unnorm_data.min())/(
        unnorm_data.max() - unnorm_data.min())
    return norm_data

def _get_tag_list():
    
    tag_list = [
        '250218_125600',
        '250307_092400',
        '250307_093100',
        '250307_093600',
        '250307_094300',
        '250307_094900',
        '250307_095500',
        '250307_100000',
        '250307_100600',
        '250307_101100',
        '250307_101700',
        '250307_102300',
        '250307_140000',
        '250307_140600',
        '250307_150900'        
        ]
    return tag_list