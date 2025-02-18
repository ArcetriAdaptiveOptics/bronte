import specula
specula.init(-1, precision=1)  # Default target=-1 (CPU), float32=1
from specula import np

from specula.data_objects.intmat import Intmat
from bronte.package_data import reconstructor_folder
import matplotlib.pyplot as plt


def load_intmat(intmat_tag):
    file_name = reconstructor_folder() / (intmat_tag + '_bronte_im.fits')
    int_mat = Intmat.restore(file_name)
    return int_mat

def show_eigenmodes(intmat_tag = '250211_143700', do_plot=True):
    
    #file_name = reconstructor_folder() / (intmat_tag + '_bronte_im.fits')
    int_mat = load_intmat(intmat_tag)
    im_mat = int_mat._intmat
    u,s,vh = np.linalg.svd(im_mat)
    if do_plot is True:
        _do_plot(s, intmat_tag)
    return s
    
def _do_plot(eigen_values, title):
    
    Nmodes = len(eigen_values)
    j_vect = np.arange(2, Nmodes+2)
    plt.figure()
    plt.clf()
    plt.semilogy(j_vect, eigen_values, '.-', label=r'$\lambda_i$')
    plt.xlabel('Noll mode index')
    plt.ylabel('Eigenvalues')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    plt.title(title)

def main_plot_all():
    
    intmat_tag_200modespp8 = '250211_154500'
    intmat_tag_200modespp3 = '250211_155500'
    intmat_tag_200modespp1 = '250211_160100'
    intmat_tag_10modespp8 = '250211_143700'
    intmat_tag_2modespp8 = '250211_140400'
    intmat_tag_200modespp5 = '250218_085300'
    intmat_tag_200modespp2 = '250218_125600'
    
    s200pp8 = show_eigenmodes(intmat_tag_200modespp8, do_plot=False)
    s200pp3 = show_eigenmodes(intmat_tag_200modespp3, do_plot=False)
    s200pp1 = show_eigenmodes(intmat_tag_200modespp1, do_plot=False)
    s200pp5 = show_eigenmodes(intmat_tag_200modespp5, do_plot=False)
    s200pp2 = show_eigenmodes(intmat_tag_200modespp2, do_plot=False)
    show_eigenmodes(intmat_tag_10modespp8, do_plot=False)
    show_eigenmodes(intmat_tag_2modespp8, do_plot=False)
    
    Nmodes = len(s200pp8)
    j_vect = np.arange(2, Nmodes+2)
    plt.figure()
    plt.clf()
    plt.semilogy(j_vect, s200pp8, '.-', label=intmat_tag_200modespp8+' (pp=8um rms/n^2)')
    plt.semilogy(j_vect, s200pp3, '.-', label=intmat_tag_200modespp3+' (pp=3um rms/n^2)')
    plt.semilogy(j_vect, s200pp1, '.-', label=intmat_tag_200modespp1+' (pp=1um rms/n^2)')
    plt.semilogy(j_vect, s200pp5, '.-', label=intmat_tag_200modespp1+' (pp=5um rms/n^2)')
    plt.semilogy(j_vect, s200pp5, '.-', label=intmat_tag_200modespp1+' (pp=2um rms/n)')
    plt.xlabel('index')
    plt.ylabel('Eigenvalues '+'$\lambda_i$')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    plt.title('Interaction Matrix SVD')
    
    plt.figure()
    plt.clf()
    plt.semilogy(j_vect, (s200pp8-s200pp8.min())/(s200pp8.max()-s200pp8.min()), '.-', label=intmat_tag_200modespp8+' (pp=8um rms/n^2)')
    plt.semilogy(j_vect, (s200pp3-s200pp3.min())/(s200pp3.max()-s200pp3.min()), '.-', label=intmat_tag_200modespp3+' (pp=3um rms/n^2)')
    plt.semilogy(j_vect, (s200pp1-s200pp1.min())/(s200pp1.max()-s200pp1.min()), '.-', label=intmat_tag_200modespp1+' (pp=1um rms/n^2)')
    plt.semilogy(j_vect, (s200pp5-s200pp5.min())/(s200pp5.max()-s200pp5.min()), '.-', label=intmat_tag_200modespp5+' (pp=5um rms/n^2)')
    plt.semilogy(j_vect, (s200pp2-s200pp2.min())/(s200pp2.max()-s200pp2.min()), '.-', label=intmat_tag_200modespp5+' (pp=2um rms/n)')
    plt.xlabel('index')
    plt.ylabel('Normalized Eigenvalues\t'+r'$\frac{\lambda_i-\lambda_{min}}{\lambda_{max}-\lambda_{min}}$')
    plt.title('Interaction Matrix SVD')
    plt.grid('--', alpha=0.3)
    plt.legend(loc='best')
    