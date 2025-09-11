import numpy as np
import matplotlib.pyplot as plt


# def main():
#
#     q = np.arange(-5,6)
#
#     eta_2 = eta(q, 2*2*np.pi)
#     eta_175 = eta(q, 1.75*2*np.pi)
#     eta_1 = eta(q, 2*np.pi)
#     eta_075 = eta(q, 0.75*2*np.pi)
#     eta_050 =  eta(q, 0.5*2*np.pi)
#
#     plt.figure()
#     plt.clf()
#     plt.bar(q, eta_2, label = r'$\phi_w = 4\pi$')
#     plt.bar(q, eta_175, label = r'$\phi_w = 3.5\pi$')
#     plt.bar(q, eta_1, label = r'$\phi_w = 2\pi$')
#     plt.bar(q, eta_075, label = r'$\phi_w=1.5\pi$')
#     plt.bar(q, eta_050, label = r'$\phi_w=\pi$')
#     plt.ylabel(r'$Diffraction efficiency \ \eta_q \ Normalized$')
#     plt.xlabel(r'$Diffraction \t order \ q$')
#     plt.grid('--', alpha=0.3)
#     plt.legend(loc='best')
#
#
#     phi_vector = np.linspace(0, 2*np.pi, 1000)
#     eta_q0 = eta(0, phi_vector)
#     eta_q1 = eta(1, phi_vector)
#
#     plt.figure()
#     plt.clf()
#     plt.plot(phi_vector*0.5/np.pi, eta_q0, label =r'$\eta_0$')
#     plt.plot(phi_vector*0.5/np.pi, eta_q1, label =r'$\eta_1$')
#     plt.xlabel(r'$\phi_w/(2\pi)$')
#     plt.ylabel(r'$\eta_q$')
#     plt.legend(loc='best')
#     plt.grid('--', alpha=0.3)
#
# def eta(q, phi_w):
#
#     return np.sinc(q - 0.5*phi_w/np.pi)**2

import numpy as np
import matplotlib.pyplot as plt

def eta(q, phi_w):
    return np.sinc(q - 0.5 * phi_w / np.pi) ** 2

def main():
    
    ax_label_fontsize=12
    
    q = np.arange(-5, 6)  # ordini diffrattivi

    # calcolo le efficienze
    curves = {
        r'$\phi_w = 4\pi$':   eta(q, 2*2*np.pi),
        r'$\phi_w = 3.5\pi$': eta(q, 1.75*2*np.pi),
        r'$\phi_w = 2\pi$':   eta(q, 2*np.pi),
        r'$\phi_w = 1.5\pi$': eta(q, 0.75*2*np.pi),
        r'$\phi_w = \pi$':    eta(q, np.pi)
    }

    # --- Primo grafico: grouped bar chart ---
    plt.figure(figsize=(8,5))
    width = 0.15  # larghezza barre
    offsets = np.linspace(- (len(curves)-1)/2, (len(curves)-1)/2, len(curves)) * width

    for (label, values), dx in zip(curves.items(), offsets):
        plt.bar(q + dx, values, width=width, label=label, alpha=0.8, edgecolor='k')

    plt.ylabel(r"$Diffraction \ efficiency \ \eta_q \ = \ I(x_q)/I_{tot}$", fontsize=ax_label_fontsize)
    plt.xlabel(r"$Diffraction \ order \ q$", fontsize=ax_label_fontsize)
    plt.xticks(q)  # mostro esattamente le tick di q
    plt.yticks(np.linspace(0,1,11))
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='best')
    plt.tight_layout()

    # --- Secondo grafico: efficienza vs phase depth ---
    phi_vector = np.linspace(0, 2*np.pi, 1000)
    eta_q0 = eta(0, phi_vector)
    eta_q1 = eta(1, phi_vector)
    eta_q2 = eta(2, phi_vector)
    eta_q3 = eta(-1, phi_vector)
    eta_q4 = eta(-2, phi_vector)

    plt.figure(figsize=(7,5))
    plt.plot(phi_vector/(2*np.pi), eta_q0, label=r'$\eta_0$', lw=2)
    plt.plot(phi_vector/(2*np.pi), eta_q1, label=r'$\eta_1$', lw=2)
    plt.plot(phi_vector/(2*np.pi), eta_q2, label=r'$\eta_{2}$', lw=2)
    plt.plot(phi_vector/(2*np.pi), eta_q3, label=r'$\eta_{-1}$', lw=2)
    plt.plot(phi_vector/(2*np.pi), eta_q4, label=r'$\eta_{-2}$', lw=2)
    plt.xlabel(r'$\phi_w / (2\pi)$', fontsize=ax_label_fontsize)
    plt.yticks(np.linspace(0,1,11))
    plt.xticks(np.linspace(0,1,11))
    plt.ylabel(r'$Diffraction \ efficiency \ \eta_q \ = \ I(x_q)/I_{tot}$', fontsize=ax_label_fontsize)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    plt.show()

# if __name__ == "__main__":
#     main()
