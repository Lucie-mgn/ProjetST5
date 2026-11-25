# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import scipy
from scipy.optimize import minimize
import scipy.io


def real_to_complex(z):
    return z[0] + 1j * z[1]


def complex_to_real(z):
    return numpy.array([numpy.real(z), numpy.imag(z)])


class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        # .. todo: deepcopy here if returning objects
        return self.memo[args]


def compute_alpha(omega, material):
    """
    .. warning: $w = 2 \pi f$
    w is called circular frequency
    f is called frequency
    """

    materials_params = {
    'BETON': {'phi':0.005, 'gamma_p':1.4, 'sigma':1e6, 'rho_0':1.225, 'alpha_h':3.0, 'c_0':343.0},
    'BIRCHLT': {'phi':0.529, 'gamma_p':7.0/5.0, 'sigma':151429.0, 'rho_0':1.2, 'alpha_h':1.37, 'c_0':340.0},
    'FOAM' : {'phi':0.99, 'gamma_p':7.0/5.0, 'sigma':14000, 'rho_0':1.2, 'alpha_h':1.02, 'c_0':340.0}
    }

    if material not in materials_params:
        raise ValueError(f"Material '{material}' not defined!")

    # récupérer les paramètres pour le matériau choisi
    params = materials_params[material]
    phi = params['phi']
    gamma_p = params['gamma_p']
    sigma = params['sigma']
    rho_0 = params['rho_0']
    alpha_h = params['alpha_h']
    c_0 = params['c_0']



    # parameters of the geometry
    L = 0.01

    # parameters of the mesh
    resolution = 12  # := number of elements along L

    # parameters of the material (cont.)
    mu_0 = 1.0
    ksi_0 = 1.0 / (c_0 ** 2)
    mu_1 = phi / alpha_h
    ksi_1 = phi * gamma_p / (c_0 ** 2)
    a = sigma * (phi ** 2) * gamma_p / ((c_0 ** 2) * rho_0 * alpha_h)

    ksi_volume = phi * gamma_p / (c_0 ** 2)
    a_volume = sigma * (phi ** 2) * gamma_p / ((c_0 ** 2) * rho_0 * alpha_h)
    mu_volume = phi / alpha_h
    k2_volume = (1.0 / mu_volume) * ((omega ** 2) / (c_0 ** 2)) * (ksi_volume + 1j * a_volume / omega)
    print(k2_volume)

    # parameters of the objective function
    A = 1.0
    B = 1.0

    # defining k, omega and alpha dependant parameters' functions
    @Memoize
    def lambda_0(k, omega):
        if k ** 2 >= (omega ** 2) * ksi_0 / mu_0:
            return numpy.sqrt(k ** 2 - (omega ** 2) * ksi_0 / mu_0)
        else:
            return numpy.sqrt((omega ** 2) * ksi_0 / mu_0 - k ** 2) * 1j

    @Memoize
    def lambda_1(k, omega):
        temp1 = (omega ** 2) * ksi_1 / mu_1
        temp2 = numpy.sqrt((k ** 2 - temp1) ** 2 + (a * omega / mu_1) ** 2)
        real = (1.0 / numpy.sqrt(2.0)) * numpy.sqrt(k ** 2 - temp1 + temp2)
        im = (-1.0 / numpy.sqrt(2.0)) * numpy.sqrt(temp1 - k ** 2 + temp2)
        return complex(real, im)

    @Memoize
    def g(y):
        # ..warning: not validated ***********************
        return 1.0

    @Memoize
    def g_k(k):
        # ..warning: not validated ***********************
        if k == 0:
            return 1.0
        else:
            return 0.0

    @Memoize
    def f(x, k):
        return ((lambda_0(k, omega) * mu_0 - x) * numpy.exp(-lambda_0(k, omega) * L) \
                + (lambda_0(k, omega) * mu_0 + x) * numpy.exp(lambda_0(k, omega) * L))

    @Memoize
    def chi(k, alpha, omega):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 - lambda_1(k, omega) * mu_1) \
                          / f(lambda_1(k, omega) * mu_1, k) - (lambda_0(k, omega) * mu_0 - alpha) / f(alpha, k)))

    @Memoize
    def eta(k, alpha, omega):
        return (g_k(k) * ((lambda_0(k, omega) * mu_0 + lambda_1(k, omega) * mu_1) \
                          / f(lambda_1(k, omega) * mu_1, k) - (lambda_0(k, omega) * mu_0 + alpha) / f(alpha, k)))

    @Memoize
    def e_k(k, alpha, omega):
        expm = numpy.exp(-2.0 * lambda_0(k, omega) * L)
        expp = numpy.exp(+2.0 * lambda_0(k, omega) * L)

        if k ** 2 >= (omega ** 2) * ksi_0 / mu_0:
            return ((A + B * (numpy.abs(k) ** 2)) \
                    * ( \
                                (1.0 / (2.0 * lambda_0(k, omega))) \
                                * ((numpy.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm) \
                                   + (numpy.abs(eta(k, alpha, omega)) ** 2) * (expp - 1.0)) \
                                + 2 * L * numpy.real(chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega)))) \
                    + B * numpy.abs(lambda_0(k, omega)) / 2.0 * ((numpy.abs(chi(k, alpha, omega)) ** 2) * (1.0 - expm) \
                                                                 + (numpy.abs(eta(k, alpha, omega)) ** 2) * (
                                                                             expp - 1.0)) \
                    - 2 * B * (lambda_0(k, omega) ** 2) * L * numpy.real(
                        chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega))))
        else:
            return ((A + B * (numpy.abs(k) ** 2)) * (L \
                                                     * ((numpy.abs(chi(k, alpha, omega)) ** 2) + (
                                numpy.abs(eta(k, alpha, omega)) ** 2)) \
                                                     + complex(0.0, 1.0) * (1.0 / lambda_0(k, omega)) * numpy.imag(
                        chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega) \
                                                          * (1.0 - expm))))) + B * L * (
                               numpy.abs(lambda_0(k, omega)) ** 2) \
                   * ((numpy.abs(chi(k, alpha, omega)) ** 2) + (numpy.abs(eta(k, alpha, omega)) ** 2)) \
                   + complex(0.0, 1.0) * B * lambda_0(k, omega) * numpy.imag(
                chi(k, alpha, omega) * numpy.conj(eta(k, alpha, omega) \
                                                  * (1.0 - expm)))

    @Memoize
    def sum_e_k(omega):
        def sum_func(alpha):
            s = 0.0
            for n in range(-resolution, resolution + 1):
                k = n * numpy.pi / L
                s += e_k(k, alpha, omega)
            return s

        return sum_func

    @Memoize
    def alpha(omega):
        alpha_0 = numpy.array(complex(40.0, -40.0))
        temp = real_to_complex(minimize(lambda z: numpy.real(sum_e_k(omega)(real_to_complex(z))), complex_to_real(alpha_0), tol=1e-4).x)
        print(temp, "------", "je suis temp")
        return temp

    @Memoize
    def error(alpha, omega):
        temp = numpy.real(sum_e_k(omega)(alpha))
        return temp

    temp_alpha = alpha(omega)
    temp_error = error(temp_alpha, omega)

    return temp_alpha, temp_error


def run_compute_alpha(material):
    print('Computing alpha...')
    numb_omega = 10
    #omegas = numpy.logspace(numpy.log10(600), numpy.log10(30000), num=numb_omega)
    omegas = numpy.linspace(2.0 * numpy.pi, numpy.pi * 10000, num=numb_omega)
    temp = [compute_alpha(omega, material=material) for omega in omegas]
    print("temp:", "------", temp)
    alphas, errors = map(list, zip(*temp))
    alphas = numpy.array(alphas)
    errors = numpy.array(errors)

    print('Writing alpha...')
    output_filename = 'dta_omega_' + str(material) + '.mtx'
    scipy.io.mmwrite(output_filename, omegas.reshape(alphas.shape[0], 1), field='complex', symmetry='general')
    output_filename = 'dta_alpha_' + str(material) + '.mtx'
    scipy.io.mmwrite(output_filename, alphas.reshape(alphas.shape[0], 1), field='complex', symmetry='general')
    output_filename = 'dta_error_' + str(material) + '.mtx'
    scipy.io.mmwrite(output_filename, errors.reshape(errors.shape[0], 1), field='complex', symmetry='general')

    return


def load_alpha_data(material):
    """Charge les données omegas, alphas et errors depuis les fichiers .mtx."""
    print(f'Reading alpha data for {material}...')
    try:
        input_filename = 'dta_omega_' + str(material) + '.mtx'
        omegas = scipy.io.mmread(input_filename).reshape(-1)
        input_filename = 'dta_alpha_' + str(material) + '.mtx'
        alphas = scipy.io.mmread(input_filename).reshape(-1)
        input_filename = 'dta_error_' + str(material) + '.mtx'
        errors = scipy.io.mmread(input_filename).reshape(-1)
        return omegas, alphas, errors
    except FileNotFoundError:
        print(f"Erreur: Fichier de données pour {material} non trouvé. Exécutez run_compute_alpha d'abord.")
        return None, None, None


def run_plot_combined(materials_list):
    """Trace les parties réelle et imaginaire d'alpha pour plusieurs matériaux sur une seule figure."""

    data = {}
    colors = {'BETON': 'darkblue', 'BIRCHLT': 'darkred','FOAM' : 'darkgreen'}
    
    # Charger les données pour tous les matériaux
    for material in materials_list:
        omegas, alphas, errors = load_alpha_data(material)
        if omegas is not None:
            data[material] = {'omegas': omegas, 'alphas': alphas, 'errors': errors}
        else:
            print(f"Données de {material} non disponibles. Saut de ce matériau.")

    if not data:
        print("Aucune donnée chargée. Fin du tracé combiné.")
        return

    fig_real = matplotlib.pyplot.figure(figsize=(10, 5))
    matplotlib.pyplot.subplot(1, 1, 1)
    for material, d in data.items():
        matplotlib.pyplot.plot(numpy.real(d['omegas']), numpy.real(d['alphas']), 
                               color=colors.get(material, 'black'), 
                               label=f'Re($\\alpha$) - {material}')
    
    matplotlib.pyplot.xlabel(r'$\omega$ (Pulsation)', fontsize=12)
    matplotlib.pyplot.ylabel(r'$\operatorname{Re}(\alpha)$', fontsize=12)
    matplotlib.pyplot.title('Partie Réelle de $\\alpha$ en fonction de la Pulsation', fontsize=14)
    matplotlib.pyplot.ylim(0, 50) # Ajuster si nécessaire
    matplotlib.pyplot.legend()
    matplotlib.pyplot.grid(True, which='both', linestyle='--')
    matplotlib.pyplot.show()
    fig_real.savefig('fig_alpha_real_combined.jpg')
    matplotlib.pyplot.close(fig_real)

    # Tracé de Im(alpha) 
    fig_imag = matplotlib.pyplot.figure(figsize=(10, 5))
    matplotlib.pyplot.subplot(1, 1, 1)
    for material, d in data.items():
        matplotlib.pyplot.plot(numpy.real(d['omegas']), numpy.imag(d['alphas']), 
                               color=colors.get(material, 'black'), 
                               label=f'Im($\\alpha$) - {material}')
    
    matplotlib.pyplot.xlabel(r'$\omega$ (Pulsation)', fontsize=12)
    matplotlib.pyplot.ylabel(r'$\operatorname{Im}(\alpha)$', fontsize=12)
    matplotlib.pyplot.title('Partie Imaginaire de $\\alpha$ en fonction de la Pulsation', fontsize=14)
    matplotlib.pyplot.ylim(-150, 10) # Ajuster si nécessaire
    matplotlib.pyplot.legend()
    matplotlib.pyplot.grid(True, which='both', linestyle='--')
    matplotlib.pyplot.show()
    fig_imag.savefig('fig_alpha_imag_combined.jpg')
    matplotlib.pyplot.close(fig_imag)


def run():
    # Définir la liste des matériaux
    materials = ['BETON', 'BIRCHLT','FOAM']

    # Exécuter le calcul pour chaque matériau (génère les fichiers .mtx)
    for material in materials:
        run_compute_alpha(material)
        
    # Exécuter le tracé combiné
    run_plot_combined(materials)
    
    return


if __name__ == '__main__':
    run()
    print('End.')
