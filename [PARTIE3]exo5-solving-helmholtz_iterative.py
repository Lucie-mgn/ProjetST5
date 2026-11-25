# Python packages
import matplotlib.pyplot as plt
import numpy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys

# MRG packages
import zsolutions4students as solutions 
# Fonction de callback pour collecter le résidu à chaque itération
# Nous stockons la norme du résidu à chaque itération dans une liste 'residuals_history'
# Le vecteur second membre B doit être accessible

def get_callback(residuals_history, A_sparse, B_vector):
    def callback_function(xk):
        residual_vector = B_vector - A_sparse.dot(xk)
        residuals_history.append(numpy.linalg.norm(residual_vector))
    return callback_function


def run_exercise_solution_helmholtz_iterative():

    # -- set equation parameters
    wavenumber = numpy.pi
    # -- set geometry parameters
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = 50, 50

   # -- generate mesh
    node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    # .. todo:: Modify the line below to define a different geometry.
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes)-1

    # -- set boundary geometry
    nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
    nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
    nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
    nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
    nodes_on_boundary = numpy.unique(numpy.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )

    # -- set exact solution
    solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    laplacian_of_solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    for i in range(nnodes):
        x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
        solexact[i] = numpy.exp(complex(0.,1.)*wavenumber*x)
        laplacian_of_solexact[i] = complex(0.,1.)*wavenumber*complex(0.,1.)*wavenumber * solexact[i]
    # ..warning: end

    # -- set dirichlet boundary conditions
    values_at_nodes_on_boundary = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
    values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

    # -- set finite element matrices and right hand side
    f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

    # ..warning: for teaching purpose only 
    for i in range(nnodes):
        # evaluate: (-\Delta - k^2) u(x,y) = ...
        f_unassembled[i] = - laplacian_of_solexact[i] - (wavenumber ** 2) * solexact[i]
        # ..warning: end

    coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
    coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
    K, M, F = solutions._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    A = K - wavenumber**2 * M
    B = F

    # -- apply Dirichlet boundary conditions
    A, B = solutions._set_dirichlet_condition(nodes_on_boundary, values_at_nodes_on_boundary, A, B)
# Convertir A en format CSR (Compressed Sparse Row) pour les solveurs itératifs
    A = scipy.sparse.csr_matrix(A) 
    B = B.reshape(nnodes)

    # --- Résolution avec les solveurs itératifs et traçage ---

    solvers = {
        'BiCGSTAB ': scipy.sparse.linalg.bicgstab,
        'BiCG': scipy.sparse.linalg.bicg,
    }
    
    results = {}
    x0 = numpy.zeros_like(B)
    tolerance = 1e-14

    for name, solver_func in solvers.items():
        print(f"Résolution avec {name}...")
        residuals_history = [] 
        callback_instance = get_callback(residuals_history, A, B)
        
        # Préparation des arguments pour le solveur
        solver_kwargs = {
            'x0': x0,
            'rtol': tolerance,
            'maxiter': 250,
            'callback': callback_instance,
        }
        

        # Exécution du solveur
        sol, info = solver_func(A, B, **solver_kwargs)
        

        initial_residual_norm = numpy.linalg.norm(B)
        residuals_history.insert(0, initial_residual_norm) 

        results[name] = {
            'sol': sol,
            'info': info,
            'residuals': numpy.array(residuals_history)
        }
        print(f"  Statut : {info} (0 = convergence). Nombre d'itérations : {len(residuals_history) - 1}")

    # --- Tracé de l'erreur ---
    
    plt.figure(figsize=(10, 6))
    
    # La norme de B est utilisée pour la normalisation
    initial_residual = numpy.linalg.norm(B)

    for name, data in results.items():
        # ... (le code de tracé est maintenant correct)
        normalized_residuals = data['residuals'] / initial_residual
        iterations = numpy.arange(len(normalized_residuals))
        
        if len(iterations) > 1:
            plt.plot(iterations, normalized_residuals, label=name)

    plt.yscale('log')
    plt.xlabel('Nombre d\'itérations', fontsize=12)
    plt.ylabel('Résidu ', fontsize=12)
    plt.title('Convergence des méthodes itératives pour Helmholtz', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.axhline(tolerance, color='r', linestyle=':', label=f'Tolérance ({tolerance})')
    plt.xlim(0)
    plt.show()

    print('End of iterative comparison.')
    return

if __name__ == '__main__':
    run_exercise_solution_helmholtz_iterative()
