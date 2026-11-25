# -*- coding: utf-8 -*-
"""
..warning:: The explanations of the functions in this file and the details of
the programming methodology have been given during the lectures.

Copyright (c) 2024 F. Magoules. All Rights Reserved
"""


# Python packages
import random
import matplotlib.pyplot as plt
import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy
import os
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys
from scipy.optimize import curve_fit


# MRG packages
import zsolutions4students as solutions


# ..todo: Uncomment for displaying limited digits
#numpy.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

############ Modification du maillage ############


def shift_mesh(node_coords,h):
    for i in range(len(node_coords)):
        dx=random.uniform(-(h/2-0.005),(h/2-0.005))
        dy=random.uniform(-(h/2-0.005),(h/2-0.005))
        x=node_coords[i][0]
        y=node_coords[i][1]
        if not(x==0. or x==1. or y==1. or y==0.):
            node_coords[i]=[node_coords[i][0]+dx,node_coords[i][1]+dy,0]
    return node_coords


############ Trouver alpha pour un maillage classique ############

#function to find alpha and C
def function(h, alpha, C):
    return C * h**alpha


def run_exercise_solution_helmholtz_dddd():

    # Define a list of mesh sizes h to evaluate
    h_values = [0.2, 0.16, 0.13, 0.10, 0.08, 0.065, 0.05, 0.04]  #Mesh sizes
    errors = []  

    for h in h_values :

        # -- set equation parameters
        wavenumber = numpy.pi
        # -- set geometry parameters
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
        nelemsx, nelemsy = int(1/h),int(1/h)

        # -- generate mesh
        nnodes = (nelemsx + 1) * (nelemsy + 1)
        nelems = nelemsx * nelemsy * 2
        node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
        # .. todo:: Modify the line below to define a different geometry.
        # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
        nnodes = node_coords.shape[0]
        nelems = len(p_elem2nodes)-1

        # -- plot mesh
        
        fig = matplotlib.pyplot.figure(1)
        ax = matplotlib.pyplot.subplot(1, 1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')
        matplotlib.pyplot.show()

        # -- set boundary geometry
        # boundary composed of nodes
        # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
        nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
        nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
        nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
        nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
        nodes_on_boundary = numpy.unique(numpy.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
        # ..warning: the ids of the nodes on the boundary should be 'global' number.
        # nodes_on_boundary = ...

        # ..warning: for teaching purpose only
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

        # -- solve linear system
        sol = scipy.linalg.solve(A, B)

        # -- plot finite element solution
        solreal = sol.reshape((sol.shape[0], ))
        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solreal))
        
        # 
        # ..warning: for teaching purpose only
        # -- plot exact solution
        solexactreal = solexact.reshape((solexact.shape[0], ))

        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solexactreal))
        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solexactreal))
        # # ..warning: end

        # ..warning: for teaching purpose only
        # -- plot exact solution - approximate solution
        solerr = solreal - solexactreal
        error = numpy.sqrt(h)* numpy.linalg.norm(solerr, 2) #attention, norme discrète
        errors.append(error)

        """_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solerr))
        _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solerr))"""
        # # ..warning: end

    # Perform curve fitting to find beta and C
    alphaC, _ = curve_fit(function, h_values, errors)
    alpha, C = alphaC


    plt.loglog(h_values, errors, 'o-', label='Erreurs')
    plt.loglog(h_values, function(numpy.array(h_values), alpha, C), '--', label= "Courbe approximée")
    plt.xlabel('h')
    plt.ylabel('Erreurs')
    plt.title('Erreur en fonction de h (échelle log)')
    plt.text(0.10, 0.006, f'On obtient : alpha = {alpha:.2f}', fontsize=12, color='red')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


#run_exercise_solution_helmholtz_dddd()


############ Trouver beta ############


def run_exercise_solution_helmholtz_dddd_beta():

    # Define a list of mesh sizes h to evaluate
    errors = []  
    k_values = [numpy.pi, 2 * numpy.pi, 3 * numpy.pi, 4 * numpy.pi]


    for k in k_values :

        # -- set equation parameters
        wavenumber = k
        # -- set geometry parameters
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
        nelemsx, nelemsy = 20, 20

        # -- generate mesh
        nnodes = (nelemsx + 1) * (nelemsy + 1)
        nelems = nelemsx * nelemsy * 2
        node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
        # .. todo:: Modify the line below to define a different geometry.
        # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
        nnodes = node_coords.shape[0]
        nelems = len(p_elem2nodes)-1

        # -- plot mesh
        
        fig = matplotlib.pyplot.figure(1)
        ax = matplotlib.pyplot.subplot(1, 1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')
        matplotlib.pyplot.show()

        # -- set boundary geometry
        # boundary composed of nodes
        # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
        nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
        nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
        nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
        nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
        nodes_on_boundary = numpy.unique(numpy.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
        # ..warning: the ids of the nodes on the boundary should be 'global' number.
        # nodes_on_boundary = ...

        # ..warning: for teaching purpose only
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

        # -- solve linear system
        sol = scipy.linalg.solve(A, B)

        # -- plot finite element solution
        solreal = sol.reshape((sol.shape[0], ))
        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solreal))
        
        # 
        # ..warning: for teaching purpose only
        # -- plot exact solution
        solexactreal = solexact.reshape((solexact.shape[0], ))

        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solexactreal))
        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solexactreal))
        # # ..warning: end

        # ..warning: for teaching purpose only
        # -- plot exact solution - approximate solution
        solerr = solreal - solexactreal
        error = numpy.sqrt(k)* numpy.linalg.norm(solerr, 2) #attention, norme discrète
        errors.append(error)

        """_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solerr))
        _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solerr))"""
        # # ..warning: end

    # Perform curve fitting to find beta and C
    betaC, _ = curve_fit(function, k_values, errors)
    beta, C = betaC

    plt.loglog(k_values, errors, 'o-', label='Erreurs')
    plt.loglog(k_values, function(numpy.array(k_values), beta, C), '--', label= "Courbe approximée")
    plt.xlabel('k')
    plt.ylabel('Erreurs')
    plt.title('Erreur en fonction de k (échelle log)')
    plt.text(6.0, 0.15, f'On obtient : beta = {beta:.2f}', fontsize=12, color='red')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


#run_exercise_solution_helmholtz_dddd_beta()


############ Erreur en fonction de kh    ############

#function to find alpha and C
def function(hk, alpha, beta, C):
    h = hk[:,0]
    k = hk[:,1]
    return C * h**alpha * k**beta

def run_exercise_solution_helmholtz_dddd_kh():

    # Define a list of mesh sizes h to evaluate
    errors = []  
    k_values = [numpy.pi, 2 * numpy.pi, 3 * numpy.pi, 4 * numpy.pi]
    h_values = [0.2, 0.16, 0.13, 0.10, 0.08, 0.065, 0.05, 0.04]
    kh=[]

    for k in k_values :
        for h in h_values:
            kh.append(k*h)
            # -- set equation parameters
            wavenumber = k
            # -- set geometry parameters
            xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
            nelemsx, nelemsy = int(1/h),int(1/h)

            # -- generate mesh
            nnodes = (nelemsx + 1) * (nelemsy + 1)
            nelems = nelemsx * nelemsy * 2
            node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
            # .. todo:: Modify the line below to define a different geometry.
            # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
            nnodes = node_coords.shape[0]
            nelems = len(p_elem2nodes)-1

            # -- plot mesh
            """
            fig = matplotlib.pyplot.figure(1)
            ax = matplotlib.pyplot.subplot(1, 1, 1)
            ax.set_aspect('equal')
            ax.axis('off')
            solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')
            matplotlib.pyplot.show()"""
            print("ok")
            # -- set boundary geometry
            # boundary composed of nodes
            # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
            nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
            nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
            nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
            nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
            nodes_on_boundary = numpy.unique(numpy.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
            # ..warning: the ids of the nodes on the boundary should be 'global' number.
            # nodes_on_boundary = ...

            # ..warning: for teaching purpose only
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

            # -- solve linear system
            sol = scipy.linalg.solve(A, B)

            # -- plot finite element solution
            solreal = sol.reshape((sol.shape[0], ))
            #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
            #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solreal))
            
            # 
            # ..warning: for teaching purpose only
            # -- plot exact solution
            solexactreal = solexact.reshape((solexact.shape[0], ))

            #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solexactreal))
            #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solexactreal))
            # # ..warning: end

            # ..warning: for teaching purpose only
            # -- plot exact solution - approximate solution
            solerr = solreal - solexactreal
            error = numpy.sqrt(k*h)* numpy.linalg.norm(solerr, 2) #attention, norme discrète
            errors.append(error)

            """_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solerr))
            _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solerr))"""
            # # ..warning: end

    # Perform curve fitting to find beta and C
        
    H, K = numpy.meshgrid(h_values, k_values) 
    h_flat = H.flatten()
    k_flat = K.flatten()
    hk = numpy.column_stack((h_flat, k_flat))
    hk_flat = numpy.empty(h_flat.size * 2)
    hk_flat[0::2] = h_flat
    hk_flat[1::2] = k_flat
    errors = numpy.array(errors)
    errors_flat = errors.flatten()
    params, covariance = curve_fit(function, hk, errors_flat, p0=[1,1,1])
    alpha, beta, C = params
    print("alpha =", alpha)
    print("beta =", beta)
    print("C =", C)


    plt.loglog(h_flat * k_flat, errors, 'o-', label='Erreurs')
    plt.loglog(h_flat * k_flat, function(numpy.column_stack((h_flat, k_flat)), alpha, beta, C), '--', label="Courbe ajustée")
    plt.xlabel('h * k')
    plt.ylabel('Erreur')
    plt.title('Erreur en fonction de h et k (échelle log)')
    plt.text(0.26, 0.037, f'On obtient : alpha = {alpha:.2f}, beta = {beta:.2f}', fontsize=12, color='red')
    plt.legend()
    plt.grid(True)
    plt.show()

#run_exercise_solution_helmholtz_dddd_kh()


############ Fonction qui donne la moyenne de h ############

def h_average(node_coords, elem2nodes,p_elem2nodes):
    total_length = 0.0
    num_edges = 0

    # Loop over each element and compute the distance between nodes
    for element_index in range(len(p_elem2nodes) - 1):
        # Get the nodes of the current element using p_elem2nodes and elem2nodes
        element_nodes = elem2nodes[p_elem2nodes[element_index]:p_elem2nodes[element_index + 1]]

        for i in range(len(element_nodes)):
            for j in range(i + 1, len(element_nodes)):
                # Compute the distance between two nodes in the element
                node_i = node_coords[element_nodes[i], :]
                node_j = node_coords[element_nodes[j], :]
                edge_length = numpy.linalg.norm(node_i - node_j)
                total_length += edge_length
                num_edges += 1

    # Return the average length
    return total_length / num_edges

############ Recherche de alpha avec maillage modifié ############

def run_exercise_solution_helmholtz_dddd_alpha_shift():

    # Define a list of mesh sizes h to evaluate
    h_values = [0.2, 0.16, 0.13, 0.10, 0.08, 0.065, 0.05, 0.04]  #Mesh sizes
    errors = []  
    h_bis=[]    

    for h in h_values :

        # -- set equation parameters
        wavenumber = numpy.pi
        # -- set geometry parameters
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
        nelemsx, nelemsy = int(1/h),int(1/h)
    

        # -- generate mesh
        nnodes = (nelemsx + 1) * (nelemsy + 1)
        nelems = nelemsx * nelemsy * 2
        node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
        node_coords=shift_mesh(node_coords,h)
        h=h_average(node_coords,elem2nodes,p_elem2nodes)
        h_bis.append(h)
        # .. todo:: Modify the line below to define a different geometry.
        # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
        nnodes = node_coords.shape[0]
        nelems = len(p_elem2nodes)-1

        # -- plot mesh
        """
        fig = matplotlib.pyplot.figure(1)
        ax = matplotlib.pyplot.subplot(1, 1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')
        matplotlib.pyplot.show()"""

        # -- set boundary geometry
        # boundary composed of nodes
        # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
        nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
        nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
        nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
        nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
        nodes_on_boundary = numpy.unique(numpy.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
        # ..warning: the ids of the nodes on the boundary should be 'global' number.
        # nodes_on_boundary = ...

        # ..warning: for teaching purpose only
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

        # -- solve linear system
        sol = scipy.linalg.solve(A, B)

        # -- plot finite element solution
        solreal = sol.reshape((sol.shape[0], ))
        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solreal))
        
        # 
        # ..warning: for teaching purpose only
        # -- plot exact solution
        solexactreal = solexact.reshape((solexact.shape[0], ))

        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solexactreal))
        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solexactreal))
        # # ..warning: end

        # ..warning: for teaching purpose only
        # -- plot exact solution - approximate solution
        solerr = solreal - solexactreal
        error = numpy.sqrt(h)* numpy.linalg.norm(solerr, 2) #attention, norme discrète
        errors.append(error)

        """_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solerr))
        _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solerr))"""
        # # ..warning: end

    # Perform curve fitting to find beta and C
    alphaC, _ = curve_fit(function, h_bis, errors)
    alpha, C = alphaC

    plt.loglog(h_bis, errors, 'o-', label='Erreurs')
    plt.loglog(h_bis, function(numpy.array(h_bis), alpha, C), '--', label= "Courbe approximée")
    plt.xlabel('h')
    plt.ylabel('Erreurs')
    plt.title('Erreur en fonction de h (échelle log)')
    plt.text(0.11, 0.012, f'On obtient : alpha = {alpha:.2f}', fontsize=12, color='red')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


#run_exercise_solution_helmholtz_dddd_alpha_shift()

############ Recherche de beta avec maillage modifié ############


def run_exercise_solution_helmholtz_dddd_beta_shift():

    # Define a list of mesh sizes h to evaluate
    errors = []  
    k_values = [numpy.pi, 2 * numpy.pi, 3 * numpy.pi, 4 * numpy.pi]
    
    # -- set geometry parameters
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    nelemsx, nelemsy = 20, 20

    # -- generate mesh
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    nelems = nelemsx * nelemsy * 2
    node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    # .. todo:: Modify the line below to define a different geometry.
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes)-1

    # -- plot mesh
    """
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')
    matplotlib.pyplot.show()"""

    # -- set boundary geometry
    # boundary composed of nodes
    # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
    nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
    nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
    nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
    nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
    nodes_on_boundary = numpy.unique(numpy.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
    # ..warning: the ids of the nodes on the boundary should be 'global' number.
    # nodes_on_boundary = ...

    h = h_average(node_coords,elem2nodes,p_elem2nodes)

    for k in k_values :

        wavenumber = k

        # ..warning: for teaching purpose only
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
            f_unassembled[i] = - laplacian_of_solexact[i] - (k ** 2) * solexact[i]
        # ..warning: end

        coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
        coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
        K, M, F = solutions._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
        A = K - k**2 * M
        B = F

        # -- apply Dirichlet boundary conditions
        A, B = solutions._set_dirichlet_condition(nodes_on_boundary, values_at_nodes_on_boundary, A, B)

        # -- solve linear system
        sol = scipy.linalg.solve(A, B)

        # -- plot finite element solution
        solreal = sol.reshape((sol.shape[0], ))
        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solreal))
        
        # 
        # ..warning: for teaching purpose only
        # -- plot exact solution
        solexactreal = solexact.reshape((solexact.shape[0], ))

        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solexactreal))
        #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solexactreal))
        # # ..warning: end

        # ..warning: for teaching purpose only
        # -- plot exact solution - approximate solution
        solerr = solreal - solexactreal
        error = numpy.sqrt(k)* numpy.linalg.norm(solerr, 2) #attention, norme discrète
        errors.append(error)

        """_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solerr))
        _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solerr))"""
        # # ..warning: end

    # Perform curve fitting to find beta and C
    betaC, _ = curve_fit(function, k_values, errors)
    beta, C = betaC

    plt.loglog(k_values, errors, 'o-', label='Erreurs')
    plt.loglog(k_values, function(numpy.array(k_values), beta, C), '--', label= "Courbe approximée")
    plt.xlabel('k')
    plt.ylabel('Erreurs')
    plt.title(f"Erreur en fonction de k (échelle log) pour h = {h:.2f}")
    plt.text(6.0, 0.15, f'On obtient : beta = {beta:.2f}', fontsize=12, color='red')
    plt.legend()
    plt.grid(True)
    plt.show()
    return

#run_exercise_solution_helmholtz_dddd_beta_shift()


#function to find alpha and C
def function(hk, alpha, beta, C):
    h = hk[:,0]
    k = hk[:,1]
    return C * h**alpha * k**beta

def run_exercise_solution_helmholtz_dddd_kh_shift():

    # Define a list of mesh sizes h to evaluate
    errors = []  
    k_values = [numpy.pi, 2 * numpy.pi, 3 * numpy.pi, 4 * numpy.pi]
    h_values_first = [0.2, 0.16, 0.13, 0.10, 0.08, 0.065, 0.05, 0.04]
    h_values=[]
    kh=[]

    for h in h_values_first :
        # -- set equation parameters
        # -- set geometry parameters
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
        nelemsx, nelemsy = int(1/h),int(1/h)

        # -- generate mesh
        nnodes = (nelemsx + 1) * (nelemsy + 1)
        nelems = nelemsx * nelemsy * 2
        node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
        node_coords=shift_mesh(node_coords,h)
        h=h_average(node_coords,elem2nodes,p_elem2nodes)
        h_values.append(h)
        # .. todo:: Modify the line below to define a different geometry.
        # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
        nnodes = node_coords.shape[0]
        nelems = len(p_elem2nodes)-1
        for k in k_values:
            wavenumber = k
            

            # -- plot mesh
            """
            fig = matplotlib.pyplot.figure(1)
            ax = matplotlib.pyplot.subplot(1, 1, 1)
            ax.set_aspect('equal')
            ax.axis('off')
            solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')
            matplotlib.pyplot.show()"""
            print("ok")
            # -- set boundary geometry
            # boundary composed of nodes
            # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
            nodes_on_north = solutions._set_square_nodes_boundary_north(node_coords)
            nodes_on_south = solutions._set_square_nodes_boundary_south(node_coords)
            nodes_on_east = solutions._set_square_nodes_boundary_east(node_coords)
            nodes_on_west = solutions._set_square_nodes_boundary_west(node_coords)
            nodes_on_boundary = numpy.unique(numpy.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
            # ..warning: the ids of the nodes on the boundary should be 'global' number.
            # nodes_on_boundary = ...

            # ..warning: for teaching purpose only
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

            # -- solve linear system
            sol = scipy.linalg.solve(A, B)

            # -- plot finite element solution
            solreal = sol.reshape((sol.shape[0], ))
            #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
            #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solreal))
            
            # 
            # ..warning: for teaching purpose only
            # -- plot exact solution
            solexactreal = solexact.reshape((solexact.shape[0], ))

            #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solexactreal))
            #_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solexactreal))
            # # ..warning: end

            # ..warning: for teaching purpose only
            # -- plot exact solution - approximate solution
            solerr = solreal - solexactreal
            error = numpy.sqrt(k*h)* numpy.linalg.norm(solerr, 2) #attention, norme discrète
            errors.append(error)

            """_ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solerr))
            _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.imag(solerr))"""
            # # ..warning: end

    # Perform curve fitting to find beta and C
        
    H, K = numpy.meshgrid(h_values, k_values) 
    h_flat = H.flatten()
    k_flat = K.flatten()
    hk = numpy.column_stack((h_flat, k_flat))
    hk_flat = numpy.empty(h_flat.size * 2)
    hk_flat[0::2] = h_flat
    hk_flat[1::2] = k_flat
    errors = numpy.array(errors)
    errors_flat = errors.flatten()
    params, covariance = curve_fit(function, hk, errors_flat, p0=[1,1,1])
    alpha, beta, C = params
    print("alpha =", alpha)
    print("beta =", beta)
    print("C =", C)


    plt.loglog(h_flat * k_flat, errors, 'o-', label='Erreurs')
    plt.loglog(h_flat * k_flat, function(numpy.column_stack((h_flat, k_flat)), alpha, beta, C), '--', label="Courbe ajustée")
    plt.xlabel('h * k')
    plt.ylabel('Erreur')
    plt.title('Erreur en fonction de h et k (échelle log)')
    plt.text(0.26, 0.037, f'On obtient : alpha = {alpha:.2f}, beta = {beta:.2f}', fontsize=12, color='red')
    plt.legend()
    plt.grid(True)
    plt.show()

run_exercise_solution_helmholtz_dddd_kh_shift()

