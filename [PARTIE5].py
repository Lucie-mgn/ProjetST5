import numpy
import math
import matplotlib.pyplot as plt
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import random
import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import solutions
import zsolutions4students

############ création d'une fractale grâce au travail intermédiaire ############

def quadtotri(node_coords,p_elem2nodes, elem2nodes):
    ntriangles = int((p_elem2nodes[len(p_elem2nodes)-1]/4)*2)
    n_p_e2n = [] 
    n_e2n = [] #nouveau maillage
    for i in range(ntriangles+1):
        n_p_e2n.append(3*i) #ajout des nouveaux élements
    for j in range(len(p_elem2nodes)-1): #division en deux triangles
        n_e2n.append(elem2nodes[4*j])
        n_e2n.append(elem2nodes[4*j+1])
        n_e2n.append(elem2nodes[4*j+2])
        n_e2n.append(elem2nodes[4*j])
        n_e2n.append(elem2nodes[4*j+2])
        n_e2n.append(elem2nodes[4*j+3])
    return(n_p_e2n,n_e2n)


def compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes):
    dimension = len(node_coords[0])
    nelem = len(p_elem2nodes)-1
    barycentre= np.zeros((nelem,dimension)) #tableau de stockage des barycentres
    for i in range(nelem):
        x, y, z = 0,0,0
        for j in range(p_elem2nodes[i],p_elem2nodes[i+1]):
            x+=node_coords[elem2nodes[j]][0]/(p_elem2nodes[i+1]-p_elem2nodes[i])
            y+=node_coords[elem2nodes[j]][1]/(p_elem2nodes[i+1]-p_elem2nodes[i])
            z+=node_coords[elem2nodes[j]][2]/(p_elem2nodes[i+1]-p_elem2nodes[i]) #on fait la somme des coordonnées divisé par le nombre de noeuds
            barycentre[i][0]=x
            barycentre[i][1]=y
            barycentre[i][2]=z
    return barycentre

def add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid_coords):
    # ajoute un point, sans le lier au maillage
    node_coords=numpy.vstack([node_coords,nodeid_coords])
    return node_coords, p_elem2nodes, elem2nodes



def add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid2nodes,new_elem_nodes):
    #les trois derniers points ajouter forment un élement
    n=len(node_coords)
    elem2nodes=numpy.append(elem2nodes,new_elem_nodes)
    p_elem2nodes=numpy.append(p_elem2nodes,p_elem2nodes[-1]+len(new_elem_nodes))
    return node_coords, p_elem2nodes, elem2nodes


def remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid): #nodeid c'est la position de l'element

    new_elem2nodes = []
    new_p_elem2nodes = [0]

    for e in range(len(p_elem2nodes) - 1): # on parcours tout les élements
        start, end = p_elem2nodes[e], p_elem2nodes[e+1]
        nodes_elem = elem2nodes[start:end].tolist() # liste de 1 élement


        if nodeid in nodes_elem : #on vérifie si nodeid est dedans
            nodes_elem.remove(nodeid)
            if len(nodes_elem) >= 3: #on regarde si l'élement reste valide
                nodes_elem = [n-1 if n > nodeid else n for n in nodes_elem]
                new_elem2nodes.extend(nodes_elem)
                new_p_elem2nodes.append(new_p_elem2nodes[-1]+len(nodes_elem)) #on change les pointages de p_elem2nodes
            #si c'est un element qui est trop petit on le supprime
        else :
            nodes_elem = [n-1 if n > nodeid else n for n in nodes_elem] #si le noeud n'est pas dans l'élement, on ne change rien
            new_elem2nodes.extend(nodes_elem)
            new_p_elem2nodes.append(new_p_elem2nodes[-1]+len(nodes_elem))


    elem2nodes =numpy.array(new_elem2nodes, dtype=int) #on transforme en liste numpy
    p_elem2nodes = numpy.array(new_p_elem2nodes, dtype=int)

    node_coords = numpy.delete(node_coords, nodeid, axis=0)
    return node_coords, p_elem2nodes, elem2nodes,

def remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid):
    start, end = p_elem2nodes[elemid], p_elem2nodes[elemid+1] #élement à retirer
    elem2nodes = numpy.delete(elem2nodes, numpy.s_[start:end]) #on supprime l'élement de la liste, attention changement d'indices
    p_elem2nodes = numpy.delete(p_elem2nodes, elemid+1)#on supprime le pointeur de p_elem2nodes
    p_elem2nodes[elemid+1:] -=(end-start) #on décale les pointages de tout les élements suivants
    return node_coords, p_elem2nodes, elem2nodes,





def frontiere(x1,y1,x2,y2,level): #définition du segment d'origine
    global fractale #liste où on va mettre les points de la frontière
    if level == 0:
        fractale.append([x2, y2, 0.0])
        return

    # définition de la forme de notre fractale
    x3, y3 = x1 + (x2-x1)/4, y1 + (y2-y1)/4
    x4, y4 = x3 - (y2-y1)/4, y3 + (x2-x1)/4
    x5, y5 = x4 + (x2-x1)/4, y4 + (y2-y1)/4
    x6, y6 = x5 + (y2-y1)/4, y5 - (x2-x1)/4
    x7, y7 = x6 + (y2-y1)/4, y6 - (x2-x1)/4
    x8, y8 = x7 + (x2-x1)/4, y7 + (y2-y1)/4
    x9, y9 = x1 + 3*(x2-x1)/4, y1 + 3*(y2-y1)/4

    #récursivité
    frontiere(x1, y1, x3, y3, level-1)
    frontiere(x3, y3, x4, y4, level-1)
    frontiere(x4, y4, x5, y5, level-1)
    frontiere(x5, y5, x6, y6, level-1)
    frontiere(x6, y6, x7, y7, level-1)
    frontiere(x7, y7, x8, y8, level-1)
    frontiere(x8, y8, x9, y9, level-1)
    frontiere(x9, y9, x2, y2, level-1)

def point_dans_frontiere(points, p): # on regarde si le point p est dans la forme définie par la barriere, méthode de ray casting
    npoints =len(points)
    intersection= 0 #on compte les intersections entre une droite passant par p et les côté de notre frontière
    for i in range(npoints):
        cpoint = points[i]
        next_point = points[(i + 1) % npoints]
        if (cpoint[1]<p[1] and next_point[1]>=p[1]) or (cpoint[1]>=p[1] and next_point[1]<p[1]):
            if cpoint[0] + (p[1]-cpoint[1]) / (next_point[1]-cpoint[1])*(next_point[0]-cpoint[0])<=p[0]:
                intersection+= 1
    return intersection %2 == 1


niveau_de_fractale = 2
xmin, xmax, ymin, ymax = 0.0, 100.0, 0, 100.0
nelemsx, nelemsy = 64,64 #pratique pour diviser
nelems = nelemsx * nelemsy
nnodes = (nelemsx + 1) * (nelemsy + 1)

#maillage création
node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)


#Création de la fractale
fractale = []
fractale.append([0.0, ymax, 0.0])
fractale.append([0.0, ymax/2, 0.0])
frontiere(0, ymax/2, xmax, ymax/2, niveau_de_fractale)
fractale.append([xmax, ymax, 0.0])


#On calcule les barycentres, et on retire les élèments qui doivent l'être 
barycentres = compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes)
remove = []
for id, point in enumerate(barycentres):
    if point_dans_frontiere(fractale, point):
        remove.append(id)
remove.reverse()  #on doit l'inverser pour ne pas avoir de problème d'indentation
for id_remove in remove:
    node_coords, p_elem2nodes, elem2nodes = remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, id_remove)

p_elem2nodes, elem2nodes=quadtotri(node_coords,p_elem2nodes,elem2nodes)

solutions._plot_mesh(p_elem2nodes, elem2nodes, node_coords)
plt.show()

############ On résout Helmholtz ############

def build_node2elems(p_elem2nodes, elem2nodes):
    e2n_coef = numpy.ones(len(elem2nodes), dtype=numpy.int64)
    e2n_mtx = scipy.sparse.csr_matrix((e2n_coef, elem2nodes, p_elem2nodes))
    n2e_mtx = e2n_mtx.transpose()
    n2e_mtx = n2e_mtx.tocsr()
    p_node2elems = n2e_mtx.indptr
    node2elems = n2e_mtx.indices
    return p_node2elems, node2elems

#identifier les noeuds sur la frontiere
def sur_frontiere(p_elem2nodes, elem2nodes, nodeid):
    p_node2elems, node2elems = build_node2elems(p_elem2nodes, elem2nodes)
    if nodeid + 1 < len(p_node2elems) and len(node2elems[p_node2elems[nodeid]:p_node2elems[nodeid+1]]) == 6:
        return False
    else:
        return True

def solve_helmholtz():
     
    #on a notre maillage fractal
    
    nodes_on_boundary = [nodeid for nodeid in range(len(node_coords)) if sur_frontiere(p_elem2nodes, elem2nodes, nodeid)]
    wavenumbers = [k*numpy.pi for k in [5/4]] #fréquences différentes [0.1,0.5,3/4, 1,2]
    nnodes = node_coords.shape[0]

    for wavenumber in wavenumbers:

        #on a déjà les valeurs aux bords    

        solexact = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

        # -- set boundary conditions
        values_at_nodes_on_boundary = numpy.zeros((nnodes, 1), dtype=numpy.complex128)
        values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

        # -- set finite element matrices and right hand side
        f_unassembled = numpy.zeros((nnodes, 1), dtype=numpy.complex128)

        # ..warning: for teaching purpose only
        for i in range(nnodes):
            # evaluate: (-\Delta - k^2) u(x,y) = ...
            f_unassembled[i] = 1
        # ..warning: end

        coef_k = numpy.ones((nelems, 1), dtype=numpy.complex128)
        coef_m = numpy.ones((nelems, 1), dtype=numpy.complex128)
        K, M, F = zsolutions4students._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
        A = K - wavenumber**2 * M
        B = F

        # -- apply Dirichlet boundary conditions
        A, B = zsolutions4students._set_dirichlet_condition(nodes_on_boundary, values_at_nodes_on_boundary, A, B)

        # -- solve linear system
        sol = scipy.linalg.solve(A, B)

        # -- plot finite element solution
        solreal = sol.reshape((sol.shape[0], ))
        _ = solutions._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, numpy.real(solreal))
        print(wavenumber)
    return A, solreal

ax = plt.subplot(1, 1, 1)
ax.set_aspect('equal')
ax.axis('off')
solve_helmholtz()
plt.show()

"""
A, solreal = solve_helmholtz()
eig_vals, eig_vecs = scipy.sparse.linalg.eigs(A)


def solve_helmholtz_modes():
    n = eig_vals.shape[0]
    # Affichage des modes propres
    for i in range(n):
        mode = numpy.real(eig_vecs[:, i])
        print(f"Mode {i+1} — λ = {eig_vals[i]}")
        _ = zsolutions4students._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, mode)
    return eig_vals, eig_vecs


ax = plt.subplot(1, 1, 1)
ax.set_aspect('equal')
ax.axis('off')
solve_helmholtz_modes()"""

