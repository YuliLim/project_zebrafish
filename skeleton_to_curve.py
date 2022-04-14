"""
This module contains functions to transform skeletons to curves (objects of class Curve from genepy3d).
"""
#Transformation from skeleton to curve
import networkx as nx
from skan import skeleton_to_csgraph
import pandas as pd
import numpy as np
from genepy3d.obj.trees import Tree
from matplotlib import pyplot as plt


def skeleton_to_graph(skeleton):
    """
    This function transforms a skeleton to a graph of NetworkX and extracts its largest connected component. 
    This largest component is represented in .swc format and saved in 'skeleton_tree.swc'.
    INPUT: skeleton (2d boolean array)
    OUTPUT: none

    """
        # obtain pixel graph (in the sparse matrix representation) and the coordinates of its nodes
    pixel_graph, coordinates = skeleton_to_csgraph(skeleton)
    #print("Size of the initial pixel graph: ",pixel_graph.size/2)

    G = nx.from_scipy_sparse_matrix(pixel_graph)
    G.edges

    components = nx.connected_components(G)

    largest_cc = max(nx.connected_components(G), key=len)

        #undirected longest tree
    longest_tree = G.subgraph(largest_cc).copy()

        #directed longest tree with double edges (in both directions)
    G1 = nx.DiGraph(longest_tree)

        # directed longest tree with single edges
    dir_tree = nx.maximum_branching(G1)

        # construct swc. file from the skeleton tree
    N = len(largest_cc)
    node_id = list(largest_cc)
    structure = list(np.zeros(N))
    z = list(np.zeros(N))
    radius = list(np.zeros(N))
    parent_list = [list(dir_tree.predecessors(i)) for i in node_id]
    x = [coordinates[i,1] for i in node_id]
    y = [coordinates[i,0] for i in node_id]

    parent_id = []
    for i in range(len(parent_list)):
        if len(parent_list[i]):
            parent_id.append(int(parent_list[i][0]))
        else:
            parent_id.append(-1)
            
    df = pd.DataFrame({'Node_id': node_id, 'structure type': structure, 'x': x,'y': y,'z': z,'radius': radius,'parent_id': parent_id})
    np.savetxt('skeleton_tree.swc', df.values, fmt='%d')

    return

def graph_to_curve():
    """
    This function transforms a graph to a curve using genepy3d library. The .swc file of oriented graph representation is uploaded.
    First, all nodes of degree 1 (leaves and the root) are collected. From this nodes we will chose the 2 corresponding to the extremities of the curve:
    these are the nodes with the minimal and the maximal x-coordinate. The obtained curve are resampled in order to obtain smoother derivatives (later).
    INPUT: none
    OUTPUT: the obtained resampled curve   
    """
    filename = "skeleton_tree.swc"
    line = Tree.from_swc(filename)

        #get the main branch
    root_coordinates = line.get_coordinates(line.get_root())
    leaves_coordinates = line.get_coordinates(line.get_leaves()) 

    end_coordinates = pd.concat([root_coordinates,leaves_coordinates], ignore_index=False)
    #print(end_coordinates)
    
        # compute the nodes on the x-axis extremities 
    leaf_maxx = end_coordinates.x.idxmax()
    leaf_minx = end_coordinates.x.idxmin()
    #print("The first leaf ",leaf_minx)
    #print("The last leaf ",leaf_maxx)

        # build a path between the chosen nodes 
    main_branch = line.path(leaf_minx,leaf_maxx)

        # build a curve from this path
    curve = line.to_curve(main_branch)
    curve = curve.resample(50)

    return curve

def compute_derivatives(curve):
    """
    Compure derivatives of the curve in the sampled points. 
    Derivatives of the extreme points cannot be calculated correctly, replace them by derivatives of the neighboring points.
    INPUT: curve object
    OUTPUT: 1-d array of the curve derivatives
    """
    derivatives = curve.compute_derivative(deg=1,dt=1)

    #correct extremities
    derivatives[0] = derivatives[1]
    derivatives[-1] = derivatives[-2]

    return derivatives

def plot_derivatives(curve, derivatives):
    """
    Visual representation of the calculated derivatives.
    INPUT: a curve object, 1-d array of the curve derivatives
    OUTPUT: none
    """
    x, y, z = curve.coors[:,0].copy(), curve.coors[:,1].copy(), curve.coors[:,2].copy()

    dt = 0.1
    N_r = 40
    ind = 10

    fig = plt.figure(figsize=(10,7))
    plt.plot(x,y)
    plt.title("Derivatives of the curve")
    plt.gca().invert_yaxis()

    N = 20

    for j in range(N):
        step = int(len(x)/N)
        ind = step*j
        t_x = [x[ind] - derivatives[ind][0]*i*dt for i in range(N_r)]
        t_y = [y[ind] - derivatives[ind][1]*i*dt for i in range(N_r)]
        plt.plot(t_x,t_y)