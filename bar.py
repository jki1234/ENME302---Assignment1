import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)
import ast
import matplotlib.patches as patches
from prettytable import PrettyTable


DISP_MAG = 10**2


def local_bar(E, A, L):
    k = np.array([
        [1, -1],
        [-1, 1]
        ])
    
    return k * E * A / L

def global_bar(E, A, L, alpha):
    k = local_bar(E, A, L)
    rad = np.radians(alpha)
    A_e = (np.array([
        [np.cos(rad), np.sin(rad), 0, 0], 
        [0, 0, np.cos(rad), np.sin(rad)]
        ]))
    
    k_e_hat = (A_e.T @ k @ A_e)

    return k_e_hat, A_e, k


def total_stiffness_matrix(A, K_hat):
    total = np.zeros((A[0].shape[0], A[0].shape[0]))
    for i in range(np.array(A).shape[0]):
        total += A[i] @ K_hat[i] @ A[i].T
        
    return total
    
    
def force(K_hat, Assembly, q):
    return K_hat @ (Assembly.T @ q)

def displacement(A, Assembly, q):
    return A @ Assembly.T @ q

def strain(d1, d2, L):
    return (d2 - d1) / L

def plot_deflected_shape(coords, displaced_coordinates, d):
    fig, ax = plt.subplots()

    # Track labeled nodes to avoid redundant information
    labeled_nodes = set()
    for i in range(len(coords)):
        node2XGlobal = coords[i][1, 0]
        node2YGlobal = coords[i][1, 1]


        x_mid = np.average(coords[i][:, 0])
        y_mid = np.average(coords[i][:, 1])
        plt.text(x_mid, y_mid, f'[{i+1}]', fontsize=12, ha='center')
       
        line1, = ax.plot(coords[i][:, 0], coords[i][:, 1], '-', label = f"Element {i+1}")
        color = line1.get_color()
        ax.plot(displaced_coordinates[i][:, 0], displaced_coordinates[i][:, 1], '--', color=color)
        
        # Adding displacement arrows and labels at nodes
        if (node2XGlobal, node2YGlobal) not in labeled_nodes:
            global_displacement = (displaced_coordinates[i] - coords[i]) / DISP_MAG
            #plt.text(displaced_coordinates[i][1, 0], displaced_coordinates[i][1, 1], f'Δx={1e3*global_displacement[1, 0]:.4f} [mm]\nΔy={1e3*global_displacement[1, 1]:.4f} [mm]', color='blue', fontsize=10, ha='center')
            plt.text(displaced_coordinates[i][1, 0], displaced_coordinates[i][1, 1], f'Δx={1e3*global_displacement[1, 0]:.4f} [mm]\nΔy={1e3*global_displacement[1, 1]:.4f} [mm]', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            
            labeled_nodes.add((node2XGlobal, node2YGlobal))

    ax.legend(fontsize='small')  # Set legend font size to 'small'
    plt.title('Deflected shape of the bar elements')
    plt.xlabel('$X^G$ [m]')
    plt.ylabel('$Y^G$ [m]')
    plt.grid()
    plt.show()
    
def plot_dof(coords, displaced_coordinates, d):
    labeled_nodes = set()
    g_radius = 0.3

    g_x = 3
    g_y = 0.5

    #Global y
    plt.plot([g_x, g_x], [g_y, g_y + 2 * g_radius],  color='black', zorder=3)
    label_vert = r"+'ve $Y^G$ " 
    plt.text(g_x, g_y + 2.25 * g_radius, label_vert, fontsize=12, ha='center', zorder=3)

    #Global x
    plt.plot([g_x, g_x + 2 * g_radius], [g_y, g_y],  color='black', zorder=3)
    label_hor = r"+'ve $X^G$" 
    plt.text(g_x + 2.5 * g_radius, g_y, label_hor, fontsize=12, ha='center', zorder=3)


    n = 0
    for i in range(len(coords)):

        node1XGlobal = coords[i][0, 0]
        node1YGlobal = coords[i][0, 1]
        node2XGlobal = coords[i][1, 0]
        node2YGlobal = coords[i][1, 1]
    
        x_mid = (node1XGlobal + node2XGlobal) / 2
        y_mid = (node1YGlobal + node2YGlobal) / 2
        plt.text(x_mid, y_mid, f'[{i+1}]', fontsize=12, ha='center')

        # Plot the original and deflected shape
        plt.plot(coords[i][:,0], coords[i][:,1], label=f'Element [{i+1}]')

         # Adding displacement arrows and labels at nodes
        if (node2XGlobal, node2YGlobal) not in labeled_nodes:
            radius = 0.15

            #Vertical degree of freedom
            plt.plot([node2XGlobal, node2XGlobal], [node2YGlobal, node2YGlobal + 2 * radius],  color='black', zorder=3)
            label_vert = r"$q_{%d}$" % (2*(n+1))
            plt.text(node2XGlobal, node2YGlobal + 2.5 * radius, label_vert, fontsize=12, ha='center', zorder=3)

            #Horizontal degree of freedom
            plt.plot([node2XGlobal, node2XGlobal + 2 * radius], [node2YGlobal, node2YGlobal],  color='black', zorder=3)
            label_hor = r"$q_{%d}$" % (2*(n+1)-1)
            plt.text(node2XGlobal + 2.5 * radius, node2YGlobal, label_hor, fontsize=12, ha='center', zorder=3)

            n += 1
            labeled_nodes.add((node2XGlobal, node2YGlobal))
        

    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.title('Degrees of freedom for the bar elements')
    plt.xlabel('$X^G$ [m]')
    plt.ylabel('$Y^G$ [m]')
    
    plt.show()



def generate_latex_matrix(matrix, label=None):
    """
    Generate LaTeX code for a matrix with optional label.
    
    Parameters:
        matrix (2D list or np.array): The matrix to convert to LaTeX.
        label (str, optional): The label for referencing the matrix.
    
    Returns:
        str: LaTeX code for the matrix.
    """
    # Convert the matrix to LaTeX format
    matrix_str = '\n'.join(
        ' & '.join(f'{elem:.0f}' for elem in row) + r' \\'
        for row in matrix
    )
    
    # Create the LaTeX environment
    latex_code = (
        r'$\begin{bmatrix}'
        f'{matrix_str}'
        r'\end{bmatrix}$'
    )
    
    if label:
        latex_code += f'\\label{{{label}}}'
    
    return latex_code

def generate_latex_table(matrices, labels=None, captions=None, add_divider=False, add_space=False):
    """
    Generate LaTeX code for a 2x4 table of matrices with optional divider or space, and mini captions.
    
    Parameters:
        matrices (list of 2D lists or np.arrays): List of matrices to include in the table.
        labels (list of str, optional): List of labels for the matrices.
        captions (list of str, optional): List of captions for each matrix.
        add_divider (bool, optional): Whether to add a horizontal divider between rows.
        add_space (bool, optional): Whether to add vertical space between rows.
    
    Returns:
        str: LaTeX code for the 2x4 table of matrices.
    """
    if labels is None:
        labels = [None] * len(matrices)
    if captions is None:
        captions = [f'Matrix {i+1}' for i in range(len(matrices))]
    
    # Generate LaTeX code for each matrix with its mini caption
    matrix_latex = [
        f'\\textbf{{{caption}}} \\\n{generate_latex_matrix(m, l)}'
        for m, l, caption in zip(matrices, labels, captions)
    ]
    
    # Generate the LaTeX code for the table
    table_latex = r'\begin{table}[htbp]' + '\n'
    table_latex += r'\centering' + '\n'
    table_latex += r'\begin{tabular}{cccc}' + '\n'
    
    for i in range(0, len(matrices), 4):
        row = ' & '.join(f'\\begin{{minipage}}{{0.22\\textwidth}}\n{matrix_latex[j]}\n\\end{{minipage}}' for j in range(i, min(i + 4, len(matrices))))
        table_latex += row + r'\\' + '\n'
        
        if add_divider and (i + 4) < len(matrices):
            table_latex += r'\hline' + '\n'
        elif add_space and (i + 4) < len(matrices):
            table_latex += r'\\[10pt]' + '\n'  # Adds vertical space between rows (10pt)

    table_latex += r'\end{tabular}' + '\n'
    table_latex += r'\caption{Table of Assembly Matrices}' + '\n'
    table_latex += r'\label{tab:assembly_matrices}' + '\n'
    table_latex += r'\end{table}' + '\n'
    
    return table_latex





def main():
    #Input the coordinates of the nodes of the frame
    mem_1_coords = np.array([
        [0, 0],
        [1.33, 0]
    ])

    mem_2_coords = np.array([
        [0, -1.2], 
        [1.33, 0]
        
    ])

    mem_3_coords = np.array([
        [0, -1.2],
        [2, -1.2]
    ])

    mem_4_coords = np.array([
        [1.33, 0],
        [2, -1.2]
    ])

    mem_5_coords = np.array([
        [1.33, 0],
        [2.66, -0.4]
    ])

    mem_6_coords = np.array([
        [2, -1.2], 
        [2.66, -0.4]
        
    ])

    mem_7_coords = np.array([
        [2, -1.2],
        [4, -1.2]
    ])

    mem_8_coords = np.array([
        [2.66, -0.4], 
        [4, -1.2]
        
    ])

    outer_diameter = lambda x: x / 2000
    inner_diameter = lambda x: x / 2000

    d_o = outer_diameter(50)
    d_i = inner_diameter(38)

    Area = np.pi * (d_o**2 - d_i**2) #m^2
    E = 200e9

    

    # Define the properties of the bar
    
    K1_hat, Lambda1, Ke_1 = global_bar(E, Area, np.linalg.norm(mem_1_coords[1] - mem_1_coords[0]), 0)
    K2_hat, Lambda2, Ke_2 = global_bar(E, Area, np.linalg.norm(mem_2_coords[1] - mem_2_coords[0]), 42.06)
    K3_hat, Lambda3, Ke_3 = global_bar(E, Area, np.linalg.norm(mem_3_coords[1] - mem_3_coords[0]), 0)
    K4_hat, Lambda4, Ke_4 = global_bar(E, Area, np.linalg.norm(mem_4_coords[1] - mem_4_coords[0]), -60.82)
    K5_hat, Lambda5, Ke_5 = global_bar(E, Area, np.linalg.norm(mem_5_coords[1] - mem_5_coords[0]), -16.74)
    K6_hat, Lambda6, Ke_6 = global_bar(E, Area, np.linalg.norm(mem_6_coords[1] - mem_6_coords[0]), 50.48)
    K7_hat, Lambda7, Ke_7 = global_bar(E, Area, np.linalg.norm(mem_7_coords[1] - mem_7_coords[0]), 0)
    K8_hat, Lambda8, Ke_8 = global_bar(E, Area, np.linalg.norm(mem_8_coords[1] - mem_8_coords[0]), -30.84)




    A1 = np.array([
        [0, 0, 1, 0], # q1
        [0, 0, 0, 1], # q2
        [0, 0, 0, 0], # q3
        [0, 0, 0, 0], # q4
        [0, 0, 0, 0], # q5
        [0, 0, 0, 0], # q6
        [0, 0, 0, 0], # q7
        [0, 0, 0, 0]  # q8
    ])

    A2 = np.array([
        [0, 0, 1, 0], # q1
        [0, 0, 0, 1], # q2
        [0, 0, 0, 0], # q3
        [0, 0, 0, 0], # q4
        [0, 0, 0, 0], # q5
        [0, 0, 0, 0], # q6
        [0, 0, 0, 0], # q7
        [0, 0, 0, 0]  # q8
    ])

    A3 = np.array([
        [0, 0, 0, 0], # q1
        [0, 0, 0, 0], # q2
        [0, 0, 1, 0], # q3
        [0, 0, 0, 1], # q4
        [0, 0, 0, 0], # q5
        [0, 0, 0, 0], # q6
        [0, 0, 0, 0], # q7
        [0, 0, 0, 0]  # q8
    ])

    A4 = np.array([
        [1, 0, 0, 0], # q1
        [0, 1, 0, 0], # q2
        [0, 0, 1, 0], # q3
        [0, 0, 0, 1], # q4
        [0, 0, 0, 0], # q5
        [0, 0, 0, 0], # q6
        [0, 0, 0, 0], # q7
        [0, 0, 0, 0]  # q8
    ])

    A5 = np.array([ 
        [1, 0, 0, 0], # q1
        [0, 1, 0, 0], # q2
        [0, 0, 0, 0], # q3
        [0, 0, 0, 0], # q4
        [0, 0, 1, 0], # q5
        [0, 0, 0, 1], # q6
        [0, 0, 0, 0], # q7
        [0, 0, 0, 0]  # q8
    ])

    A6 = np.array([ 
        [0, 0, 0, 0], # q1
        [0, 0, 0, 0], # q2
        [1, 0, 0, 0], # q3
        [0, 1, 0, 0], # q4
        [0, 0, 1, 0], # q5
        [0, 0, 0, 1], # q6
        [0, 0, 0, 0], # q7
        [0, 0, 0, 0]  # q8
    ])

    A7 = np.array([
        [0, 0, 0, 0], # q1
        [0, 0, 0, 0], # q2
        [1, 0, 0, 0], # q3
        [0, 1, 0, 0], # q4
        [0, 0, 0, 0], # q5
        [0, 0, 0, 0], # q6
        [0, 0, 1, 0], # q7
        [0, 0, 0, 1]  # q8
    ])

    A8 = np.array([
        [0, 0, 0, 0], # q1
        [0, 0, 0, 0], # q2
        [0, 0, 0, 0], # q3
        [0, 0, 0, 0], # q4
        [1, 0, 0, 0], # q5
        [0, 1, 0, 0], # q6
        [0, 0, 1, 0], # q7
        [0, 0, 0, 1]  # q8
    ])


    #Update these!!!
    coords = [mem_1_coords, mem_2_coords, mem_3_coords, mem_4_coords, mem_5_coords, mem_6_coords, mem_7_coords, mem_8_coords]
    Kes = [Ke_1, Ke_2, Ke_3, Ke_4, Ke_5, Ke_6, Ke_7, Ke_8]
    Assembly_matrices = [A1, A2, A3, A4, A5, A6, A7, A8]    
    lambdas  = [Lambda1, Lambda2, Lambda3, Lambda4, Lambda5, Lambda6, Lambda7, Lambda8]
    K_hats = [K1_hat, K2_hat, K3_hat, K4_hat, K5_hat, K6_hat, K7_hat, K8_hat]

    Q = np.array([0, 0, 0, 0, 0, 0, 0, -25000], dtype = np.float64).T
    

    KG = total_stiffness_matrix(Assembly_matrices, K_hats) 
    q = np.linalg.solve(KG, Q)

    d = []
    
    for i in range(len(coords)):
        d.append(displacement(lambdas[i], Assembly_matrices[i], q))

    
    displaced_coordinates = np.copy(coords)

    for i in range(len(coords)):
        for n in range(Assembly_matrices[i].shape[0]):
            if Assembly_matrices[i][n][0] == 1:
                displaced_coordinates[i][0,0] += DISP_MAG*q[n]

            if Assembly_matrices[i][n][1] == 1:
                displaced_coordinates[i][0,1] += DISP_MAG*q[n]

            if Assembly_matrices[i][n][2] == 1:
                displaced_coordinates[i][1,0] += DISP_MAG*q[n]

            if Assembly_matrices[i][n][3] == 1:
                displaced_coordinates[i][1,1] += DISP_MAG*q[n]
   
    #plot_deflected_shape(coords, displaced_coordinates, d)
    #plot_dof(coords, displaced_coordinates, d)


    print(f"Element local displacements = {d}")
    print("Original vs Displaced coordinates of element members: \n")
    for i in range(len(coords)):
        print(f"Member [{i+1}]")
        print(f"Local displacement (x_e): {d[i]}")
        print(f"Global displacement(X_g, Y_g): {(displaced_coordinates[i] - coords[i]) / DISP_MAG}\n")

    print("_"*90)
    print(f"q displacements = {1e3*q.T}\n")
    print(f"Total Q force vector: {Q}\n\n")

    F_1 = force(K_hats[0], Assembly_matrices[0], q)
    F_2 = force(K_hats[1], Assembly_matrices[1], q)
    F_3 = force(K_hats[2], Assembly_matrices[2], q)

    top_left = F_1[0:2]
    bottom_left = F_2[0:2] + F_3[0:2]

    print(f"Support reactions:\nTop left support reactions: {1e-3*top_left}\nBottom left support reactions: {1e-3*bottom_left}\n\n")
    print(f"Sum of support reactions = {1e-3*(top_left + bottom_left)}[kN]\n\n")

    print("-"*50)
    print(f"Induced stress within each element:\n")
    for i in range(len(coords)):
        sigma_axial = abs(force(K_hats[i], Assembly_matrices[i], q)[-2] / Area)
        #print(f"Element {i+1} axial force: {1e-3*abs(force(K_hats[i], Assembly_matrices[i], q)[-2]):.2f} [kN]")
        print(f"Element {i+1} stress: {1e-6*(sigma_axial):.2f} [MPa]")

    labels = [f"matrix{i+1}" for i in range(8)]
    captions = [f'A{i+1}' for i in range(8)]

    # Set add_divider to True or add_space to True as needed
    latex_code = generate_latex_table(
        Assembly_matrices,
        labels,
        captions,
        add_divider=False,  # Set to True to add horizontal lines between rows
        add_space=True    # Set to True to add vertical space between rows
    )

    #print(latex_code)




main()




