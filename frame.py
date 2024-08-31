import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)
import ast
import matplotlib.patches as patches
DISP_MAG = 10**2

def local_frame(E, A, L, I):
    beta = A * L **2 / I
    k = np.array([
        [beta, 0, 0, -beta, 0, 0],
        [0, 12, 6*L, 0, -12, 6*L],
        [0, 6*L, 4*L**2, 0, -6*L, 2*L**2],
        [-beta, 0, 0, beta, 0, 0],
        [0, -12, -6*L, 0, 12, -6*L],
        [0, 6*L, 2*L**2, 0, -6*L, 4*L**2]
    ])
    return E * I * k / (L ** 3) 

def global_frame(E, I, A, L, alpha):
    k = local_frame(E, A, L, I)
    
    rad = np.radians(alpha)
    c = np.cos(rad)
    s = np.sin(rad)
    lambda_ = np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])
    zero_3x3 = np.zeros((3,3))
    A_e = np.block([
        [lambda_, zero_3x3],
        [zero_3x3, lambda_]
    ])
    k_e_hat = (A_e.T @ k @ A_e)
    mask = np.abs(k_e_hat) < 1
    k_e_hat[mask] = 0
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

def plot_deflected_shape(coords, d, N_points, q, DISP_MAG):

    # Track labeled nodes to avoid redundant information
    labeled_nodes = set()
    
    for i in range(len(coords)):

        node1XGlobal = coords[i][0, 0]
        node1YGlobal = coords[i][0, 1]
        node2XGlobal = coords[i][1, 0]
        node2YGlobal = coords[i][1, 1]

        L = np.linalg.norm(np.array([node1XGlobal, node1YGlobal]) - np.array([node2XGlobal, node2YGlobal]))
        x_e = np.linspace(0, L, N_points)
        phi_1 = lambda x: 1 - x/L
        phi_2 = lambda x: x/L

        N_1 = lambda x: 1 - 3 * x**2 / L**2 + 2 * x**3 / L**3
        N_2 = lambda x: x**3 / L**2 - 2 * x**2 / L + x
        N_3 = lambda x: 3 * x**2 / L**2 - 2 * x**3 / L**3
        N_4 = lambda x: x**3 / L**2 - x**2 / L

        u = lambda x: phi_1(x) * d[i][0] + phi_2(x) * d[i][3]
        v = lambda x: N_1(x) * d[i][1] + N_2(x) * d[i][2] + N_3(x) * d[i][4] + N_4(x) * d[i][5]

        alpha = np.arctan2(node2YGlobal - node1YGlobal, node2XGlobal - node1XGlobal)
        deflections_XG = u(x_e) * np.cos(alpha) - v(x_e) * np.sin(alpha)
        deflections_YG = u(x_e) * np.sin(alpha) + v(x_e) * np.cos(alpha) 

        if i == None: #For the X_G, Y_G displacement of an Element [i] at the midpoint
            dist = 3*L/4
            deflections_X = u(dist)*np.cos(alpha) - v(dist)*np.sin(alpha)
            deflections_Y = u(dist)*np.sin(alpha) + v(dist)*np.cos(alpha) 
            print(f"N_1 = {N_1(dist)}")
            print(f"N_2 = {N_2(dist)}")
            print(f"N_3 = {N_3(dist)}")
            print(f"N_4 = {N_4(dist)}")

            print(deflections_X)
            print(deflections_Y)

        Undeflected_baseline_XG = np.linspace(node1XGlobal, node2XGlobal, N_points)
        Undeflected_baseline_YG = np.linspace(node1YGlobal, node2YGlobal, N_points)

        Deflected_XG = Undeflected_baseline_XG + DISP_MAG * deflections_XG
        Deflected_YG = Undeflected_baseline_YG + DISP_MAG * deflections_YG

        x_mid = np.average(Undeflected_baseline_XG)
        y_mid = np.average(Undeflected_baseline_YG)
        plt.text(x_mid, y_mid, f"[{i+1}]", fontsize=12, ha='center')


        # Plot the original and deflected shape
        line1, = plt.plot(Undeflected_baseline_XG, Undeflected_baseline_YG, label=f'Element [{i+1}]')
        color = line1.get_color()
        plt.plot(Deflected_XG, Deflected_YG, '--', color=color)

        # Adding displacement arrows and labels at nodes
        if (node2XGlobal, node2YGlobal) not in labeled_nodes:
            plt.text(node2XGlobal + DISP_MAG * d[i][3] * 0.5, node2YGlobal + DISP_MAG * d[i][4] * 0.5, f'Δx={1e3*d[i][3]:.4f} [mm]\nΔy={1e3*d[i][4]:.4f} [mm]\nθ={1e3*d[i][5]:.4f}° [mrad]', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
            
            labeled_nodes.add((node2XGlobal, node2YGlobal))

    plt.legend()
    plt.grid()
    plt.title('Deflected shape of the frame elements')
    plt.xlabel('$X^G$ [m]')
    plt.ylabel('$Y^G$ [m]')
    
    plt.show()

def plot_dof(coords):
    # Track labeled nodes to avoid redundant information
    labeled_nodes = set()

    g_radius = 0.3
    g_theta2 = 315
    g_theta_rad = np.deg2rad(g_theta2)

    g_x = 3
    g_y = 0.5
    # Global z
    g_arc = patches.Arc([g_x, g_y], g_radius, g_radius, angle=0, theta1=45, theta2=g_theta2, linewidth=2, zorder=3)
    plt.gca().add_patch(g_arc)
    label_rot = r"+'ve $Z^G$ " 
    plt.text(g_x + 2*g_radius * np.cos(g_theta_rad)/2, g_y + 2*g_radius * np.sin(g_theta_rad)/2, label_rot, fontsize=12, ha='center', zorder=3)

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
            theta2 = 315
            theta_rad = np.deg2rad(theta2)

            # Rotational degree of freedom
            # Label with subscript using LaTeX format
            arc = patches.Arc([node2XGlobal, node2YGlobal], radius, radius, angle=0, theta1=45, theta2=theta2, linewidth=2, zorder=3)
            plt.gca().add_patch(arc)
            label_rot = r"$q_{%d}$" % (3*(n+1))
            plt.text(node2XGlobal + 2*radius * np.cos(theta_rad)/2, node2YGlobal + 2*radius * np.sin(theta_rad)/2, label_rot, fontsize=12, ha='center', zorder=3)

            #Vertical degree of freedom
            plt.plot([node2XGlobal, node2XGlobal], [node2YGlobal, node2YGlobal + 2 * radius],  color='black', zorder=3)
            label_vert = r"$q_{%d}$" % (3*(n+1)-1)
            plt.text(node2XGlobal, node2YGlobal + 2.5 * radius, label_vert, fontsize=12, ha='center', zorder=3)

            #Horizontal degree of freedom
            plt.plot([node2XGlobal, node2XGlobal + 2 * radius], [node2YGlobal, node2YGlobal],  color='black', zorder=3)
            label_hor = r"$q_{%d}$" % (3*(n+1)-2)
            plt.text(node2XGlobal + 2.5 * radius, node2YGlobal, label_hor, fontsize=12, ha='center', zorder=3)


            n += 1
            labeled_nodes.add((node2XGlobal, node2YGlobal))
        

    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.title('Degrees of freedom for the frame elements')
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

    I = np.pi * (d_o**4 - d_i**4) / 4 
    E = 200e9

    

    # Define the properties of the frame
    K1_hat, Lambda1, Ke_1 = global_frame(E, I, Area, np.linalg.norm(mem_1_coords[1] - mem_1_coords[0]), 0)
    K2_hat, Lambda2, Ke_2 = global_frame(E, I, Area, np.linalg.norm(mem_2_coords[1] - mem_2_coords[0]), 42.06)
    K3_hat, Lambda3, Ke_3 = global_frame(E, I, Area, np.linalg.norm(mem_3_coords[1] - mem_3_coords[0]), 0)
    K4_hat, Lambda4, Ke_4 = global_frame(E, I, Area, np.linalg.norm(mem_4_coords[1] - mem_4_coords[0]), -60.82)
    K5_hat, Lambda5, Ke_5 = global_frame(E, I, Area, np.linalg.norm(mem_5_coords[1] - mem_5_coords[0]), -16.74)
    K6_hat, Lambda6, Ke_6 = global_frame(E, I, Area, np.linalg.norm(mem_6_coords[1] - mem_6_coords[0]), 50.48)
    K7_hat, Lambda7, Ke_7 = global_frame(E, I, Area, np.linalg.norm(mem_7_coords[1] - mem_7_coords[0]), 0)
    K8_hat, Lambda8, Ke_8 = global_frame(E, I, Area, np.linalg.norm(mem_8_coords[1] - mem_8_coords[0]), -30.84)




    A1 = np.array([
        [0, 0, 0, 1, 0, 0], # q1
        [0, 0, 0, 0, 1, 0], # q2
        [0, 0, 0, 0, 0, 1], # q3
        [0, 0, 0, 0, 0, 0], # q4
        [0, 0, 0, 0, 0, 0], # q5
        [0, 0, 0, 0, 0, 0], # q6
        [0, 0, 0, 0, 0, 0], # q7
        [0, 0, 0, 0, 0, 0], # q8
        [0, 0, 0, 0, 0, 0], # q9
        [0, 0, 0, 0, 0, 0], # q10
        [0, 0, 0, 0, 0, 0], # q11
        [0, 0, 0, 0, 0, 0]  # q12
    ])

    A2 = np.array([
        [0, 0, 0, 1, 0, 0], # q1
        [0, 0, 0, 0, 1, 0], # q2
        [0, 0, 0, 0, 0, 1], # q3
        [0, 0, 0, 0, 0, 0], # q4
        [0, 0, 0, 0, 0, 0], # q5
        [0, 0, 0, 0, 0, 0], # q6
        [0, 0, 0, 0, 0, 0], # q7
        [0, 0, 0, 0, 0, 0], # q8
        [0, 0, 0, 0, 0, 0], # q9
        [0, 0, 0, 0, 0, 0], # q10
        [0, 0, 0, 0, 0, 0], # q11
        [0, 0, 0, 0, 0, 0]  # q12
    ])

    A3 = np.array([
        [0, 0, 0, 0, 0, 0], # q1
        [0, 0, 0, 0, 0, 0], # q2
        [0, 0, 0, 0, 0, 0], # q3
        [0, 0, 0, 1, 0, 0], # q4
        [0, 0, 0, 0, 1, 0], # q5
        [0, 0, 0, 0, 0, 1], # q6
        [0, 0, 0, 0, 0, 0], # q7
        [0, 0, 0, 0, 0, 0], # q8
        [0, 0, 0, 0, 0, 0], # q9
        [0, 0, 0, 0, 0, 0], # q10
        [0, 0, 0, 0, 0, 0], # q11
        [0, 0, 0, 0, 0, 0]  # q12
    ])

    A4 = np.array([
        [1, 0, 0, 0, 0, 0], # q1
        [0, 1, 0, 0, 0, 0], # q2
        [0, 0, 1, 0, 0, 0], # q3
        [0, 0, 0, 1, 0, 0], # q4
        [0, 0, 0, 0, 1, 0], # q5
        [0, 0, 0, 0, 0, 1], # q6
        [0, 0, 0, 0, 0, 0], # q7
        [0, 0, 0, 0, 0, 0], # q8
        [0, 0, 0, 0, 0, 0], # q9
        [0, 0, 0, 0, 0, 0], # q10
        [0, 0, 0, 0, 0, 0], # q11
        [0, 0, 0, 0, 0, 0]  # q12
    ])

    A5 = np.array([ 
        [1, 0, 0, 0, 0, 0], # q1
        [0, 1, 0, 0, 0, 0], # q2
        [0, 0, 1, 0, 0, 0], # q3
        [0, 0, 0, 0, 0, 0], # q4
        [0, 0, 0, 0, 0, 0], # q5
        [0, 0, 0, 0, 0, 0], # q6
        [0, 0, 0, 1, 0, 0], # q7
        [0, 0, 0, 0, 1, 0], # q8
        [0, 0, 0, 0, 0, 1], # q9
        [0, 0, 0, 0, 0, 0], # q10
        [0, 0, 0, 0, 0, 0], # q11
        [0, 0, 0, 0, 0, 0]  # q12
    ])

    A6 = np.array([ 
        [0, 0, 0, 0, 0, 0], # q1
        [0, 0, 0, 0, 0, 0], # q2
        [0, 0, 0, 0, 0, 0], # q3
        [1, 0, 0, 0, 0, 0], # q4
        [0, 1, 0, 0, 0, 0], # q5
        [0, 0, 1, 0, 0, 0], # q6
        [0, 0, 0, 1, 0, 0], # q7
        [0, 0, 0, 0, 1, 0], # q8
        [0, 0, 0, 0, 0, 1], # q9
        [0, 0, 0, 0, 0, 0], # q10
        [0, 0, 0, 0, 0, 0], # q11
        [0, 0, 0, 0, 0, 0]  # q12
    ])

    A7 = np.array([
        [0, 0, 0, 0, 0, 0], # q1
        [0, 0, 0, 0, 0, 0], # q2
        [0, 0, 0, 0, 0, 0], # q3
        [1, 0, 0, 0, 0, 0], # q4
        [0, 1, 0, 0, 0, 0], # q5
        [0, 0, 1, 0, 0, 0], # q6
        [0, 0, 0, 0, 0, 0], # q7
        [0, 0, 0, 0, 0, 0], # q8
        [0, 0, 0, 0, 0, 0], # q9
        [0, 0, 0, 1, 0, 0], # q10
        [0, 0, 0, 0, 1, 0], # q11
        [0, 0, 0, 0, 0, 1]  # q12
    ])

    A8 = np.array([
        [0, 0, 0, 0, 0, 0], # q1
        [0, 0, 0, 0, 0, 0], # q2
        [0, 0, 0, 0, 0, 0], # q3
        [0, 0, 0, 0, 0, 0], # q4
        [0, 0, 0, 0, 0, 0], # q5
        [0, 0, 0, 0, 0, 0], # q6
        [1, 0, 0, 0, 0, 0], # q7
        [0, 1, 0, 0, 0, 0], # q8
        [0, 0, 1, 0, 0, 0], # q9
        [0, 0, 0, 1, 0, 0], # q10
        [0, 0, 0, 0, 1, 0], # q11
        [0, 0, 0, 0, 0, 1]  # q12
    ])


    #Update these!!!
    coords = [mem_1_coords, mem_2_coords, mem_3_coords, mem_4_coords, mem_5_coords, mem_6_coords, mem_7_coords, mem_8_coords]
    Kes = [Ke_1, Ke_2, Ke_3, Ke_4, Ke_5, Ke_6, Ke_7, Ke_8]
    Assembly_matrices = [A1, A2, A3, A4, A5, A6, A7, A8]    
    lambdas  = [Lambda1, Lambda2, Lambda3, Lambda4, Lambda5, Lambda6, Lambda7, Lambda8]
    K_hats = [K1_hat, K2_hat, K3_hat, K4_hat, K5_hat, K6_hat, K7_hat, K8_hat]


    yudl = [0, 0, 0, 0, 0, 0, 0, 0] #Uniformly distributed load on some element in local Y direction
    ypl = [0, 0, 0, 0, 0, 0, 0, 0] #Point load applied some distance 'a' from Node (1) of the element in local Y direction
    ypl_a = [0, 0, 0, 0, 0, 0, 0, 0] #'a' distance from Node (1) of the element in local Y direction

    lvl = [0, 0, 0, 0, 0, 0, 0, 0] #Linearly varying load on some element in local Y direction

    xudl = [0, 0, 0, 0, 0, 0, 0, 0] #Uniformly distributed load on some element in local X direction
    xpl = [0, 0, 0, 0, 0, 0, 0, 0]  #Point load applied some distance 'b' from Node (1) of the element in local X direction
    xpl_b = [0, 0, 0, 0, 0, 0, 0, 0]  #'b' distance from Node (1) of the element in local X direction


    Q = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -25000, 0], dtype = np.float64).T
    
    
    Q_total = Q.copy()

    for i in range(len(coords)):
        L = np.linalg.norm(coords[i][1] - coords[i][0])
        a = L/2 # Point load is applied at the midpoint of the element by default
        b = L/2 # Point load is applied at the midpoint of the element by default

        if ypl_a[i] != 0:
            a = ypl_a[i]
        if xpl_b[i] != 0:
            b = xpl_b[i]

        #Uniformly distributed load in local Y direction
        f_eq_yudl = np.array([0, yudl[i]*L/2, yudl[i]*L**2/12, 0, yudl[i]*L/2, -yudl[i]*L**2/12])
        F_eq_yudl  = lambdas[i].T @ f_eq_yudl
        Q_eq_yudl = Assembly_matrices[i] @ F_eq_yudl
        Q_total += Q_eq_yudl.T
        #Uniformly distributed load axially in local X direction
        f_eq_xudl = xudl[i] * np.array([L/2, 0, 0, L/2, 0, 0])
        F_eq_xudl = lambdas[i].T @ f_eq_xudl
        Q_eq_xudl = Assembly_matrices[i] @ F_eq_xudl
        Q_total += Q_eq_xudl.T


        #Linearly varying distributed load in local Y direction
        f_eq_lvl = lvl[i] * np.array([0, 3*L/20, L**2/30, 0, 7*L/20, -L**2/20])
        F_eq_lvl = lambdas[i].T @ f_eq_lvl
        Q_eq_lvl = Assembly_matrices[i] @ F_eq_lvl
        Q_total += Q_eq_lvl.T


        #Point load in the local Y direction at a distance 'a' from Node (1) of the element
        f_eq_ypl_a = ypl[i] * np.array([0, 1 - 3 * a ** 2 / L **2 + 2 * a ** 3 / L ** 3, a ** 3 / L ** 2 - 2 * a ** 2 / L + a, 0, 3 * a ** 2 / L ** 2 - 2 * a ** 3 / L ** 3, a ** 3 / L ** 2 - a ** 2 / L])
        F_eq_ypl_a = lambdas[i].T @ f_eq_ypl_a
        Q_eq_ypl_a = Assembly_matrices[i] @ F_eq_ypl_a
        Q_total += Q_eq_ypl_a.T
        #Concentrated axial point load in the local X direction at a distance 'b' from Node (1) of the element
        f_eq_xpl_b = xpl[i] * np.array([1-b/L, 0, 0, b/L, 0, 0])
        F_eq_xpl_b = lambdas[i].T @ f_eq_xpl_b
        Q_eq_xpl_b = Assembly_matrices[i] @ F_eq_xpl_b
        Q_total += Q_eq_xpl_b.T
        

    
    

    KG = total_stiffness_matrix(Assembly_matrices, K_hats) 
    q = np.linalg.solve(KG, Q_total)

    d = []
    
    for i in range(len(coords)):
        d.append(displacement(lambdas[i], Assembly_matrices[i], q))

    
    for i in range(0):

        L = np.linalg.norm(coords[i][1] - coords[i][0])
        a = L/2 # Point load is applied at the midpoint of the element by default
        b = L/2 # Point load is applied at the midpoint of the element by default

        if ypl_a[i] != 0:
            a = ypl_a[i]
        if xpl_b[i] != 0:
            b = xpl_b[i]

        #Uniformly distributed load in local Y direction
        f_eq_yudl = np.array([0, yudl[i]*L/2, yudl[i]*L**2/12, 0, yudl[i]*L/2, -yudl[i]*L**2/12])
        F_eq_yudl = lambdas[i].T @ f_eq_yudl
    
        #Uniformly distributed load axially in local X direction
        f_eq_xudl = xudl[i] * np.array([L/2, 0, 0, L/2, 0, 0])
        F_eq_xudl = lambdas[i].T @ f_eq_xudl


        #Linearly varying distributed load in local Y direction
        f_eq_lvl = lvl[i] * np.array([0, 3*L/20, L**2/30, 0, 7*L/20, -L**2/20])
        F_eq_lvl = lambdas[i].T @ f_eq_lvl


        #Point load in the local Y direction at a distance 'a' from Node (1) of the element
        f_eq_ypl_a = ypl[i] * np.array([0, 1 - 3 * a ** 2 / L **2 + 2 * a ** 3 / L ** 3, a ** 3 / L ** 2 - 2 * a ** 2 / L + a, 0, 3 * a ** 2 / L ** 2 - 2 * a ** 3 / L ** 3, a ** 3 / L ** 2 - a ** 2 / L])
        F_eq_ypl_a = lambdas[i].T @ f_eq_ypl_a
       
        #Concentrated axial point load in the local X direction at a distance 'b' from Node (1) of the element
        f_eq_xpl_b = xpl[i] * np.array([1-b/L, 0, 0, b/L, 0, 0])
        F_eq_xpl_b = lambdas[i].T @ f_eq_xpl_b

        element_force = lambdas[i] @ force(K_hats[i], Assembly_matrices[i], q) - F_eq_yudl - F_eq_xudl- F_eq_lvl - F_eq_ypl_a - F_eq_xpl_b
        F_eq_total = F_eq_yudl + F_eq_xudl + F_eq_lvl + F_eq_ypl_a + F_eq_xpl_b


        print('-'*50)
        print(f"f_e for Element[{i+1}] in Local element coordinates: {Kes[i] @ d[i]}")
        print(f"Sum of f_eq's for Element[{i+1}] in Local element coordinates: {f_eq_yudl + f_eq_xudl + f_eq_lvl + f_eq_ypl_a + f_eq_xpl_b}")
        print(f"f_e{i+1} - f_eq(s): {Kes[i] @ d[i] - f_eq_yudl - f_eq_xudl - f_eq_lvl - f_eq_ypl_a - f_eq_xpl_b}\n")

        print(f"F{i+1} for Element {i+1} in Global system coordinates: {lambdas[i] @ force(K_hats[i], Assembly_matrices[i], q)}")
        print(f"Sum of F_eq's for Element[{i+1}] in Global system coordinates: {F_eq_total}")
        print(f"F{i+1} - F_eq(s): {lambdas[i] @ force(K_hats[i], Assembly_matrices[i], q) - F_eq_total}\n\n\n") 
    


    print(f"q displacements = {1e3*q.T}\n\n")

    #Support reactions

    F_1 = force(K_hats[0], Assembly_matrices[0], q) 
    F_2 = force(K_hats[1], Assembly_matrices[1], q)
    F_3 = force(K_hats[2], Assembly_matrices[2], q)

    top_left = F_1[0:3]
    bottom_left = F_2[0:3] + F_3[0:3]

    
    print(f"Total Q force vector: {Q_total}\n\n")
    

    print(f"Support reactions:\nTop left support reactions: {top_left}\nBottom left support reactions: {bottom_left}\n\n")
    print(f"Sum of support reactions = {1e-3*(top_left + bottom_left)}[kN]\n\n")

    print("-"*50)
    print(f"Induced stress within each element:\n")
    for i in range(len(coords)):
        sigma_axial = abs(force(K_hats[i], Assembly_matrices[i], q)[-3] / Area)
        sigma_bending = abs(force(K_hats[i], Assembly_matrices[i], q)[-1] * d_o / (2 * I))
        #print(f"Element {i+1} axial force: {1e-3*abs(force(K_hats[i], Assembly_matrices[i], q)[-3]):.2f} [kN]")
        print(f"Element {i+1} stress: {1e-6*(sigma_axial + sigma_bending):.2f} [MPa]")



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
   
    
    #plot_dof(coords)
    
    #plot_deflected_shape(coords, d, 10, q, DISP_MAG)

main()

