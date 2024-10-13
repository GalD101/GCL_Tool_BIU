import gc

# Data manipulation libraries
import numpy as np
import pandas as pd

# Mathematical functions (ode solvers, spearman, PCA)
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, scale

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

## TODO: Refactor to use pandas dataframes

from gcl_library import jackknife


# Set up the system of ODEs
def ode_system(t, x, W, B, zero_mask):
    n = len(x)  # Get the number of elements in the state vector
    derivative_vector = np.zeros(n)  # Initialize the derivative vector (set zeros as default)
    for i in range(n):  # Loop over each element in the state vector
        # Compute the sum term for the i-th element according to the formula given in the paper
        sum_term = sum(W[i, j] * (x[j] / (1 + x[j])) for j in range(n) if j != i)
        derivative_vector[i] = -B * x[i] + sum_term
    # Set the derivative of the zeroed genes to 0
    for z in zero_mask:
        derivative_vector[z] = 0
    return derivative_vector


def run_simulation(run_index):
    # Initialize parameters
    N = 200  # Number of genes in a cell
    num_of_cells = 100  # Number of cells in each cohort
    avg_deg = 3  # Average number of non-zero elements in the matrix W
    q = 2  # Affects the range of the random numbers generated
    num_of_cohorts = 5
    p = 0.25
    step_size = 1.0 / (num_of_cohorts - 1) if num_of_cohorts > 1 else 0
    p_ratios = [i * step_size for i in range(num_of_cohorts)]
    num_of_cells_to_affect = [round(p_ratios[i] * num_of_cells) for i in range(num_of_cohorts)]
    decide = lambda x: np.random.random() < x

    W = [[np.zeros((N, N)) for _ in range(num_of_cells)] for ___ in range(num_of_cohorts)]
    initial_conditions = [[np.random.rand(N) for _ in range(num_of_cells)] for __ in range(num_of_cohorts)]
    num_of_time_stamps = 1000
    t_final = 20
    t = np.linspace(0, t_final, num_of_time_stamps)
    print("FINISHED INITIALIZING PARAMETERS")

    for cohort_index in range(num_of_cohorts):
        cur_num_of_cells_to_affect = num_of_cells_to_affect[cohort_index]
        base_model = np.array([[np.random.uniform(0, q) if (gene_i != gene_j and decide(avg_deg / (N - 1))) else 0
                                for gene_j in range(N)] for gene_i in range(N)])
        for defect_index in range(num_of_cells):
            effective_p_value = p if defect_index <= cur_num_of_cells_to_affect else 0
            for gene_i in range(N):
                for gene_j in range(N):
                    if base_model[gene_i, gene_j] != 0:
                        if decide(effective_p_value):
                            W[cohort_index][defect_index][gene_i, gene_j] = np.random.uniform(0, q)
                        else:
                            W[cohort_index][defect_index][gene_i, gene_j] = base_model[gene_i, gene_j]
                    else:
                        W[cohort_index][defect_index][gene_i, gene_j] = 0

    print("FINISHED CREATING W")
    results = [[np.zeros((num_of_time_stamps, N)) for _ in range(num_of_cells)] for __ in range(num_of_cohorts)]
    B = 1
    for cohort_index in range(num_of_cohorts):
        print("STARTING COHORT", cohort_index)
        for defect_index in range(num_of_cells):
            print("STARTING DEFECT", defect_index)
            zero_mask = []
            num_of_zero_genes = 5
            for _ in range(num_of_zero_genes):
                gene_index = np.random.randint(0, N - 1)
                initial_conditions[cohort_index][defect_index][gene_index] = 0
                zero_mask.append(gene_index)

            print("SOLVING START")
            results[cohort_index][defect_index] = solve_ivp(
                ode_system, [t[0], t[-1]], initial_conditions[cohort_index][defect_index],
                args=(W[cohort_index][defect_index], B, zero_mask), t_eval=t
            )["y"].T
            print("SOLVING FINISHED")

    print("FINISHED SOLVING, START PLOTTING")
    # for cohort_index in range(num_of_cohorts):
    #     for defect_index in range(num_of_cells):
    #         for gene in range(N):
    #             plt.plot(t, results[cohort_index][defect_index][:, gene])
    # plt.xlabel('Time')
    # plt.ylabel('x value')
    # plt.title(f'Michaelis-Menten kinetics - Run {run_index}')
    # plt.savefig(f'plot_run_{run_index}results.png')
    # plt.close()
    # gc.collect()

    steady_state = np.zeros((num_of_cohorts, N, num_of_cells))
    for cohort_j in range(num_of_cohorts):
        for cell_index in range(num_of_cells):
            steady_state[cohort_j, :, cell_index] = results[cohort_j][cell_index][
                -1]
    print("FINISHED CALCULATING STEADY STATE")

    negative_spearman_steady_state = [(1 - (spearmanr(steady_state[i])[0])) for i in range(num_of_cohorts)]
    off_diagonal_elements = [[] for _ in range(num_of_cohorts)]

    for cohort_index in range(num_of_cohorts):
        for cell_i in range(num_of_cells):
            for cell_j in range(num_of_cells):
                if cell_j > cell_i:
                    off_diagonal_elements[cohort_index].append(negative_spearman_steady_state[cohort_index][cell_i][cell_j])

    print("FINISHED CALCULATING OFF DIAGONAL")
    # Round to 2 decimal places to make the graph look normal
    off_diagonal_elements = [np.round(element, decimals=4) for element in off_diagonal_elements]

    # Plot the off-diagonal elements distribution
    for cohort_index in range(num_of_cohorts):
        plt.hist(off_diagonal_elements[cohort_index], alpha=0.5,
                 label='cohort with {:.2f}% affected cells'.format(p_ratios[cohort_index]))

    plt.title('Off-diagonal negative spearman matrix elements')  # (matrix is symmetric)
    plt.xlabel('Negative Spearman value')
    plt.ylabel('Frequency')

    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(f'plot_run_{run_index}off_diagonal.png')
    # plt.close()
    # gc.collect()

    flattened_data = []
    cohort_labels = []

    # Flatten results and create cohort labels (p = 0, ..., p = 1)
    for cohort_index in range(num_of_cohorts):
        for cell_index in range(num_of_cells):
            # Flatten gene data for each cell and add to flattened_data
            flattened_data.append(negative_spearman_steady_state[cohort_index][cell_index].flatten())
            cohort_labels.append(f'percentage of affected cells = {p_ratios[cohort_index]}')

    # Convert to a DataFrame
    flattened_data = np.array(flattened_data)
    data_df = pd.DataFrame(flattened_data)

    # Step 2: Scale the data across *all* cells
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_df)

    # Step 3: Perform PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    pc1 = 'Principal Component 1 (accounts for {:.2f}% of the variation)'.format(pca.explained_variance_ratio_[0] * 100)
    pc2 = 'Principal Component 2 (accounts for {:.2f}% of the variation)'.format(pca.explained_variance_ratio_[1] * 100)

    # Convert PCA result to a DataFrame and add cohort information
    pca_df = pd.DataFrame(data=pca_data, columns=[pc1, pc2])
    pca_df['cohort'] = cohort_labels

    # Step 4: Plot the PCA results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pc1, y=pc2, hue='cohort', data=pca_df, legend='full')
    plt.title('PCA of All Cells group by Cohort')
    plt.show()
    # plt.savefig(f'plot_run_{run_index}pca.png')
    # plt.close()
    # gc.collect()

    jackknife_results = []
    for cohort_index in range(num_of_cohorts):
        jackknife_results.append(jackknife(steady_state[cohort_index], 50, 0.75, 50))

    # Plot a violin plot of the jackknife results such that every violin represents a cohort
    plt.figure(figsize=(10, 8))
    sns.violinplot(data=jackknife_results, legend='full')
    plt.title('Jackknife results for every cohort')
    plt.show()
    # plt.savefig(f'plot_run_{run_index}jackknife.png')
    # plt.close()
    # gc.collect()



# Run the simulation multiple times and save the plots
num_runs = 10
for run_index in range(num_runs):
    print("STARTING RUN")
    run_simulation(run_index)
