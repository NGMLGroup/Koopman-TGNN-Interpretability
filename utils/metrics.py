import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from einops import rearrange

# Configure Matplotlib to use LaTeX for text rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']


def threshold_based_detection(signal, ground_truth, threshold=None, window_size=5, plot=False):
    """
    Define a threshold that captures significant changes in the derivative.
    Compare the time instants where the derivative exceeds this threshold with the ground-truth instants.
    Use metrics such as precision, recall, and F1-score to quantify the correspondence.

    Args:
        signal (np.ndarray): The time signal.
        ground_truth (np.ndarray): The ground-truth signal.
        threshold (float): The threshold to apply, as percentage of the maximum derivative.
                            If `None`, it will be computed as mean+std of the derivative.
        window_size (int): The window size around each instant to consider.
        plot (bool): Whether to plot the signal and derivative.

    Returns:
        precision, recall, F1-score, baseline.
        figure: The plot of the signal and derivative.
    """

    derivative = np.abs(np.gradient(signal))
    
    if threshold is None:
        threshold = np.mean(derivative) + np.std(derivative)
    else:
        threshold = threshold * np.max(derivative)

    filter = np.ones(window_size)
    ground_truth_c = np.convolve(ground_truth, filter, mode='valid')
    ground_truth_i = np.where(ground_truth_c > 0)[0]
    
    derivative = derivative[window_size//2:-window_size//2+1]
    detected = np.where(derivative > threshold)[0]

    true_positives = np.intersect1d(detected, ground_truth_i)
    false_positives = np.setdiff1d(detected, ground_truth_i)
    false_negatives = np.setdiff1d(ground_truth_i, detected)

    precision_denominator = len(true_positives) + len(false_positives)
    recall_denominator = len(true_positives) + len(false_negatives)

    precision = len(true_positives) / precision_denominator if precision_denominator > 0 else 0
    recall = len(true_positives) / recall_denominator if recall_denominator > 0 else 0

    # Calculate F1 score with a check to avoid division by zero
    f1_score_denominator = precision + recall
    f1_score = 2 * (precision * recall) / f1_score_denominator if f1_score_denominator > 0 else 0

    # Baseline F1 score
    baseline_f1 = F1_baseline(ground_truth_i)

    if plot:

        # Plot the signal
        fig, ax = plt.subplots(figsize=(8, 2))
        ax2 = ax.twinx()
        
        # Plot the trajectory of the signal
        sig = signal[window_size//2:-window_size//2+1]
        ax.plot(sig, c='yellow', label=r"$s^{(i)}(\tau)$", zorder=1, linewidth=2)
        
        # Plot the ground truth
        ax2.plot(ground_truth_c, c='orangered', label=r"$m_t(\tau)$", zorder=1, linewidth=2)

        # Create a color scale using the derivative
        im = ax.imshow(derivative.reshape(1,-1), 
                       cmap='viridis', aspect='auto', alpha=1, zorder=0,
                       extent=[0, derivative.shape[0], ax.get_ylim()[0], ax.get_ylim()[1]])

        # Add detected points
        det = np.zeros_like(ground_truth_c)
        det[detected] = 1
        ax.plot(np.where(det, sig, np.nan),
                '*', markerfacecolor='yellow', markeredgecolor='deeppink', markersize=10,
                label='Detected')

        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'$w_t^{(i)}(\tau)$')

        # Add labels and legend
        ax.set_xlabel(r'Time $\tau$')

        # Combine handles and labels from both axes
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles2 += handles
        labels2 += labels

        # Add a single legend
        ax2.legend(handles2, labels2, loc='upper left')

        # Hide the secondary y-axis
        ax2.yaxis.set_visible(False)

        # Adjust layout
        plt.tight_layout()

        # Move the colorbar out of the plot
        cbar.ax.set_position([1.001, 0.275, 0.05, 0.65])

        return fig, precision, recall, f1_score, baseline_f1

    return None, precision, recall, f1_score, baseline_f1
    

def windowing_analysis(signal, ground_truth, window_size=5, threshold=None, plot=False):
    """
    Define a threshold that captures significant changes in the derivative of the
    moving average of the signal.
    Compare the time instants where the derivative exceeds this threshold with the ground-truth instants.
    Use metrics such as precision, recall, and F1-score to quantify the correspondence.

    Args:
        signal (np.ndarray): The time signal.
        ground_truth (np.ndarray): The ground-truth signal.
        window_size (int): The window size for the moving average.
                            And the window size around each instant to consider.
        threshold (float): The threshold to apply, as percentage of the maximum derivative.
                            If `None`, it will be computed as mean+std of the derivative.
        plot (bool): Whether to plot the signal and derivative.

    Returns:
        precision, recall, F1-score, baseline.
        figure: The plot of the signal and derivative.
    """

    # Moving average
    signal = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
    derivative = np.abs(np.gradient(signal))
    
    if threshold is None:
        threshold = np.mean(derivative) + np.std(derivative)
    else:
        threshold = threshold * np.max(derivative)
    
    detected = np.where(derivative > threshold)[0]
    
    filter = np.ones(window_size)
    ground_truth_c = np.convolve(ground_truth, filter, mode='valid')
    ground_truth_i = np.where(ground_truth_c > 0)[0]

    true_positives = np.intersect1d(detected, ground_truth_i)
    false_positives = np.setdiff1d(detected, ground_truth_i)
    false_negatives = np.setdiff1d(ground_truth_i, detected)

    precision_denominator = len(true_positives) + len(false_positives)
    recall_denominator = len(true_positives) + len(false_negatives)

    precision = len(true_positives) / precision_denominator if precision_denominator > 0 else 0
    recall = len(true_positives) / recall_denominator if recall_denominator > 0 else 0

    # Calculate F1 score with a check to avoid division by zero
    f1_score_denominator = precision + recall
    f1_score = 2 * (precision * recall) / f1_score_denominator if f1_score_denominator > 0 else 0

    # Baseline F1 score
    baseline_f1 = F1_baseline(ground_truth_i)

    if plot:

        # Plot the signal
        fig, ax = plt.subplots(figsize=(8, 2))
        ax2 = ax.twinx()
        
        # Plot the trajectory of the signal
        ax.plot(signal, c='yellow', label=r"$s^{(i)}(\tau)$", zorder=1, linewidth=2)
        
        # Plot the ground truth
        ax2.plot(ground_truth_c, c='orangered', label=r"$m_t(\tau)$", zorder=1, linewidth=2)

        # Create a color scale using the derivative
        im = ax.imshow(derivative.reshape(1,-1), 
                       cmap='viridis', aspect='auto', alpha=1, zorder=0,
                       extent=[0, derivative.shape[0], ax.get_ylim()[0], ax.get_ylim()[1]])

        # Add detected points
        det = np.zeros_like(ground_truth_c)
        det[detected] = 1
        ax.plot(np.where(det, signal, np.nan),
                '*', markerfacecolor='yellow', markeredgecolor='deeppink', markersize=10,
                label='Detected')

        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'$w_t^{(i)}(\tau)$')

        # Add labels and legend
        ax.set_xlabel(r'Time $\tau$')

        # Combine handles and labels from both axes
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles2 += handles
        labels2 += labels

        # Add a single legend
        ax2.legend(handles2, labels2, loc='upper left')

        # Hide the secondary y-axis
        ax2.yaxis.set_visible(False)

        # Adjust layout
        plt.tight_layout()
        
        # Move the colorbar out of the plot
        cbar.ax.set_position([1.001, 0.275, 0.05, 0.65])

        return fig, precision, recall, f1_score, baseline_f1

    return None, precision, recall, f1_score, baseline_f1


def F1_baseline(ground_truth):
    """
    Calculate the F1 score of a baseline that always predicts True.

    Args:
        ground_truth (np.ndarray): The ground-truth signal.
    
    Returns:
        float: The F1 score of the baseline.
    """

    detected = np.ones_like(ground_truth)
    true_positives = np.intersect1d(detected, ground_truth)
    false_positives = np.setdiff1d(detected, ground_truth)
    false_negatives = np.setdiff1d(ground_truth, detected)

    precision_denominator = len(true_positives) + len(false_positives)
    recall_denominator = len(true_positives) + len(false_negatives)

    precision = len(true_positives) / precision_denominator if precision_denominator > 0 else 0
    recall = len(true_positives) / recall_denominator if recall_denominator > 0 else 0

    # Calculate F1 score with a check to avoid division by zero
    f1_score_denominator = precision + recall
    f1_score = 2 * (precision * recall) / f1_score_denominator if f1_score_denominator > 0 else 0

    return f1_score
    

def cross_correlation(signal, ground_truth, plot=False):
    """
    Compute the derivative of your signal.
    Compute the cross-correlation between the binary ground-truth signal and the derivative of the signal.
    High cross-correlation values at or near zero lag indicate correspondence.

    Args:
        signal (np.ndarray): The time signal.
        ground_truth (np.ndarray): The ground-truth signal.
        plot (bool, optional): Whether to plot the cross-correlation. Defaults to False.

    Returns:
        max_corr_lag_error: The absolute value of the lag with the highest cross-correlation, 
            i.e. how far the actual peak is from the desired lag of 0.
        corr_at_lag_0: The correlation at lag 0.
        figure: The plot of the signal and cross-correlation.
    """

    derivative = np.gradient(signal)

    # Cut ground truth to the same length as the derivative
    ground_truth = ground_truth[:len(derivative)]

    # Compute cross-correlation
    cross_correlation = np.correlate(derivative, ground_truth, mode='same')

    lags = np.arange(len(cross_correlation))
    
    # Identify the lag with the highest cross-correlation value
    max_corr_lag = lags[np.argmax(cross_correlation)]
    max_corr_lag_error = np.abs(max_corr_lag)
    
    # Compute correlation at lag 0
    corr_at_lag_0 = np.corrcoef(derivative, ground_truth)[0, 1]

    if plot:

        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle('Cross-correlation Analysis')

        axs[0].plot(signal, label='Signal')
        axs[0].plot(derivative, label='Derivative')
        axs[0].plot(ground_truth, label='Ground Truth')
        axs[0].set_xlabel('Time')
        axs[0].legend()

        axs[1].plot(lags, cross_correlation, label='Cross-correlation')
        axs[1].axvline(max_corr_lag, color='r', linestyle='--', label='Max Corr Lag')
        axs[1].set_xlabel('Lag')
        axs[1].legend()

        axs[1].text(0, -0.3, f"Correlation at lag 0: {corr_at_lag_0}", transform=axs[1].transAxes)

        return fig, max_corr_lag_error, corr_at_lag_0

    return None, max_corr_lag_error, corr_at_lag_0


def mann_whitney_test(signal, ground_truth, window_size=5, plot=False):
    """
    Statistical test to check if the derivative values around ground-truth instants are significantly different
    from those around random instants. This is done using the Mann-Whitney U test.

    Args:
        signal (np.ndarray): The time signal.
        ground_truth (np.ndarray): The ground-truth signal.
        window_size (int, optional): The window size around each instant to consider. Defaults to 5.
        plot (bool, optional): Whether to plot the Mann-Whitney U test. Defaults to False.

    Returns:
        MW_U_test_p_value: The p-value of the Mann-Whitney U test.
        figure: The histogram of derivative values around ground-truth and random instants.
    """

    # Ground-truth instants
    ground_truth_instants = np.where(ground_truth > 0)[0]

    # Compute the derivative of the signal
    derivative = np.diff(signal)

    # Define a window size around each instant
    half_window = window_size // 2

    # Extract derivative values around ground-truth instants
    gt_derivative_values = []
    for gt in ground_truth_instants:
        window_start = max(0, gt - half_window)
        window_end = min(len(derivative), gt + half_window)
        gt_derivative_values.extend(np.abs(derivative[window_start:window_end]))

    # Extract derivative values around random instants
    # Choose a number of random instants, here we choose the same number as ground-truth instants for consistency
    num_random_instants = len(ground_truth_instants)
    # Remove instants already used for ground-truth
    eligible_indices = list(set(range(len(derivative))) - set(gt_derivative_values))
    random_instants = np.random.choice(eligible_indices, num_random_instants, replace=False)
    random_derivative_values = []
    for ri in random_instants:
        window_start = max(0, ri - half_window)
        window_end = min(len(derivative), ri + half_window)
        random_derivative_values.extend(np.abs(derivative[window_start:window_end]))

    # Perform Mann-Whitney U test
    stat, MW_U_test_p_value = mannwhitneyu(gt_derivative_values, random_derivative_values, alternative='greater')

    if plot:

        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Mann-Whitney U Test')

        axs.hist(gt_derivative_values, bins=20, alpha=0.5, label='Ground Truth')
        axs.hist(random_derivative_values, bins=20, alpha=0.5, label='Random')
        axs.set_xlabel('Derivative Values')
        axs.legend()

        axs.text(0.6, 0.8, f"Mann-Whitney p-value: {MW_U_test_p_value:.2f}", transform=axs.transAxes)

        return fig, MW_U_test_p_value

    # If p-value is less than 0.05, we reject the null hypothesis that the two distributions are the same    
    return None, MW_U_test_p_value


def mann_whitney_test_dataset(signal, ground_truth, window_size=5, plot=False):
    """
    Statistical test to check if the derivative values around ground-truth instants are significantly different
    from those around random instants. This is done using the Mann-Whitney U test.

    Args:
        signal (np.ndarray): The time signals, shape (b, t, f).
        ground_truth (np.ndarray): The ground-truth signals, shape (b, t).
        window_size (int, optional): The window size around each instant to consider. Defaults to 5.
        plot (bool, optional): Whether to plot the Mann-Whitney U test. Defaults to False.

    Returns:
        MW_U_test_p_value: The p-value of the Mann-Whitney U test.
        figure: The histogram of derivative values around ground-truth and random instants.
    """

    # Ground-truth instants
    ground_truth_instants = np.argwhere(ground_truth)

    # Compute derivative and stack features
    derivative = np.gradient(signal, axis=1)

    # Define a window size around each instant
    half_window = window_size // 2

    # Extract derivative values around ground-truth instants
    gt_derivative_values = []
    for gt in ground_truth_instants:
        b, t = gt
        window_start = max(0, t - half_window)
        window_end = min(derivative.shape[1], t + half_window)
        gt_derivative_values.extend(np.abs(derivative[b,window_start:window_end]))

    # Extract derivative values around random instants
    # Choose a number of random instants, here we choose the same number as ground-truth instants for consistency
    num_random_instants = len(ground_truth_instants)
    # Remove instants already used for ground-truth
    all_indices = np.argwhere(ground_truth >= 0)
    excl_indices = {tuple(gt) for gt in ground_truth_instants}
    excl_indices = set(excl_indices)
    eligible_indices = [(b,t) for (b,t) in all_indices if (b,t) not in excl_indices]
    random_instants = random.sample(eligible_indices, num_random_instants)
    random_derivative_values = []
    for ri in random_instants:
        b, t = ri
        window_start = max(0, t - half_window)
        window_end = min(derivative.shape[1], t + half_window)
        random_derivative_values.extend(np.abs(derivative[b,window_start:window_end]))

    # Perform Mann-Whitney U test
    stat, MW_U_test_p_value = mannwhitneyu(gt_derivative_values, random_derivative_values, alternative='greater')

    if plot:

        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Mann-Whitney U Test')

        axs.hist(gt_derivative_values, bins=20, alpha=0.5, label='Ground Truth')
        axs.hist(random_derivative_values, bins=20, alpha=0.5, label='Random')
        axs.set_xlabel('Derivative Values')
        axs.legend()

        axs.text(0.6, 0.8, f"Mann-Whitney p-value: {MW_U_test_p_value:.2f}", transform=axs.transAxes)

        return fig, MW_U_test_p_value

    # If p-value is less than 0.05, we reject the null hypothesis that the two distributions are the same    
    return None, MW_U_test_p_value


def ml_probes(signal, ground_truth, seed=42, verbose=False):
    """
    Train a classifier to detect changes in the derivative of the signal 
    that correspond to ground-truth events.
    Use features derived from the derivative signal (e.g., derivative) 
    in a time window around each ground-truth instant.

    Args:
        signal (np.ndarray): The time signal.
        ground_truth (np.ndarray): The ground-truth signal.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        verbose (bool, optional): Whether to print classification reports. Defaults to False.

    Returns:
        dict: ROC AUC scores for Random Forest and Logistic Regression classifiers.
    """
    
    # Extract features and labels
    window_size = 5

    # Compute derivative and stack features
    derivative = np.gradient(signal, axis=1)
    signal = signal[:,:derivative.shape[1]]  # Match the length of the derivative

    features = np.concatenate((signal, derivative), axis=-1)

    def extract_features(features, idx, window_size):
        start = idx
        end = idx + window_size
        
        return features[:,start:end,:]

    inputs = []
    for idx in range(features.shape[1] - window_size + 1):
        inputs.append(extract_features(features, idx, window_size))

    inputs = np.array(inputs)
    labels = ground_truth[:,:inputs.shape[0]]  # Match the length of features
    labels = np.heaviside(labels, 0)

    inputs = rearrange(inputs, 'b t w f -> (b t) (w f)')
    labels = rearrange(labels, 'b t -> (b t)')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.3, random_state=seed)

    # Train a Random Forest classifier
    random_forest_results = random_forest(X_train, X_test, y_train, y_test, verbose, seed)

    # Train a Logistic Regression classifier
    log_regr_results = logistic_regression(X_train, X_test, y_train, y_test, verbose, seed)

    return {**random_forest_results, **log_regr_results}


def random_forest(X_train, X_test, y_train, y_test, verbose, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score

    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Evaluate the model
    if verbose:
        print(classification_report(y_test, y_pred))

    random_forest_roc_auc = roc_auc_score(y_test.squeeze(), y_pred_proba)

    return {'random_forest_roc_auc': random_forest_roc_auc}


def logistic_regression(X_train, X_test, y_train, y_test, verbose, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, roc_auc_score

    # Train a Logistic Regression classifier
    clf = LogisticRegression(random_state=seed, max_iter=10000)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Evaluate the model
    if verbose:
        print(classification_report(y_test, y_pred))
    
    log_regr_roc_auc = roc_auc_score(y_test.squeeze(), y_pred_proba)
    
    return {'log_regr_roc_auc': log_regr_roc_auc}


def auc_analysis_edges(weights, edge_index, edge_gt, num_nodes, plot=False):
    """
    Takes the edge weights computed via SINDy and 
    computes the ROC AUC score between the edge weights and GT.

    Args:
        weights (torch.Tensor): Edge weights.
        edge_index (torch.Tensor): Graph topology.
        edge_gt (np.ndarray): The ground-truth edge index.
        num_nodes (int): The number of nodes in the graph.
        plot (bool): Whether to plot the graph and ground truth.
    
    Returns:
        auc_score: The ROC AUC score.
        figure: The graph and ground truth plots.
    """

    from sklearn.metrics import roc_auc_score
    import networkx as nx
    import torch
    
    # Compute the AUC score
    auc_score = roc_auc_score(edge_gt, weights)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Graph AUC Analysis: {auc_score:.2f}')
        G = nx.DiGraph()

        # Add nodes
        G.add_nodes_from(range(num_nodes))
        n_labels = {i: str(i) for i in range(num_nodes)}

        # Add edges
        edge_index = torch.cat(edge_index, dim=1)
        edge_index = torch.unique(edge_index.T, dim=0).T
        G.add_edges_from(edge_index.T.tolist())

        # Plot the graph
        pos = nx.kamada_kawai_layout(G)
        cmap = matplotlib.colormaps.get_cmap('viridis')
        pax = nx.draw_networkx_nodes(G, pos, ax=axs[0], node_color='lightblue', node_size=200)
        norm = plt.Normalize(min(weights), max(weights))
        colors = [cmap(norm(w)) for w in weights]
        nx.draw_networkx_edges(G, pos, ax=axs[0], edge_color=colors, width=4)
        n_labels = {i: str(i) for i in range(num_nodes)}
        nx.draw_networkx_labels(G, pos, n_labels, ax=axs[0], font_size=10,font_color='r')
        axs[0].set_title('Graph with weights')

        # Plot the ground truth
        G_gt = nx.DiGraph()
        G_gt.add_nodes_from(range(num_nodes))
        G_gt.add_edges_from(edge_index.T.tolist())
        nx.draw_networkx_nodes(G_gt, pos, ax=axs[1], node_color='lightblue', node_size=200)
        colors = ['r' if edge else 'lightblue' for edge in edge_gt.tolist()]
        nx.draw_networkx_edges(G_gt, pos, ax=axs[1], edge_color=colors, width=4)
        axs[1].set_title('Ground Truth Graph')

        plt.colorbar(pax, ax=axs[0])

        return fig, auc_score
    
    return None, auc_score


def auc_analysis_nodes(weights, node_gt, edge_index, plot=False):
    """
    Takes the node weights computed with DMD analysis and 
    computes the ROC AUC score between the node weights and GT.

    Args:
        weights (torch.Tensor): Node weights.
        node_gt (np.ndarray): The ground-truth for nodes.
        edge_index (torch.Tensor): Graph topology.
        plot (bool): Whether to plot the graph and ground truth.
    
    Returns:
        auc_score: The ROC AUC score.
        figure: The graph and ground truth plots.
    """

    from sklearn.metrics import roc_auc_score
    import networkx as nx
    import torch

    # Compute the AUC score
    auc_score = roc_auc_score(node_gt, weights)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Graph AUC Analysis: {auc_score:.2f}')
        G = nx.DiGraph()

        # Add edges
        edge_index = torch.cat(edge_index, dim=1)
        edge_index = torch.unique(edge_index.T, dim=0).T
        G.add_edges_from(edge_index.T.tolist())

        # Add nodes
        num_nodes = edge_index.max().item() + 1
        G.add_nodes_from(range(num_nodes))
        n_labels = {i: str(i) for i in range(num_nodes)}

        # Plot the graph
        pos = nx.kamada_kawai_layout(G)
        cmap = matplotlib.colormaps.get_cmap('viridis')
        norm = plt.Normalize(min(weights), max(weights))
        colors = [cmap(norm(w)) for w in weights]
        pax = nx.draw_networkx_nodes(G, pos, ax=axs[0], node_color=colors, node_size=200)
        nx.draw_networkx_edges(G, pos, ax=axs[0], edge_color='lightblue', width=4)
        n_labels = {i: str(i) for i in range(num_nodes)}
        nx.draw_networkx_labels(G, pos, n_labels, ax=axs[0], font_size=10,font_color='r')
        axs[0].set_title('Graph with weights')

        # Plot the ground truth % FIXME: Check if this is correct
        G_gt = nx.DiGraph()
        G_gt.add_nodes_from(range(num_nodes))
        G_gt.add_edges_from(edge_index.T.tolist())
        colors = ['r' if node else 'lightblue' for node in node_gt.tolist()]
        nx.draw_networkx_nodes(G_gt, pos, ax=axs[1], node_color=colors, node_size=200)
        nx.draw_networkx_edges(G_gt, pos, ax=axs[1], edge_color='lightblue', width=4)
        axs[1].set_title('Ground Truth Graph')

        plt.colorbar(pax, ax=axs[0])

        return fig, auc_score
    
    return None, auc_score
