import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from einops import rearrange
import matplotlib.pyplot as plt


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
        dict: precision, recall, F1-score.
        figure: The plot of the signal and derivative.
    """

    derivative = np.gradient(signal)
    
    if threshold is None:
        threshold = np.mean(derivative) + np.std(derivative)
    else:
        threshold = threshold * np.max(derivative)
    
    detected = np.where(np.abs(derivative) > threshold)[0]

    filter = np.ones(window_size)
    ground_truth = np.convolve(ground_truth, filter, mode='same')
    ground_truth = np.where(ground_truth > 0)[0]

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

    result = {
        'thr_precision': precision,
        'thr_recall': recall,
        'thr_f1_score': f1_score
    }

    if plot:

        # Plot the signal
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle('Threshold-based Detection')

        axs[0].plot(signal, label='Signal')
        axs[0].plot(ground_truth, label='Ground Truth')
        axs[0].plot(np.where(np.abs(derivative) > threshold, signal, np.nan), 'r.', label='Detected')
        axs[0].set_xlabel('Time')
        axs[0].legend()

        axs[1].plot(derivative, label='Derivative')
        axs[1].plot(ground_truth, label='Ground Truth')
        axs[1].plot(np.where(np.abs(derivative) > threshold, derivative, np.nan), 'r.', label='Detected')
        axs[1].set_xlabel('Time')
        axs[1].legend()

        # Print the results
        axs[1].text(0, -0.3, f"Precision: {result['thr_precision']:.2f}", transform=axs[1].transAxes)
        axs[1].text(0.4, -0.3, f"Recall: {result['thr_recall']:.2f}", transform=axs[1].transAxes)
        axs[1].text(0.8, -0.3, f"F1-score: {result['thr_f1_score']:.2f}", transform=axs[1].transAxes)

        return fig, result

    return result
    

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
        dict: precision, recall, F1-score.
        figure: The plot of the signal and derivative.
    """

    # Moving average
    signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    derivative = np.gradient(signal)
    
    if threshold is None:
        threshold = np.mean(derivative) + np.std(derivative)
    else:
        threshold = threshold * np.max(derivative)
    
    detected = np.where(np.abs(derivative) > threshold)[0]
    
    filter = np.ones(window_size)
    ground_truth = np.convolve(ground_truth, filter, mode='same')
    ground_truth = np.where(ground_truth > 0)[0]

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

    result = {
        'window_precision': precision,
        'window_recall': recall,
        'window_f1_score': f1_score
    }

    if plot:

        # Plot the signal
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle('Windowing Analysis')

        axs[0].plot(signal, label='Signal')
        axs[0].plot(ground_truth, label='Ground Truth')
        axs[0].plot(np.where(np.abs(derivative) > threshold, signal, np.nan), 'r.', label='Detected')
        axs[0].set_xlabel('Time')
        axs[0].legend()

        axs[1].plot(derivative, label='Derivative')
        axs[1].plot(ground_truth, label='Ground Truth')
        axs[1].plot(np.where(np.abs(derivative) > threshold, derivative, np.nan), 'r.', label='Detected')
        axs[1].set_xlabel('Time')
        axs[1].legend()

        # Print the results
        axs[1].text(0, -0.3, f"Precision: {result['window_precision']:.2f}", transform=axs[1].transAxes)
        axs[1].text(0.4, -0.3, f"Recall: {result['window_recall']:.2f}", transform=axs[1].transAxes)
        axs[1].text(0.8, -0.3, f"F1-score: {result['window_f1_score']:.2f}", transform=axs[1].transAxes)

        return fig, result

    return result
    

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
        dict: The absolute value of the lag with the highest cross-correlation, 
                i.e. how far the actual peak is from the desired lag of 0.
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

    if plot:

        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle('Cross-correlation Analysis')

        axs[0].plot(signal, label='Signal')
        axs[0].plot(ground_truth, label='Ground Truth')
        axs[0].set_xlabel('Time')
        axs[0].legend()

        axs[1].plot(lags, cross_correlation, label='Cross-correlation')
        axs[1].axvline(max_corr_lag, color='r', linestyle='--', label='Max Corr Lag')
        axs[1].set_xlabel('Lag')
        axs[1].legend()

        axs[1].text(0, -0.3, f"Max Corr Lag Error: {max_corr_lag_error}", transform=axs[1].transAxes)

        return fig, {'max_corr_lag_error': max_corr_lag_error}

    return {'max_corr_lag_error': max_corr_lag_error}


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
        dict: The p-value of the Mann-Whitney U test.
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

        return fig, {'mw_p_value': MW_U_test_p_value}

    # If p-value is less than 0.05, we reject the null hypothesis that the two distributions are the same    
    return {'mw_p_value': MW_U_test_p_value}


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

