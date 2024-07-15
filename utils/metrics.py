import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split


def threshold_based_detection(signal, ground_truth, threshold=None):
    """
    Define a threshold that captures significant changes in the derivative.
    Compare the time instants where the derivative exceeds this threshold with the ground-truth instants.
    Use metrics such as precision, recall, and F1-score to quantify the correspondence.

    Args:
        signal (np.ndarray): The time signal.
        ground_truth (np.ndarray): The ground-truth signal.
        threshold (float): The threshold to apply, as percentage of the maximum derivative.
                            If `None`, it will be computed as mean+std of the derivative.

    Returns:
        dict: precision, recall, F1-score.
    """

    derivative = np.gradient(signal)
    
    if threshold is None:
        threshold = np.mean(derivative) + np.std(derivative)
    
    detected = np.where(np.abs(derivative) > threshold)[0]

    true_positives = np.intersect1d(detected, ground_truth)
    false_positives = np.setdiff1d(detected, ground_truth)
    false_negatives = np.setdiff1d(ground_truth, detected)

    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    recall = len(true_positives) / (len(true_positives) + len(false_negatives))
    f1_score = 2 * (precision * recall) / (precision + recall)

    result = {
        'thr_precision': precision,
        'thr_recall': recall,
        'thr_f1_score': f1_score
    }

    return result
    

def windowing_analysis(signal, ground_truth, window_size=10, threshold=None):
    """
    Define a threshold that captures significant changes in the derivative of the
    moving average of the signal.
    Compare the time instants where the derivative exceeds this threshold with the ground-truth instants.
    Use metrics such as precision, recall, and F1-score to quantify the correspondence.

    Args:
        signal (np.ndarray): The time signal.
        ground_truth (np.ndarray): The ground-truth signal.
        threshold (float): The threshold to apply, as percentage of the maximum derivative.
                            If `None`, it will be computed as mean+std of the derivative.

    Returns:
        dict: precision, recall, F1-score.
    """

    # Moving average
    signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    derivative = np.gradient(signal)
    
    if threshold is None:
        threshold = np.mean(derivative) + np.std(derivative)
    
    detected = np.where(np.abs(derivative) > threshold)[0]

    true_positives = np.intersect1d(detected, ground_truth)
    false_positives = np.setdiff1d(detected, ground_truth)
    false_negatives = np.setdiff1d(ground_truth, detected)

    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    recall = len(true_positives) / (len(true_positives) + len(false_negatives))
    f1_score = 2 * (precision * recall) / (precision + recall)

    result = {
        'window_precision': precision,
        'window_recall': recall,
        'window_f1_score': f1_score
    }

    return result
    

def cross_correlation(signal, ground_truth):
    """
    Compute the derivative of your signal.
    Compute the cross-correlation between the binary ground-truth signal and the derivative of the signal.
    High cross-correlation values at or near zero lag indicate correspondence.

    Args:
        signal (np.ndarray): The time signal.
        ground_truth (np.ndarray): The ground-truth signal.

    Returns:
        dict: The absolute value of the lag with the highest cross-correlation, 
                i.e. how far the actual peak is from the desired lag of 0.
    """

    derivative = np.gradient(signal)

    # Cut ground truth to the same length as the derivative
    ground_truth = ground_truth[:len(derivative)]

    # Compute cross-correlation
    cross_correlation = np.correlate(derivative, ground_truth, mode='same')

    # The full cross-correlation array length is (2N-1) where N is the length of the derivative
    lags = np.arange(-len(derivative) + 1, len(derivative))
    
    # Identify the lag with the highest cross-correlation value
    max_corr_lag = lags[np.argmax(cross_correlation)]
    max_corr_lag_error = np.abs(max_corr_lag)

    return {'max_corr_lag_error': max_corr_lag_error}


def mann_whitney_test(signal, ground_truth, window_size=5):
    """
    Statistical test to check if the derivative values around ground-truth instants are significantly different
    from those around random instants. This is done using the Mann-Whitney U test.

    Args:
        signal (np.ndarray): The time signal.
        ground_truth (np.ndarray): The ground-truth signal.
        window_size (int, optional): The window size around each instant to consider. Defaults to 5.

    Returns:
        dict: The p-value of the Mann-Whitney U test.
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

    # If p-value is less than 0.05, we reject the null hypothesis that the two distributions are the same    
    return {'p_value': MW_U_test_p_value}


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
    half_window = window_size // 2

    # Compute derivative and stack features
    derivative = np.gradient(signal)
    signal = signal[:len(derivative)]  # Match the length of the derivative

    features = np.stack((signal, derivative), axis=-1)

    def extract_features(features, idx, window_size):
        half_window = window_size // 2
        start = max(0, idx - half_window)
        end = min(len(signal), idx + half_window + 1)
        
        return features[start:end]

    inputs = []
    for idx in range(features.shape[0]):
        inputs.append(extract_features(features, idx, window_size))

    inputs = np.array(inputs)
    labels = ground_truth[:len(derivative)]  # Match the length of features

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=seed)

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

    random_forest_roc_auc = roc_auc_score(y_test, y_pred_proba)

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
    
    log_regr_roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return {'log_regr_roc_auc': log_regr_roc_auc}

