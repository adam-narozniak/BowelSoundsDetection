from keras import backend as K


def _recall_m(y_true, y_pred):
    """Computes recall (uses Keras backend's epsilon)."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def _precision_m(y_true, y_pred):
    """Computes precision (uses Keras backend's epsilon)."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    """Computes f1 score based using precision and recall (uses Keras backend's epsilon)."""
    precision = _precision_m(y_true, y_pred)
    recall = _recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_mean_scores(scores: list[dict]):
    """Calculates mean metrics from list of metrics from cross validation.

    F1 mean score is calculated by adding TP, FP, FN, TN from each fold instead of
    averaging that may lead to wrong results when the classes are imbalanced.

    Args:
        scores: List of metrics.

    Returns:
        dict: Dictionary with averaged metrics.
    """
    mean_score = {}
    for key in scores[0].keys():
        if key != "f1":
            mean_score[key] = sum(d[key] for d in scores) / len(scores)
    mean_score["f1"] = 2 * mean_score["TP"] / (2 * mean_score["TP"] + mean_score["FP"] + mean_score["FN"])
    return mean_score
