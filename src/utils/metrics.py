from sklearn.metrics import precision_score, recall_score


def compute_precision(pred, real):
    pred = set(pred)
    real = set(real)
    return len(pred & real) / (len(pred) + 1e-10)


def compute_recall(pred, real):
    pred = set(pred)
    real = set(real)
    return len(pred & real) / (len(real) + 1e-10)



