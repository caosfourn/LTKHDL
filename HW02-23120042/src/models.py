import numpy as np
from collections import Counter

class Metrics:
    
    @staticmethod
    def accuracy(y_true, y_pred):
        if len(y_true) == 0: return 0
        return np.sum(y_true == y_pred) / len(y_true)

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        # Tính TP, TN, FP, FN cho bài toán nhị phân
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp, tn, fp, fn

    @staticmethod
    def precision(y_true, y_pred):
        tp, tn, fp, fn = Metrics.confusion_matrix(y_true, y_pred)
        if (tp + fp) == 0: return 0
        return tp / (tp + fp)

    @staticmethod
    def recall(y_true, y_pred):
        tp, tn, fp, fn = Metrics.confusion_matrix(y_true, y_pred)
        if (tp + fn) == 0: return 0
        return tp / (tp + fn)

    @staticmethod
    def f1_score(y_true, y_pred):
        p = Metrics.precision(y_true, y_pred)
        r = Metrics.recall(y_true, y_pred)
        if (p + r) == 0: return 0
        return 2 * (p * r) / (p + r)

    # --- Regression Metrics ---

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def r2_score(y_true, y_pred):
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        if denominator == 0: return 0
        numerator = np.sum((y_true - y_pred) ** 2)
        return 1 - (numerator / denominator)


# Loss function

class Loss:
    class MSE:
        """Mean Squared Error cho Linear Regression"""
        @staticmethod
        def loss(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        @staticmethod
        def gradient(y_true, y_pred, X=None):
            """
            Sử dụng np.einsum để tối ưu hóa tính toán gradient
            """
            n = len(y_true)
            diff = y_pred - y_true 
            
            if X is None:
                return (2 / n) * diff
            else:
                # FIX: Dùng einsum thay cho np.dot theo yêu cầu Advanced Numpy
                # 'ji,j->i': X.T (ji) nhân với diff (j) -> weights (i)
                dw = (2 / n) * np.einsum('ji,j->i', X, diff)
                db = (2 / n) * np.sum(diff)
                return dw, db

    class BinaryCrossEntropy:
        """Log Loss cho Logistic Regression"""
        @staticmethod
        def loss(y_true, y_pred):
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        @staticmethod
        def gradient(y_true, y_pred):
            n = len(y_true)
            return (1 / n) * (y_pred - y_true)
        
# Cross-Validation
class ModelSelection:
    @staticmethod
    def k_fold_split(X, y, k=5, shuffle=True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
            
        fold_sizes = np.full(k, n_samples // k, dtype=int)
        fold_sizes[:n_samples % k] += 1
        current = 0
        
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, val_indices
            current = stop


class Optimizer:
    def update(self, w, b, dw, db):
        pass

class SGD(Optimizer):
    def __init__(self, lr=0.001):
        self.lr = lr

    def update(self, w, b, dw, db):
        w -= self.lr * dw
        b -= self.lr * db
        return w, b

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m_w, self.v_w = None, None 
        self.m_b, self.v_b = 0, 0       
        self.t = 0                      

    def update(self, w, b, dw, db):
        if self.m_w is None:
            self.m_w = np.zeros_like(w)
            self.v_w = np.zeros_like(w)
        
        self.t += 1
        
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)
        
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        
        w -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)
        
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        
        b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        
        return w, b


# LOGISTIC REGRESSION 

class LogisticRegression:
    def __init__(self, optimizer=None, n_iters=1000):
        self.n_iters = n_iters
        self.optimizer = optimizer if optimizer else SGD(lr=0.001)
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights, self.bias = self.optimizer.update(self.weights, self.bias, dw, db)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        y_predicted = self.predict_proba(X)
        # VECTORIZATION: Broadcasting & Casting
        return (y_predicted > threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return Metrics.accuracy(y, y_pred)


# K-NEAREST NEIGHBORS (KNN)

class KNN: 
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.astype(int) 

    def predict(self, X):
        # 1. Tính khoảng cách 
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        train_sq = np.sum(self.X_train**2, axis=1)

        dot_product = np.dot(X, self.X_train.T) 
        dists_sq = np.maximum(X_sq + train_sq - 2 * dot_product, 0)
        
        # 2. Lấy index k láng giềng
        k_indices = np.argpartition(dists_sq, self.k, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_indices] # Shape (N_samples, k)
        
        # 3. Voting (Vectorized implementation)
        def get_mode(row):
            return np.bincount(row).argmax()
            
        y_pred = np.apply_along_axis(get_mode, axis=1, arr=k_nearest_labels)
        
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return Metrics.accuracy(y, y_pred)


# DECISION TREE

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value 

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features_val = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
            
        # Chọn ngẫu nhiên features
        feat_idxs = np.random.choice(n_feats, self.n_features_val, replace=False)
        
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        if best_feat is None:
            return Node(value=self._most_common_label(y))
            
        left_mask, right_mask = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_mask, :], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask, :], y[right_mask], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            unique_values = np.unique(X_column)
            
            # Heuristic để tăng tốc độ nếu feature có quá nhiều giá trị unique
            if len(unique_values) > 10:
                thresholds = np.random.choice(unique_values, 10, replace=False)
            else:
                thresholds = unique_values
                
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        
        left_mask, right_mask = self._split(X_column, threshold)
        
        if not np.any(left_mask) or not np.any(right_mask): 
            return 0
            
        n = len(y)
        n_l, n_r = np.sum(left_mask), np.sum(right_mask)
        e_l, e_r = self._entropy(y[left_mask]), self._entropy(y[right_mask])
        
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        left_mask = X_column <= split_thresh
        return left_mask, ~left_mask

    def _entropy(self, y):
        hist = np.bincount(y.astype(int))
        ps = hist / len(y)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log(ps))

    def _most_common_label(self, y):
        counter = Counter(y)
        if not counter: return None
        return counter.most_common(1)[0][0]

    def predict(self, X):
        # VECTORIZATION: Batch prediction
        return self._traverse_tree_batch(X, self.root)

    def _traverse_tree_batch(self, X, node):
        if node.is_leaf_node():
            return np.full(X.shape[0], node.value)
        
        mask_left = X[:, node.feature] <= node.threshold
        mask_right = ~mask_left
        
        results = np.empty(X.shape[0], dtype=int)
        
        if np.any(mask_left):
            results[mask_left] = self._traverse_tree_batch(X[mask_left], node.left)
            
        if np.any(mask_right):
            results[mask_right] = self._traverse_tree_batch(X[mask_right], node.right)
            
        return results
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return Metrics.accuracy(y, y_pred)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features_total = X.shape
        
        if self.n_features is None:
            features_per_tree = int(np.sqrt(n_features_total))
        else:
            features_per_tree = self.n_features

        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=features_per_tree
            )
            
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        # List comprehension lặp qua SỐ CÂY 
        tree_preds = np.stack([tree.predict(X) for tree in self.trees])
        
        # Đảo trục về (n_samples, n_trees) để vote
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        def get_mode(row):
            return np.bincount(row.astype(int)).argmax()
            
        # VECTORIZATION: Apply function trên từng hàng mẫu
        y_pred = np.apply_along_axis(get_mode, axis=1, arr=tree_preds)
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return Metrics.accuracy(y, y_pred)