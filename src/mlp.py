"""Stage 4.4: a small PyTorch multilayer perceptron with a scikit-learn-like interface.

Why not sklearn's MLPClassifier/MLPRegressor (rule 13): they offer no class weights, only squared loss for regression,
and their early stopping uses a *random* validation fraction, not our chronological slice.

The estimator fits its own preprocessing (one-hot month + StandardScaler, see models.linear_pre) on the training rows
passed to `fit`, so scaling never sees validation data. Regression targets are divided by their training std.
"""
import numpy as np
import torch
from torch import nn

from models import linear_pre

LOSSES = {"bce": None, "mae": nn.L1Loss, "huber": nn.HuberLoss, "mse": nn.MSELoss}


class TorchMLP:
    def __init__(self, task="cls", loss="bce", hidden=(64, 64), dropout=0.1, weight_decay=1e-4, lr=1e-3,
                 batch_size=256, max_epochs=150, patience=15, class_weight="balanced", seed=0, threads=4):
        self.task, self.loss, self.hidden, self.dropout = task, loss, hidden, dropout
        self.weight_decay, self.lr, self.batch_size = weight_decay, lr, batch_size
        self.max_epochs, self.patience, self.class_weight, self.seed, self.threads = max_epochs, patience, class_weight, seed, threads

    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    def _net(self, d):
        layers, width = [], d
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(self.dropout)]
            width = h
        return nn.Sequential(*layers, nn.Linear(width, 1))

    def _criterion(self, y):
        if self.task == "cls":
            pos_w = (y == 0).sum() / max((y == 1).sum(), 1) if self.class_weight == "balanced" else 1.0
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_w)))
        return LOSSES[self.loss]()

    def _tensor(self, X):
        return torch.tensor(self.pre_.transform(X), dtype=torch.float32)

    def fit(self, X, y, eval_set=None):
        """Train; with eval_set=(Xv, yv) keep the weights of the best epoch on that set (early stopping)."""
        torch.manual_seed(self.seed); np.random.seed(self.seed); torch.set_num_threads(self.threads)
        y = np.asarray(y, dtype=np.float32)
        self.pre_ = linear_pre().fit(X)
        self.y_scale_ = 1.0 if self.task == "cls" else float(y.std())
        Xt, yt = self._tensor(X), torch.tensor(y / self.y_scale_).unsqueeze(1)
        if eval_set is not None:
            Xv, yv = self._tensor(eval_set[0]), torch.tensor(np.asarray(eval_set[1], np.float32) / self.y_scale_).unsqueeze(1)
        self.net_ = self._net(Xt.shape[1])
        crit, opt = self._criterion(y), torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.seed)
        self.history_, best, best_state, wait = {"train": [], "val": []}, np.inf, None, 0
        self.best_epoch_ = self.max_epochs
        for epoch in range(self.max_epochs):
            self.net_.train()
            for idx in torch.randperm(len(Xt), generator=gen).split(self.batch_size):
                opt.zero_grad()
                crit(self.net_(Xt[idx]), yt[idx]).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                self.history_["train"].append(crit(self.net_(Xt), yt).item())
                if eval_set is not None:
                    v = crit(self.net_(Xv), yv).item()
                    self.history_["val"].append(v)
                    if v < best - 1e-5:
                        best, wait, self.best_epoch_ = v, 0, epoch + 1
                        best_state = {k: t.clone() for k, t in self.net_.state_dict().items()}
                    else:
                        wait += 1
                        if self.patience and wait >= self.patience:
                            break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def _raw(self, X):
        self.net_.eval()
        with torch.no_grad():
            return self.net_(self._tensor(X)).squeeze(1).numpy()

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self._raw(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self._raw(X) * self.y_scale_ if self.task == "reg" else (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
