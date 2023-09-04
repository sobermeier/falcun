from copy import deepcopy

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        self.sampling_info = []
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.idxs_additive_lb = idxs_lb
        self.net_args = net["net_args"]
        self.weight_reset_type = net["weight_reset_type"]
        if self.weight_reset_type in ["no", "additive", "weight_reset"]:
            self.net = net["net"](**self.net_args)
        else:
            self.net = net["net"]
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.class_idxs = np.unique(Y)
        self.current_round, self.total_rounds = 0, 0
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.out_dir = None
        self.patience, self.frequency = 10, 5

    def query(self, n):
        pass

    def update(self, idxs_lb, new_idx):
        self.idxs_lb = idxs_lb
        if self.weight_reset_type == "additive":
            self.idxs_additive_lb = np.zeros(len(idxs_lb), dtype=bool)
            self.idxs_additive_lb[new_idx] = True

    def save_stats(self, df):
        file_name = f"{self.current_round}_statistics.csv"
        if self.out_dir is not None:
            df.to_csv(self.out_dir / file_name)
        else:
            print("did not save, no out_dir specified.")

    def set_current_round(self, iteration):
        self.current_round = iteration

    def set_total_rounds(self, total_rounds):
        self.total_rounds = total_rounds

    def set_path(self, out_dir):
        """ create and set output directory for current seed and experiment run. """
        if out_dir.is_dir():
            print(f'Output path already exists! {out_dir}')
        out_dir.mkdir(exist_ok=True, parents=True)
        self.out_dir = out_dir

    def _train(self, loader_tr, optimizer):
        self.clf.train()
        train_accuracy = 0.
        train_loss = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            train_accuracy += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_accuracy /= len(loader_tr.dataset.X)
        train_loss /= len(loader_tr.dataset.X)
        return train_accuracy, train_loss

    def train(
            self,
            early_stopping=True,
            lr=0.001,
            wd=0,
            optimizer="adam",
            momentum=0,
            epochs=100,
            lr_scheduler=False,
    ):
        def weight_reset_func(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        if self.weight_reset_type == "no" or self.weight_reset_type == "additive":
            self.clf = self.net.to(self.device)
        elif self.weight_reset_type == "weight_reset":
            self.clf = self.net.apply(weight_reset_func).to(self.device)
        else:  # = new_init
            self.clf = self.net(**self.net_args).to(self.device)

        if optimizer == "adam":
            _optimizer = optim.Adam(
                self.clf.parameters(),
                lr=lr,
                weight_decay=wd
            )
        elif optimizer == "sgd":
            _optimizer = optim.SGD(
                self.clf.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=wd
            )
        else:
            raise NotImplementedError

        if lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max=200)
        elif lr_scheduler == "multistep":
            milestones = [0.5 * epochs, 0.75 * epochs]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                _optimizer, milestones=milestones, gamma=0.1
            )
        else:
            scheduler = None

        if self.weight_reset_type == "additive":
            idxs_train = np.arange(self.n_pool)[self.idxs_additive_lb]
        else:
            idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        loader_tr = DataLoader(
            self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['ds_params']['transform']),
            shuffle=True,
            **self.args['ds_params']['loader_tr_args']
        )
        loss_hist, acc_hist = [], []
        tr_loss, tr_acc = 0, 0
        for epoch in range(epochs):
            tr_acc, tr_loss = self._train(loader_tr, _optimizer)
            if scheduler:
                scheduler.step()
            loss_hist.append(tr_loss)
            acc_hist.append(tr_acc)
            print(f'Epoch {epoch:5}: {tr_loss:2.7f} (acc: {tr_acc})')

            if early_stopping and epoch > self.patience:
                if tr_acc >= 0.99:
                    print('Early Stopping.')
                    break
        return tr_acc, tr_loss

    def predict(self, X, Y):
        loader_te = DataLoader(
            self.handler(X, Y, transform=self.args["ds_params"]['transform']),
            shuffle=False,
            **self.args["ds_params"]['loader_te_args']
        )
        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = torch.max(out, dim=-1)[1]
                P[idxs] = pred.cpu()
        return P

    def predict_prob(self, X, Y):
        loader_te = DataLoader(
            self.handler(X, Y, transform=self.args["ds_params"]['transform']),
            shuffle=False,
            **self.args["ds_params"]['loader_te_args']
        )

        self.clf.eval()
        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()

        return probs

    def predict_logits(self, X, Y):
        loader_te = DataLoader(
            self.handler(X, Y, transform=self.args["ds_params"]['transform']),
            shuffle=False,
            **self.args["ds_params"]['loader_te_args']
        )

        self.clf.eval()
        logits = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                logits[idxs] = out.cpu()
        return logits

    def predict_prob_dropout_split(self, X, Y, n_drop):
        # n_drop inferences --> returns probs of shape (T, batch_size, classes)
        loader_te = DataLoader(
            self.handler(X, Y, transform=self.args["ds_params"]['transform']),
            shuffle=False,
            **self.args["ds_params"]['loader_te_args']
        )

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        assert not (probs[0] == probs[1]).all()
        return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(
            self.handler(X, Y, transform=self.args["ds_params"]['transform']),
            shuffle=False,
            **self.args["ds_params"]['loader_te_args']
        )

        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()
        return embedding

    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y):
        model = self.clf
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(
            self.handler(X, Y, transform=self.args["ds_params"]['transform']),
            shuffle=False,
            **self.args["ds_params"]['loader_te_args']
        )
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)  # for each entry get idx of maximum label
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (
                                    -1 * batchProbs[j][c])
            return torch.Tensor(embedding)

    def get_diversity_embeddings(self, emb_type: str, x, y) -> np.ndarray:
        """
        get vectors on which diversity sampling is performed
        """
        out_dim = self.net_args["output_dim"]
        if emb_type == "latent":
            print("latent")
            embedding = self.get_embedding(x, y)
            return embedding.numpy()
        elif emb_type == "pca":
            print("pca")
            embedding = self.get_embedding(x, y).numpy()
            pca = PCA(n_components=out_dim)
            pca.fit(embedding)
            return pca.transform(embedding)
        elif emb_type == "output":
            print("output")
            embedding = self.predict_prob(x, y)
            return embedding.numpy()
        else:
            print("Embedding Type Not Found. Must be one of 'latent', 'pca', 'output'")
            raise NotImplementedError
