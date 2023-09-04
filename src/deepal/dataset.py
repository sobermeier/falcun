import os
import ssl
import urllib

import numpy as np
import scipy
import torch
import PIL
import openml
import torchvision
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image

from src.deepal.settings import DATA_ROOT, DIRTY_MNIST_CONFIG

DATASET_PATH = DATA_ROOT + '/data'


def _get_download_dir(downloads_path: str) -> str:
    """
    Helper function to define the path where the data files should be stored. If downloads_path is None then default path
    '[USER]/Downloads/clustpy_datafiles' will be used. If the directory does not exists it will be created.
    Parameters
    ----------
    downloads_path : str
        path to the directory where the data will be stored. Can be None
    Returns
    -------
    downloads_path : str
        path to the directory where the data will be stored. If input was None this will be equal to
        '[USER]/Downloads/clustpy_datafiles'
    """
    if downloads_path is None:
        downloads_path = DATA_ROOT
    if not os.path.isdir(downloads_path):
        os.makedirs(downloads_path)
        with open(downloads_path + "/info.txt", "w") as f:
            f.write("This directory was created by the ClustPy python package to store real world data sets.\n"
                    "The default directory is '[USER]/Downloads/clustpy_datafiles' and can be changed with the "
                    "'downloads_path' parameter when loading a data set.")
    return downloads_path


def _download_file(file_url: str, filename_local: str) -> None:
    """
    Helper function to download a file into a specified location.
    Parameters
    ----------
    file_url : str
        URL of the file
    filename_local : str
        local name of the file after it has been downloaded
    """
    print("Downloading data set from {0} to {1}".format(file_url, filename_local))
    default_ssl = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(file_url, filename_local)
    ssl._create_default_https_context = default_ssl


def _torch_flatten_shape(data: torch.Tensor, is_color_channel_last: bool, normalize_channels: bool):
    """
    Convert torch data tensor from image to numerical vector.
    Parameters
    ----------
    data : torch.Tensor
    is_color_channel_last : bool
        if true, the color channels should be in the last dimension, known as HWC representation. Alternatively the color channel can be at the first position, known as CHW representation.
        Only relevant for color images -> Should be None for grayscale images
    normalize_channels : bool
        normalize each color-channel of the images
    Returns
    -------
    The flatten data vector
    """
    # Flatten shape
    if data.dim() == 3:
        data = data.reshape(-1, data.shape[1] * data.shape[2])
    elif data.dim() == 4:
        # In case of 3d grayscale image is_color_channel_last is None
        if is_color_channel_last is not None and (not is_color_channel_last or normalize_channels):
            # Change representation to HWC
            data = data.permute(0, 2, 3, 1)
        assert is_color_channel_last is None or data.shape[3] == 3, "Colored image must consist of three channels not {0}".format(data.shape[3])
        data = data.reshape(-1, data.shape[1] * data.shape[2] * data.shape[3])
    elif data.dim() == 5:
        if not is_color_channel_last or normalize_channels:
            # Change representation to HWDC
            data = data.permute(0, 2, 3, 4, 1)
        assert data.shape[4] == 3, "Colored image must consist of three channels not {0}".format(data.shape[4])
        data = data.reshape(-1, data.shape[1] * data.shape[2] * data.shape[3] * data.shape[4])
    return data


def _torch_normalize_channels(data: torch.Tensor, is_color_channel_last: bool) -> torch.Tensor:
    """
    Normalize the color channels of a torch dataset
    Parameters
    ----------
    data : torch.Tensor
        The torch data tensor
    is_color_channel_last : bool
        if true, the color channels should be in the last dimension, known as HWC representation. Alternatively the color channel can be at the first position, known as CHW representation.
        Only relevant for color images -> Should be None for grayscale images
    Returns
    -------
    The normalized data tensor
    """
    if data.dim() == 3 or (data.dim() == 4 and is_color_channel_last is None):
        # grayscale images (2d or 3d)
        data_mean = [data.mean()]
        data_std = [data.std()]
    elif data.dim() == 4:  # equals 2d color images
        if is_color_channel_last:
            # Change to CHW representation
            data = data.permute(0, 3, 1, 2)
        assert data.shape[1] == 3, "Colored image must consist of three channels not " + data.shape[1]
        # color images
        data_mean = data.mean([0, 2, 3])
        data_std = data.std([0, 2, 3])
    elif data.dim() == 5:  # equals 3d color-images
        if is_color_channel_last:
            # Change to CHWD representation
            data = data.permute(0, 4, 1, 2, 3)
        assert data.shape[1] == 3, "Colored image must consist of three channels not {0}".format(data.shape[1])
        # color images
        data_mean = data.mean([0, 2, 3, 4])
        data_std = data.std([0, 2, 3, 4])
    normalize = torchvision.transforms.Normalize(data_mean, data_std)
    data = normalize(data)
    return data


def _load_medical_mnist_data(dataset_name: str, subset: str, normalize_channels: bool, colored: bool,
                             multiple_labelings: bool, downloads_path: str) -> (
        np.ndarray, np.ndarray):
    """
    Helper function to load medical MNIST data from https://medmnist.com/.
    Parameters
    ----------
    dataset_name : str
        name of the data set
    subset : str
        can be 'all', 'test', 'train' or 'val'. 'all' combines test, train and validation data
    normalize_channels : bool
        normalize each color-channel of the images
    colored : bool
        specifies if the images in the dataset are grayscale or colored
    multiple_labelings : bool
        specifies if the data set contains multiple labelings (for alternative clusterings)
    downloads_path : str
        path to the directory where the data is stored. If input was None this will be equal to
        '[USER]/Downloads/clustpy_datafiles'
    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array, the labels numpy array
    """
    subset = subset.lower()
    assert subset in ["all", "train",
                      "test", "val"], "subset must match 'all', 'train', 'test' or 'val'. Your input {0}".format(subset)
    # Check if data exists
    filename = _get_download_dir(downloads_path) + "/" + dataset_name + ".npz"
    if not os.path.isfile(filename):
        _download_file("https://zenodo.org/record/6496656/files/" + dataset_name + ".npz?download=1", filename)
    # Load data
    dataset = np.load(filename)
    if subset == "all" or subset == "train":
        data = dataset["train_images"]
        labels = dataset["train_labels"]
    if subset == "all" or subset == "test":
        test_data = dataset["test_images"]
        test_labels = dataset["test_labels"]
        if subset == "all":
            data = np.r_[data, test_data]
            labels = np.r_[labels, test_labels]
        else:
            data = test_data
            labels = test_labels
    if subset == "all" or subset == "val":
        val_data = dataset["val_images"]
        val_labels = dataset["val_labels"]
        if subset == "all":
            data = np.r_[data, val_data]
            labels = np.r_[labels, val_labels]
        else:
            data = val_data
            labels = val_labels
    dataset = None  # is needed so that the test folder can be deleted after the unit tests have finished
    # If desired, normalize channels
    data_torch = torch.Tensor(data)
    is_color_channel_last = 1 if colored else None
    if normalize_channels:
        data_torch = _torch_normalize_channels(data_torch, is_color_channel_last)
    # Flatten shape
    data_torch = _torch_flatten_shape(data_torch, is_color_channel_last, normalize_channels)
    # Move data to CPU
    data = data_torch.detach().cpu().numpy()
    # Sometimes the labels are contained in a separate dimension
    if labels.ndim != 1 and not multiple_labelings:
        assert labels.shape[1] == 1, "Data should only contain a single labeling"
        labels = labels[:, 0]
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    return data, labels


# training and data settings for dataset
def get_transform(name):
    if name == 'MNIST' or name == "RepeatedMNIST":
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif name == 'FashionMNIST':
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif name == 'SVHN':
        return transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])
    elif "EMNIST" in name:
        return transforms.Compose([transforms.ToTensor()])
    elif name in ["bloodmnist", "dermamnist"]:
        return transforms.Compose(
            [transforms.Resize((32,32), interpolation=PIL.Image.NEAREST), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    elif "openml" in name:
        return transforms.Compose([transforms.ToTensor()])


def get_dataset(name, ds_args, path=DATASET_PATH, seed=0):
    np.random.seed(seed)
    if name == 'MNIST':
        return get_MNIST(path)
    elif name == 'EMNIST':
        return get_EMNIST(path)
    elif name in ["bloodmnist", "dermamnist"]:
        return get_MedicalMNIST(path, medname=name)
    elif name == 'FashionMNIST':
        return get_FashionMNIST(path)
    elif name == 'SVHN':
        return get_SVHN(path)
    elif name == 'RepeatedMNIST':
        return get_RepeatedMNIST(path, **ds_args)
    elif "openml" in name:
        did = name.split("-")[-1]
        return get_openml(path, did)


def get_openml(path, did):
    openml.config.apikey = '3411e20aff621cc890bf403f104ac4bc'
    openml.config.set_cache_directory(path)
    ds = openml.datasets.get_dataset(did)

    # print(ds.default_target_attribute)
    data = ds.get_data(target=ds.default_target_attribute)
    # print(ds)
    # print(data[0])
    # print(data[0].dtypes)
    df = data[0].dropna(axis=1, how='all')
    df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    categorical_data = np.where(df.dtypes == "category")[0]
    # print(categorical_data)
    if len(categorical_data) and did != 40670:
        print("categorical")
        columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), categorical_data)], remainder='passthrough')
        X = columnTransformer.fit_transform(df)

        if isinstance(X, scipy.sparse._csr.csr_matrix):
            X = X.toarray()
        X = np.array(X, dtype=np.float32)
        # print(X.shape)

    else:
        print("numerical")
        X = np.asarray(df, dtype=np.float32)

    # X = np.asarray(data[0], dtype=np.float16)
    y = np.asarray(data[1])

    y = LabelEncoder().fit(y).transform(y)
    nSamps, dim = np.shape(X)
    testSplit = .2
    inds = np.random.permutation(nSamps)
    X = X[inds]
    y = y[inds]

    split = int((1. - testSplit) * nSamps)
    inds = np.random.permutation(nSamps)
    X_tr = X[inds][:split]
    X_tr = torch.Tensor(X_tr)
    y_tr = y[inds][:split]
    Y_tr = torch.Tensor(y_tr).long()

    X_te = torch.Tensor(X[inds][split:])
    Y_te = torch.Tensor(y[inds][split:]).long()

    # if did == "156":  # scalability analysis
    #     keep = 20_000
        # keep = 100_000
        # keep = 500_000
        # shape = X_tr.shape
        # original_length = shape[0]
        # n_delete = 800_000 - keep
        # rand_remove = np.random.choice(range(original_length), size=n_delete, replace=False)
        # Y_tr = np.delete(Y_tr, rand_remove, axis=0)
        # X_tr = np.delete(X_tr, rand_remove, axis=0)

    class_index, class_count = np.unique(Y_tr, return_counts=True)
    N = np.zeros(len(class_index))
    N[class_index.astype(int)] = class_count
    N = torch.tensor(N)

    return X_tr, Y_tr, X_te, Y_te, N


def get_SVHN(path):
    data_tr = datasets.SVHN(path, split='train', download=True)
    data_te = datasets.SVHN(path, split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)

    class_index, class_count = np.unique(Y_tr, return_counts=True)
    N = np.zeros(10)
    N[class_index.astype(int)] = class_count
    N = torch.tensor(N)
    return X_tr, Y_tr, X_te, Y_te, N


def get_MNIST(path):
    raw_tr = datasets.MNIST(path, train=True, download=True)
    raw_te = datasets.MNIST(path, train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets

    class_index, class_count = np.unique(Y_tr, return_counts=True)
    N = np.zeros(10)
    N[class_index.astype(int)] = class_count
    N = torch.tensor(N)
    return X_tr, Y_tr, X_te, Y_te, N


def get_MedicalMNIST(path, medname="dermamnist"):
    X_tr, Y_tr = _load_medical_mnist_data(medname, "train", normalize_channels=False, colored=True, multiple_labelings=False, downloads_path=path)
    X_te, Y_te = _load_medical_mnist_data(medname, "test", normalize_channels=False, colored=True, multiple_labelings=False, downloads_path=path)
    X_tr = X_tr.reshape(X_tr.shape[0], 28, 28, 3)
    # X_tr = ((X_tr - X_tr.min()) * (1 / (X_tr.max() - X_tr.min()) * 1)).astype('float32')

    X_te = X_te.reshape(X_te.shape[0], 28, 28, 3)
    # X_te = ((X_te - X_te.min()) * (1 / (X_te.max() - X_te.min()) * 1)).astype('float32')

    class_index, class_count = np.unique(Y_tr, return_counts=True)
    N = np.zeros(len(class_index))
    N[class_index.astype(int)] = class_count
    N = torch.tensor(N)
    return torch.Tensor(X_tr), torch.Tensor(Y_tr).long(), torch.Tensor(X_te), torch.Tensor(Y_te).long(), N


def get_EMNIST(path):
    raw_tr = datasets.EMNIST(path, split="balanced", train=True, download=True)
    raw_te = datasets.EMNIST(path, split="balanced", train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets

    class_index, class_count = np.unique(Y_tr, return_counts=True)
    N = np.zeros(62)
    N[class_index.astype(int)] = class_count
    N = torch.tensor(N)
    return X_tr, Y_tr, X_te, Y_te, N


def get_RepeatedMNIST(path, repetitions=50, ds_size=60_000):
    raw_tr = datasets.MNIST(path, train=True, download=True)
    raw_te = datasets.MNIST(path, train=False, download=True)

    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    shape = X_tr.shape
    original_length = shape[0]
    n_single = int(ds_size / repetitions)
    n_delete = int(original_length - n_single)
    rand_remove = np.random.choice(range(original_length), size=n_delete, replace=False)

    tr_y_rep = np.delete(Y_tr, rand_remove, axis=0)
    tr_x_rep = np.delete(X_tr, rand_remove, axis=0)

    X_tr = tr_x_rep.repeat(repetitions, 1, 1)
    Y_tr = tr_y_rep.repeat(repetitions)

    for rep in range(1, repetitions, 1):
        dataset_noise = torch.empty((n_single, 28, 28), dtype=torch.float32).normal_(0.0, 0.25)  #0.1
        start_idx = int(rep*n_single)
        end_idx = int((rep*n_single)+n_single)
        X_tr[start_idx:end_idx] = X_tr[start_idx:end_idx] + dataset_noise

    X_te = raw_te.data
    Y_te = raw_te.targets

    class_index, class_count = np.unique(Y_tr, return_counts=True)
    N = np.zeros(10)
    N[class_index.astype(int)] = class_count
    N = torch.tensor(N)
    return X_tr, Y_tr, X_te, Y_te, N


def get_FashionMNIST(path):
    raw_tr = datasets.FashionMNIST(path, train=True, download=True)
    raw_te = datasets.FashionMNIST(path, train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    class_index, class_count = np.unique(Y_tr, return_counts=True)
    N = np.zeros(10)
    N[class_index.astype(int)] = class_count
    N = torch.tensor(N)

    return X_tr, Y_tr, X_te, Y_te, N


def get_handler(name):
    if name == 'MNIST' or name == "RepeatedMNIST" or name == "EMNIST":
        return MNISTHandler
    elif name == 'FashionMNIST':
        return FashionMNISTHandler
    elif name == 'SVHN':
        return SVHNHandler
    elif "openml" in name:
        return OpenMLHandler
    elif name in ["pathmnist", "bloodmnist", "retinamnist", "dermamnist"]:
        return MNISTMEDHandler


class SVHNHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
            # x = x.type(torch.DoubleTensor)
        return x, y, index

    def __len__(self):
        return len(self.X)


class OpenMLHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


class FashionMNISTHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class MNISTHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy())
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class MNISTMEDHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = x.numpy()
            x = x.astype(np.uint8)
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
