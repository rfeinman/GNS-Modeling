import os
import shutil
import pickle
from sklearn.model_selection import train_test_split

from .. import DATADIR


class Dataset:
    def __init__(self, drawings, images):
        self.drawings = drawings
        self.images = images
        self.attributes = ['drawings', 'images']

    def keys(self):
        return self.drawings.keys()

    def sub(self, select):
        single = isinstance(select, str)
        if single:
            assert select in self.keys()
        else:
            assert all([key in self.keys() for key in select])
        kwargs = {}
        for attr in self.attributes:
            x = getattr(self, attr)
            if single:
                kwargs[attr] = x[select]
            else:
                kwargs[attr] = {key : x[key] for key in select}
        D = type(self)(**kwargs)
        return D

    def save(self, save_dir):
        if os.path.exists(save_dir):
            if overwrite():
                shutil.rmtree(save_dir)
            else:
                return
        os.mkdir(save_dir)
        for attr in self.attributes:
            fname = os.path.join(save_dir, attr+'.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(getattr(self, attr), f)

    def alphabet_split(self, test_size, random_state=None):
        akeys = list(self.keys())
        akeys_train, akeys_test = train_test_split(
            akeys, test_size=test_size, random_state=random_state
        )
        kwargs_train = {}
        kwargs_test = {}
        for attr in self.attributes:
            x = getattr(self, attr)
            kwargs_train[attr] = {a : x[a] for a in akeys_train}
            kwargs_test[attr] = {a : x[a] for a in akeys_test}
        D_train = type(self)(**kwargs_train)
        D_test = type(self)(**kwargs_test)

        return D_train, D_test

    def character_split(self, test_size, random_state=None):
        ckeys_train = {}
        ckeys_test = {}
        for a in self.keys():
            sub = self.sub(a)
            ckeys = list(sub.keys())
            ckeys_train[a], ckeys_test[a] = train_test_split(
                ckeys, test_size=test_size, random_state=random_state
            )

        kwargs_train = {}
        kwargs_test = {}
        for attr in self.attributes:
            x = getattr(self, attr)
            kwargs_train[attr] = {}
            kwargs_test[attr] = {}
            for a in x.keys():
                kwargs_train[attr][a] = {c : x[a][c] for c in ckeys_train[a]}
                kwargs_test[attr][a] = {c : x[a][c] for c in ckeys_test[a]}
        D_train = type(self)(**kwargs_train)
        D_test = type(self)(**kwargs_test)

        return D_train, D_test

    def example_split(self, test_size, random_state=None):
        ekeys_train = {}
        ekeys_test = {}
        for a in self.keys():
            sub_a = self.sub(a)
            ekeys_train[a] = {}
            ekeys_test[a] = {}
            for c in sub_a.keys():
                sub_c = sub_a.sub(c)
                ekeys = list(sub_c.keys())
                ekeys_train[a][c], ekeys_test[a][c] = train_test_split(
                    ekeys, test_size=test_size, random_state=random_state
                )

        kwargs_train = {}
        kwargs_test = {}
        for attr in self.attributes:
            x = getattr(self, attr)
            kwargs_train[attr] = {}
            kwargs_test[attr] = {}
            for a in x.keys():
                kwargs_train[attr][a] = {}
                kwargs_test[attr][a] = {}
                for c in x[a].keys():
                    kwargs_train[attr][a][c] = {e : x[a][c][e] for e in ekeys_train[a][c]}
                    kwargs_test[attr][a][c] = {e : x[a][c][e] for e in ekeys_test[a][c]}
        D_train = type(self)(**kwargs_train)
        D_test = type(self)(**kwargs_test)

        return D_train, D_test


class ProcessedDataset(Dataset):
    def __init__(self, drawings, images, splines, canvases):
        super().__init__(drawings, images)
        self.splines = splines
        self.canvases = canvases
        self.attributes += ['splines', 'canvases']


def overwrite():
    while True:
        response = input(
            "Save path already exists. "
            "Do you want to overwrite? [y/n]: "
        )
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Invalid input. Please choose 'y' or 'n'.")

def load_dataset_pkl(save_dir=None, processed=True, key=None):
    if save_dir is None:
        assert key in ['train', 'test']
        if key == 'train':
            save_dir = os.path.join(DATADIR, 'background_set_pkl')
        else:
            save_dir = os.path.join(DATADIR, 'evaluation_set_pkl')

    kwargs = {}
    attributes = ['drawings', 'images']
    if processed:
        attributes += ['splines', 'canvases']
    for attr in attributes:
        fname = os.path.join(save_dir, attr+'.pkl')
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                kwargs[attr] = pickle.load(f)
        else:
            kwargs[attr] = None
    if processed:
        D = ProcessedDataset(**kwargs)
    else:
        D = Dataset(**kwargs)

    return D