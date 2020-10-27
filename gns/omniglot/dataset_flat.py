import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from pybpl.util.stroke import dist_along_traj
from pybpl.data import unif_space

from .dataset import Dataset, ProcessedDataset
from .dataset import load_dataset_pkl


class Example:
    def __init__(self, alphabet, character_id, image, drawing,
                 splines=None, canvases=None):
        self.alphabet = alphabet
        self.character_id = character_id
        self.image = image
        self.drawing = list(drawing.values())
        self.splines = list(splines.values())
        self.canvases = list(canvases.values())

    @property
    def num_strokes(self):
        return len(self.drawing)

    def get_stk_lengths(self, uniform=False):
        lengths = []
        for i in range(self.num_strokes):
            stk_i = self.drawing[i][:,:2]
            length_i = stk_length(stk_i, uniform)
            lengths.append(length_i)
        return lengths

    def rm_small_strokes(self, mindist):
        ns = self.num_strokes
        lengths = self.get_stk_lengths()
        for attr_key in ['drawing', 'splines']:
            attr = getattr(self, attr_key)
            attr = [attr[i] for i in range(ns) if lengths[i] >= mindist]
            setattr(self, attr_key, attr)
        return self.num_strokes > 0

def stk_length(stk, uniform=False):
    if uniform:
        stk = unif_space(stk)
    length = dist_along_traj(stk)
    return length



class DatasetFlat:
    def __init__(self, examples=None):
        if examples is not None:
            assert isinstance(examples, list)
            self.examples = examples
        else:
            self.examples = []

    def load_from_original(self, D):
        assert isinstance(D, Dataset) or isinstance(D, ProcessedDataset)
        processed = isinstance(D, ProcessedDataset)

        examples = []
        c_id = 0
        for a in D.drawings.keys():
            for c in D.drawings[a].keys():
                for e in D.drawings[a][c].keys():
                    if processed:
                        ex = Example(
                            alphabet=a,
                            character_id=c_id,
                            image=D.images[a][c][e],
                            drawing=D.drawings[a][c][e],
                            splines=D.splines[a][c][e],
                            canvases=D.canvases[a][c][e]
                        )
                    else:
                        ex = Example(
                            alphabet=a,
                            character_id=c_id,
                            image=D.images[a][c][e],
                            drawing=D.drawings[a][c][e]
                        )
                    examples.append(ex)
                c_id += 1
        self.examples = examples

        return self

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def rm_small_strokes(self, mindist):
        keep_fn = lambda ex : ex.rm_small_strokes(mindist)
        self.examples = list(filter(keep_fn, self.examples))

    def alphabet_split(self, test_size, random_state=None):
        alphabets = [ex.alphabet for ex in self.examples]
        alphabets = np.unique(alphabets)
        a_train, a_test = train_test_split(
            alphabets, test_size=test_size, random_state=random_state
        )
        D_train = DatasetFlat([ex for ex in self.examples if ex.alphabet in a_train])
        D_test = DatasetFlat([ex for ex in self.examples if ex.alphabet in a_test])
        return D_train, D_test

    def character_split(self, test_size, random_state=None):
        idx = np.arange(len(self.examples))
        labels = [ex.alphabet for ex in self.examples]
        idx_train, idx_test = train_test_split(
            idx, test_size=test_size, stratify=labels,
            random_state=random_state
        )
        D_train = DatasetFlat([self.examples[i] for i in idx_train])
        D_test = DatasetFlat([self.examples[i] for i in idx_test])
        return D_train, D_test

    def example_split(self, test_size, random_state=None):
        idx = np.arange(len(self.examples))
        labels = [ex.character_id for ex in self.examples]
        idx_train, idx_test = train_test_split(
            idx, test_size=test_size, stratify=labels,
            random_state=random_state
        )
        D_train = DatasetFlat([self.examples[i] for i in idx_train])
        D_test = DatasetFlat([self.examples[i] for i in idx_test])
        return D_train, D_test

def load_from_orig(root, background=True, processed=True, canvases=True):
    D = load_dataset_pkl(root, background, processed)
    D = DatasetFlat().load_from_original(D)
    if canvases:
        return D
    for ex in D:
        del ex.canvases
    return D

def load_from_pkl(root, background=True, canvases=True):
    if background:
        save_file = os.path.join(root, 'background_set_new.pkl')
    else:
        save_file = os.path.join(root, 'evaluation_set_new.pkl')
    with open(save_file, 'rb') as f:
        D = pickle.load(f)
    if canvases:
        return D
    for ex in D:
        del ex.canvases
    return D