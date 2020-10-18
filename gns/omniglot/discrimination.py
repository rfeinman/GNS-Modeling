import os
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder

from .dataset_flat import load_from_pkl
from .classification import Run

TARGETS = [
    {'alphabet': 'Atemayar_Qelisayer',
     'label': 'Atemayar (A)',
     'background': False,
     'char_ids' : [6,10,23,25]},

    {'alphabet': 'Inuktitut_(Canadian_Aboriginal_Syllabics)',
     'label': 'Inukitut (I)',
     'background': True,
     'char_ids' : [1,2,3,11]},

    {'alphabet': 'Korean',
     'label': 'Korean (K)',
     'background': True,
     'char_ids' : [4,28,29,39]},

    {'alphabet': 'Burmese_(Myanmar)',
     'label': 'Myanmar (M)',
     'background': True,
     'char_ids' : [1,25,31,32]},

    {'alphabet': 'Sylheti',
     'label': 'Sylheti (S)',
     'background': False,
     'char_ids' : [17,19,20,28]},

    {'alphabet': 'Tagalog',
     'label': 'Tagalog (T)',
     'background': True,
     'char_ids' : [2,5,6,10]}
]


def get_class_data(dataset, tgt):
    images = np.stack([ex.image for ex in dataset if ex.alphabet == tgt['alphabet']])
    labels = np.array([ex.character_id for ex in dataset if ex.alphabet == tgt['alphabet']])
    labels = LabelEncoder().fit_transform(labels) + 1
    sel = np.isin(labels, tgt['char_ids']) # keep only target character classes
    return images[sel], labels[sel]

def load_discrim_data(root):
    background_set = load_from_pkl(root, background=True)
    evaluation_set = load_from_pkl(root, background=False)
    images, characters, alphabets = [],[],[]
    for tgt in TARGETS:
        dataset = background_set if tgt['background'] else evaluation_set
        images_t, characters_t = get_class_data(dataset, tgt)
        alphabets_t = [tgt['label']] * len(characters_t)
        images.append(images_t)
        characters.append(characters_t)
        alphabets.extend(alphabets_t)
    images = np.concatenate(images)
    characters = np.concatenate(characters)
    alphabets = np.array(alphabets)

    return images, characters, alphabets


class DiscriminationDataset:
    """
    The classification dataset includes X unique classification "runs",
    each with 24 trials
    """
    def __init__(self, root):
        self.images, self.labels_c, self.labels_a = load_discrim_data(root)
        self.labels = list(zip(self.labels_a, self.labels_c))
        self.classes = sorted(set(self.labels))

    def get(self):
        train_idx = np.zeros(len(self.classes), dtype=np.int64)
        test_idx = np.zeros(len(self.classes), dtype=np.int64)
        for j, (a,c) in enumerate(self.classes):
            idx, = np.where((self.labels_a == a) & (self.labels_c == c))
            train_idx[j], test_idx[j] = np.random.choice(idx, 2, replace=False)

        return Run(self.images[train_idx], self.images[test_idx])


class DiscriminationDatasetPrecomputed(DiscriminationDataset):
    """
    Precomputed version
    """
    def __init__(self, root, num_runs=10, seed=383):
        super().__init__(root)
        np.random.seed(seed)
        self.runs = [self.get() for _ in range(num_runs)]

    def __getitem__(self, idx):
        return self.runs[idx]



# load human data

def load_human_data(root=None):
    # load
    human_results = sio.loadmat(os.path.join(root, 'confusable_sim_mat.mat'))
    discrim_set = sio.loadmat(os.path.join(root, 'discrim_select_newset.mat'))
    # extract elements
    acc_M = human_results['acc_M']
    #names_discrim = discrim_set['names_discrim']
    alphabet_names = discrim_set['names_discrim_short'][0]
    character_ids = discrim_set['char_num']
    # process
    sim_human_M = (acc_M + acc_M.T) / 2
    alphabet_names = [elt[0] for elt in alphabet_names] # list of length 6
    character_ids = np.concatenate([elt[0].T for elt in character_ids]) # array of size [6,4]

    return sim_human_M, alphabet_names, character_ids