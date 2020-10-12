import numpy as np
from sklearn.preprocessing import LabelEncoder

from .dataset_flat import load_from_pkl

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
