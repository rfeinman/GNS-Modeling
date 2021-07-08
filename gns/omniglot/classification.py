import os
import numpy as np


def load_image(fname):
    from imageio import imread
    img = imread(fname)
    img = np.array(img, dtype=bool)
    img = np.logical_not(img)
    return img

def get_run_data(osc_folder, run_id):
    """
    Parameters
    ----------
    osc_folder : str
        path to one-shot-classification folder
        see https://github.com/brendenlake/omniglot/tree/master/python/one-shot-classification
    run_id : int
        an integer in range [1,20] inclusive
    """
    assert 1 <= run_id <= 20
    runs_folder = os.path.join(osc_folder, 'all_runs')
    fname_label = os.path.join(runs_folder, 'run%0.2i'%run_id, 'class_labels.txt')
    with open(fname_label) as f:
        content = f.read().splitlines()
    pairs = [line.split() for line in content]
    test_files  = [pair[0] for pair in pairs]
    train_files = [pair[1] for pair in pairs]
    ntrain = len(train_files)
    ntest = len(test_files)

    train_imgs = np.zeros((ntrain, 105, 105), dtype=bool)
    for i, f in enumerate(train_files):
        train_imgs[i] = load_image(os.path.join(runs_folder, f))

    test_imgs = np.zeros((ntest, 105, 105), dtype=bool)
    for i, f in enumerate(test_files):
        test_imgs[i] = load_image(os.path.join(runs_folder, f))

    return train_imgs, test_imgs


class Run:
    """
    One classification "run" includes 20 individual trials
    """
    def __init__(self, train_imgs, test_imgs):
        """

        """
        self.train_imgs = train_imgs
        self.test_imgs = test_imgs

    def __len__(self):
        return self.train_imgs.shape[0]

    def __getitem__(self, ix):
        return self.train_imgs[ix], self.test_imgs


class ClassificationDataset:
    """
    The classification dataset includes 20 unique classification "runs",
    each with 20 trials
    """
    def __init__(self, osc_folder):
        """
        Parameters
        ----------
        osc_folder : str
            path to one-shot-classification folder
            see https://github.com/brendenlake/omniglot/tree/master/python/one-shot-classification
        """
        assert os.path.exists(osc_folder)
        runs = []
        for run_id in range(1,21):
            train_imgs, test_imgs = get_run_data(osc_folder, run_id)
            run = Run(train_imgs, test_imgs)
            runs.append(run)
        self.runs = runs
        self.run_lengths = [len(run) for run in self.runs]

    def __len__(self):
        return sum(self.run_lengths)

    def __getitem__(self, ix):
        curr = 0
        for run in self.runs:
            ntrials = len(run)
            for trial_id in range(ntrials):
                if curr == ix:
                    return run[trial_id]
                curr += 1