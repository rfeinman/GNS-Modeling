import argparse
from gns.omniglot.classification import ClassificationDataset

from refit_parses_single import refit_parses_single



def refit_parses_multi(run_id, iterations=1500, reverse=False):
    dataset = ClassificationDataset(osc_folder='./one-shot-classification')
    run = dataset.runs[run_id]
    ntest = len(run.test_imgs)

    for test_id in range(ntest):
        print('test_id: %i' % test_id)
        refit_parses_single(
            run_id, test_id, iterations=iterations, reverse=reverse, run=run)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    kwargs = vars(args)
    refit_parses_multi(**kwargs)