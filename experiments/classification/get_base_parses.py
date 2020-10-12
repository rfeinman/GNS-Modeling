import argparse
import os
import time
import torch
from pybpl.util import nested_map
from pybpl.splines import get_stk_from_bspline
from gns.inference.parsing import get_topK_parses
from gns.omniglot.classification import ClassificationDataset
from gns.type import TypeModel
from gns.utils.experiments import mkdir, time_string



@torch.no_grad()
def model_score_fn(model, parses):
    drawings = nested_map(lambda x: get_stk_from_bspline(x), parses)
    if torch.cuda.is_available():
        drawings = nested_map(lambda x: x.cuda(), drawings)
        parses = nested_map(lambda x: x.cuda(), parses)
    losses = model.losses_fn(
        parses, drawings, filter_small=False, denormalize=True)
    return -losses.cpu()

def save_img_results(save_dir, img_id, parses, log_probs, reverse):
    appendix = 'test' if reverse else 'train'
    save_dir_i = os.path.join(save_dir, appendix+'_%0.2i' % img_id)
    mkdir(save_dir_i)
    # save log_probs
    lp_file = os.path.join(save_dir_i, 'log_probs.pt')
    torch.save(log_probs, lp_file)
    # save parses
    K = len(parses)
    for k in range(K):
        parse = parses[k]
        parse_file = os.path.join(save_dir_i, 'parse_%i.pt' % k)
        torch.save(parse, parse_file)

def get_base_parses(run_id, trials_per=800, reverse=False, dry_run=False):
    print('run_id: %i' % run_id)
    print('Loading model...')
    type_model = TypeModel().eval()
    if torch.cuda.is_available():
        type_model = type_model.cuda()
    score_fn = lambda parses : model_score_fn(type_model, parses)

    print('Loading classification dataset...')
    dataset = ClassificationDataset(osc_folder='./one-shot-classification')
    run = dataset.runs[run_id]
    if reverse:
        imgs = run.test_imgs
    else:
        imgs = run.train_imgs
    run_dir = './run%0.2i' % (run_id+1)
    save_dir = os.path.join(run_dir, 'base_parses')
    if not dry_run:
        mkdir(run_dir)
        mkdir(save_dir)

    print('Collecting top-K parses for each train image...')
    nimg = len(imgs)
    for i in range(nimg):
        start_time = time.time()
        parses, log_probs = get_topK_parses(
            imgs[i], k=5, score_fn=score_fn, configs_per=1,
            trials_per=trials_per)
        total_time = time.time() - start_time
        print('image %i/%i took %s' % (i+1, nimg, time_string(total_time)))
        if dry_run:
            continue
        save_img_results(save_dir, i, parses, log_probs, reverse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials_per', type=int, default=800)
    parser.add_argument('--run_id', type=int, required=True)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    kwargs = vars(args)
    get_base_parses(**kwargs)