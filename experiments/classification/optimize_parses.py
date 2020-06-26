import argparse
import os
import time
import torch
from gns.rendering import Renderer
from gns.token import TokenModel
from gns.type import TypeModel
from gns.inference import optimization as opt
from gns.omniglot.classification import ClassificationDataset

from util import mkdir, time_string



def load_parses(run, load_dir, reverse=False):
    train_imgs = run.test_imgs if reverse else run.train_imgs
    appendix = 'test' if reverse else 'train'
    ntrain = len(train_imgs)
    base_parses = []
    images = []
    K_per_img = {}
    for i in range(ntrain):
        load_dir_i = os.path.join(load_dir, appendix + '_%0.2i' % i)
        #log_probs = torch.load(os.path.join(load_dir_i, 'log_probs.pt'))
        parse_files = [f for f in os.listdir(load_dir_i) if f.startswith('parse')]
        parses_i = [torch.load(os.path.join(load_dir_i,f)) for f in sorted(parse_files)]
        img_i = torch.from_numpy(train_imgs[i]).float()
        if torch.cuda.is_available():
            img_i = img_i.cuda()
        K = len(parses_i)
        base_parses.extend(parses_i)
        images.append(img_i[None].expand(K,105,105))
        K_per_img[i] = K
    images = torch.cat(images)

    return base_parses, images, K_per_img

def save_new_parses(parse_list, log_probs, save_dir, K_per_img, reverse=False):
    appendix = 'test' if reverse else 'train'
    curr = 0
    for i, K in K_per_img.items():
        parse_list_img = parse_list[curr:curr+K]
        log_probs_img = log_probs[curr:curr+K]
        curr += K
        save_dir_i = os.path.join(save_dir, appendix+'_%0.2i' % i)
        mkdir(save_dir_i)
        # save log_probs
        lp_file = os.path.join(save_dir_i, 'log_probs.pt')
        torch.save(log_probs_img, lp_file)
        # save parses
        for k in range(K):
            parse = parse_list_img[k]
            parse_file = os.path.join(save_dir_i, 'parse_%i.pt' % k)
            torch.save(parse.state_dict(), parse_file)

def optimize_parses(run_id, iterations=1500, reverse=False, dry_run=False):
    run_dir = './run%0.2i' % (run_id+1)
    load_dir = os.path.join(run_dir, 'base_parses')
    save_dir = os.path.join(run_dir, 'tuned_parses')
    assert os.path.exists(run_dir)
    assert os.path.exists(load_dir)
    if not dry_run:
        mkdir(save_dir)

    print('Loading model and data...')
    type_model = TypeModel().eval()
    token_model = TokenModel()
    renderer = Renderer()
    # move to GPU if available
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        type_model = type_model.cuda()
        token_model = token_model.cuda()
        renderer = renderer.cuda()

    # build full model
    model = opt.FullModel(
        renderer=renderer, type_model=type_model, token_model=token_model,
        denormalize=True)

    print('Loading data...')
    # load classification dataset and select run
    dataset = ClassificationDataset(osc_folder='./one-shot-classification')
    run = dataset.runs[run_id]

    # load images and base parses for this run
    base_parses, images, K_per_img = load_parses(run, load_dir, reverse)
    assert len(base_parses) == len(images)
    print('total # parses: %i' % len(images))


    print('Optimizing parses...')
    # initialize Parse modules and optimizer
    parse_list = [opt.ParseWithToken(p) for p in base_parses]
    render_params = [p for parse in parse_list for p in parse.render_params]
    stroke_params = [p for parse in parse_list for p in parse.stroke_params]
    param_groups = [
        {'params': render_params, 'lr': 0.306983},
        {'params': stroke_params, 'lr': 0.044114}
    ]
    optimizer = torch.optim.Adam(param_groups)

    # optimize
    start_time = time.time()
    losses, states = opt.optimize_parselist(
        parse_list, images,
        loss_fn=model.losses_fn,
        iterations=iterations,
        optimizer=optimizer,
        tune_blur=True,
        tune_fn=model.likelihood_losses_fn
    )
    total_time = time.time() - start_time
    time.sleep(0.5)
    print('Took %s' % time_string(total_time))
    if dry_run:
        return

    parse_scores = -losses[-1]
    save_new_parses(parse_list, parse_scores, save_dir, K_per_img, reverse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    kwargs = vars(args)
    optimize_parses(**kwargs)