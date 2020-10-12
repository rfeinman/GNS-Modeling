import argparse
import os
import time
import numpy as np
from imageio import imread
import torch
from pybpl.util import nested_map
from pybpl.splines import get_stk_from_bspline
from gns.inference.parsing import get_topK_parses
from gns.rendering import Renderer
from gns.token import TokenModel
from gns.type import TypeModel
from gns.inference import optimization as opt
from gns.utils.experiments import mkdir, time_string



def load_image(fname):
    img = imread(fname)
    img = np.array(img, dtype=bool)
    img = np.logical_not(img)
    return img

@torch.no_grad()
def parse_score_fn(model, parses):
    drawings = nested_map(lambda x: get_stk_from_bspline(x), parses)
    if torch.cuda.is_available():
        drawings = nested_map(lambda x: x.cuda(), drawings)
        parses = nested_map(lambda x: x.cuda(), parses)
    losses = model.losses_fn(
        parses, drawings, filter_small=False, denormalize=True)
    return -losses.cpu()

def select_parses(score_fn, images, configs_per=1, trials_per=800):
    nimg = len(images)
    base_parses = []
    for i in range(nimg):
        start_time = time.time()
        parses, log_probs = get_topK_parses(
            images[i], k=5, score_fn=score_fn,
            configs_per=configs_per, trials_per=trials_per)
        total_time = time.time() - start_time
        print('parsing image %i/%i took %s' % (i+1, nimg, time_string(total_time)))
        base_parses.append(parses)
    return base_parses

def process_for_opt(base_parses, images):
    nimg = len(images)
    parse_list = []
    target_imgs = []
    K_per_img = {}
    for i in range(nimg):
        img = torch.from_numpy(images[i]).float()
        if torch.cuda.is_available():
            img = img.cuda()
        K = len(base_parses[i])
        parse_list.extend(base_parses[i])
        target_imgs.append(img[None].expand(K,105,105))
        K_per_img[i] = K
    parse_list = [opt.ParseWithToken(p) for p in parse_list]
    target_imgs = torch.cat(target_imgs)
    return parse_list, target_imgs, K_per_img

def optimize_parses(model, parse_list, target_imgs, iterations=1500):
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
        parse_list, target_imgs,
        loss_fn=model.losses_fn,
        iterations=iterations,
        optimizer=optimizer,
        tune_blur=True,
        tune_fn=model.likelihood_losses_fn
    )
    total_time = time.time() - start_time
    time.sleep(0.5)
    print('Took %s' % time_string(total_time))
    parse_scores = -losses[-1]
    return parse_list, parse_scores

def save_parses(parse_list, log_probs, save_dir, K_per_img, idx):
    curr = 0
    for i, K in K_per_img.items():
        parse_list_img = parse_list[curr:curr+K]
        log_probs_img = log_probs[curr:curr+K]
        curr += K
        save_dir_i = os.path.join(save_dir, 'img_%0.2i' % (idx[i]+1))
        mkdir(save_dir_i)
        # save log_probs
        lp_file = os.path.join(save_dir_i, 'log_probs.pt')
        torch.save(log_probs_img, lp_file)
        # save parses
        for k in range(K):
            parse = parse_list_img[k]
            parse_file = os.path.join(save_dir_i, 'parse_%i.pt' % k)
            torch.save(parse.state_dict(), parse_file)

def main(batch_ix, configs_per=1, trials_per=800, iterations=1500, dry_run=False):
    assert 0 <= batch_ix < 5
    idx = np.arange(batch_ix*10, (batch_ix+1)*10)
    target_dir = './targets'
    save_dir = os.path.join('./parses')
    mkdir(save_dir)

    # load images
    images = np.zeros((10,105,105), dtype=bool)
    for i, ix in enumerate(idx):
        images[i] = load_image(os.path.join(target_dir, 'handwritten%i.png' % (ix+1)))

    # ------------------
    #   Select Parses
    # ------------------

    # load type model
    print('loading model...')
    type_model = TypeModel().eval()
    if torch.cuda.is_available():
        type_model = type_model.cuda()
    score_fn = lambda parses : parse_score_fn(type_model, parses)

    # get base parses
    print('Collecting top-K parses for each train image...')
    base_parses = select_parses(score_fn, images, configs_per, trials_per)
    parse_list, target_imgs, K_per_img = process_for_opt(base_parses, images)


    # --------------------
    #   Optimize Parses
    # --------------------

    # load full model
    token_model = TokenModel()
    renderer = Renderer()
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        token_model = token_model.cuda()
        renderer = renderer.cuda()
    model = opt.FullModel(
        renderer=renderer, type_model=type_model, token_model=token_model,
        denormalize=True
    )

    print('Optimizing top-K parses...')
    parse_list, parse_scores = optimize_parses(
        model, parse_list, target_imgs, iterations)

    if not dry_run:
        save_parses(parse_list, parse_scores, save_dir, K_per_img, idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_ix', type=int, required=True)
    parser.add_argument('--configs_per', type=int, default=1)
    parser.add_argument('--trials_per', type=int, default=800)
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)