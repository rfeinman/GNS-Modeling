import argparse
import os
import time
import torch
from gns.rendering import Renderer
from gns.token import TokenModel
from gns.inference import optimization as opt
from gns.omniglot.classification import ClassificationDataset
from gns.utils.experiments import mkdir, time_string



def config_for_refit(parse):
    parse.x.requires_grad_(False)
    parse.blur_base.data = torch.tensor(16., dtype=torch.float)
    parse.epsilon_base.data = torch.tensor(0.5, dtype=torch.float)

def load_tuned_parses(load_dir, ntrain, reverse=False):
    appendix = 'test' if reverse else 'train'
    tuned_parses = []
    K_per_img = {}
    for img_id in range(ntrain):
        savedir = os.path.join(load_dir, appendix + '_%0.2i' % img_id)
        parse_files = [f for f in os.listdir(savedir) if f.startswith('parse')]
        K_per_img[img_id] = len(parse_files)
        for f in sorted(parse_files):
            state_dict = torch.load(os.path.join(savedir, f), map_location='cpu')
            init_parse = [val for key,val in state_dict.items() if key.startswith('x')]
            parse = opt.ParseWithToken(init_parse)
            parse.load_state_dict(state_dict)
            config_for_refit(parse)
            tuned_parses.append(parse)

    return tuned_parses, K_per_img

def save_new_parses(parses_j, log_probs_j, save_dir, K_per_img, test_id, reverse=False):
    """
    i : train image index
    k : parse index
    """
    appendix_i = 'test' if reverse else 'train'
    appendix_j = 'train' if reverse else 'test'
    curr = 0
    for train_id, K in K_per_img.items():
        # get savedir paths
        save_dir_i = os.path.join(save_dir, appendix_i+'_%0.2i' % train_id)
        mkdir(save_dir_i)
        save_dir_ij = os.path.join(save_dir_i, appendix_j+'_%0.2i' % test_id)
        mkdir(save_dir_ij)
        # get data subset
        parses_ij = parses_j[curr : curr+K]
        log_probs_ij = log_probs_j[curr : curr+K]
        curr += K
        # save log-probs
        lp_file = os.path.join(save_dir_ij, 'log_probs.pt')
        torch.save(log_probs_ij, lp_file)
        # save parses
        for k in range(K):
            parse = parses_ij[k]
            parse_file = os.path.join(save_dir_ij, 'parse_%i.pt' % k)
            torch.save(parse.state_dict(), parse_file)


def refit_parses_single(run_id, test_id, iterations=1500, reverse=False,
                        run=None, dry_run=False):
    run_dir = './results/run%0.2i' % (run_id+1)
    load_dir = os.path.join(run_dir, 'tuned_parses')
    save_dir = os.path.join(run_dir, 'refitted_parses')
    assert os.path.exists(run_dir)
    assert os.path.exists(load_dir)
    if not dry_run:
        mkdir(save_dir)

    print('Loading model...')
    token_model = TokenModel()
    renderer = Renderer(blur_fsize=21)
    if torch.cuda.is_available():
        token_model = token_model.cuda()
        renderer = renderer.cuda()
    model = opt.FullModel(renderer=renderer, token_model=token_model)

    print('Loading parses...')
    # load classification dataset and select run
    if run is None:
        dataset = ClassificationDataset(osc_folder='./one-shot-classification')
        run = dataset.runs[run_id]
    if reverse:
        ntrain = len(run.test_imgs)
        test_img = torch.from_numpy(run.train_imgs[test_id]).float()
    else:
        ntrain = len(run.train_imgs)
        test_img = torch.from_numpy(run.test_imgs[test_id]).float()
    if torch.cuda.is_available():
        test_img = test_img.cuda()

    # load tuned parses
    parse_list, K_per_img = load_tuned_parses(load_dir, ntrain, reverse)
    images = test_img.expand(len(parse_list), 105, 105)
    print('total # parses: %i' % len(images))


    print('Optimizing parses...')
    # initialize Parse modules and optimizer
    render_params = [p for parse in parse_list for p in parse.render_params]
    stroke_params = [p for parse in parse_list for p in parse.stroke_params if p.requires_grad]
    param_groups = [
        {'params': render_params, 'lr': 0.087992},
        {'params': stroke_params, 'lr': 0.166810}
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
    save_new_parses(parse_list, parse_scores, save_dir, K_per_img, test_id, reverse)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--test_id', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    kwargs = vars(args)
    refit_parses_single(**kwargs)