import os
import argparse
import pickle
import submitit

from get_base_parses import get_base_parses
from optimize_parses import optimize_parses
from refit_parses_multi import refit_parses_multi


def array_step(executor, func, jobs, inputs, errors):
    filt = lambda y : not errors[y]
    # submit jobs
    with executor.batch():
        for x in filter(filt, inputs):
            jobs[x] = executor.submit(func, x)
    # wait for completion
    for x in filter(filt, inputs):
        try:
            jobs[x].result()
        except Exception as err:
            errors[x] = err

    return jobs, errors

def save_errors(errors):
    with open("./logs/errors.pkl", "wb") as f:
        pickle.dump(errors, f)

def main(args):
    if not os.path.exists('./results'):
        os.mkdir('./results')
    run_IDs = list(range(20))
    errors = {r:None for r in run_IDs}
    jobs = {r:None for r in run_IDs}

    # initialize job executor
    executor = submitit.AutoExecutor(folder="./logs")
    executor.update_parameters(
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=3,
        slurm_mem='20GB',
        slurm_gres='gpu:1',
        slurm_time='8:00:00',
        slurm_job_name='osc',
        slurm_array_parallelism=20
    )

    # execute 3-step process sequentially
    print('step 1: parsing')
    fn = lambda r : get_base_parses(r, reverse=args.reverse)
    jobs, errors = array_step(executor, fn, jobs, run_IDs, errors)
    save_errors(errors)

    print('step 2: optimization')
    fn = lambda r : optimize_parses(r, reverse=args.reverse)
    jobs, errors = array_step(executor, fn, jobs, run_IDs, errors)
    save_errors(errors)

    print('step 3: re-fitting')
    executor.update_parameters(slurm_time='48:00:00') # more compute time needed for this step
    fn = lambda r : refit_parses_multi(r, reverse=args.reverse)
    jobs, errors = array_step(executor, fn, jobs, run_IDs, errors)
    save_errors(errors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    main(args)