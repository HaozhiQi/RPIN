import os
import torch
import random
from pprint import pprint
import argparse
import numpy as np
from torch.utils.data import DataLoader
from neuralphys.datasets.pyp import PyPhys
from neuralphys.utils.config import _C as C
from neuralphys.models import *
from neuralphys.evaluator_pred import PredEvaluator


def arg_parse():
    parser = argparse.ArgumentParser(description='RPIN parameters')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--predictor-init', type=str, default=None)
    parser.add_argument('--predictor-arch', type=str, default=None)
    parser.add_argument('--gpus', type=str)
    parser.add_argument('--eval-hit', action='store_true')
    return parser.parse_args()


def main():
    args = arg_parse()
    pprint(vars(args))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if torch.cuda.is_available():
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        num_gpus = 1  # torch.cuda.device_count()
        print('Use {} GPUs'.format(num_gpus))
    else:
        assert NotImplementedError

    # --- setup config files
    C.merge_from_file(args.cfg)
    C.INPUT.PRELOAD_TO_MEMORY = False
    C.freeze()

    cache_name = 'figures/' + C.DATA_ROOT.split('/')[1] + '/'
    if args.predictor_init:
        cache_name += args.predictor_init.split('/')[-2]
    output_dir = os.path.join(C.OUTPUT_DIR, cache_name)

    if args.eval_hit and 'phyre' in C.DATA_ROOT:
        from neuralphys.evaluator_phyre_plan import PhyrePlanEvaluator
        model = eval(args.predictor_arch + '.Net')()
        model.to(torch.device('cuda'))
        model = torch.nn.DataParallel(
            model, device_ids=[0]
        )
        cp = torch.load(args.predictor_init, map_location=f'cuda:0')
        model.load_state_dict(cp['model'])
        tester = PhyrePlanEvaluator(
            device=torch.device(f'cuda'),
            num_gpus=1,
            pred_model=model,
            output_dir=output_dir,
        )
        tester.test()
        return

    # --- setup data loader
    print('initialize dataset')
    split_name = 'planning' if (args.eval_hit and 'phyre' not in C.DATA_ROOT) else 'test'
    val_set = PyPhys(data_root=C.DATA_ROOT, split=split_name)
    batch_size = 1 if C.RPIN.VAE else C.SOLVER.BATCH_SIZE
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16)

    # prediction evaluation
    if not args.eval_hit:
        model = eval(args.predictor_arch + '.Net')()
        model.to(torch.device('cuda'))
        model = torch.nn.DataParallel(
            model, device_ids=[0]
        )
        cp = torch.load(args.predictor_init, map_location=f'cuda:0')
        model.load_state_dict(cp['model'])
        tester = PredEvaluator(
            device=torch.device('cuda'),
            val_loader=val_loader,
            num_gpus=1,
            model=model,
            output_dir=output_dir,
        )
        tester.test()
        return


if __name__ == '__main__':
    main()
