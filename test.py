import os
import torch
import random
import argparse
import numpy as np

from pprint import pprint
from torch.utils.data import DataLoader

from rpin.models import *
from rpin.datasets import *
from rpin.utils.config import _C as C
from rpin.evaluator_plan import PlannerPHYRE
from rpin.evaluator_pred import PredEvaluator


def arg_parse():
    parser = argparse.ArgumentParser(description='RPIN parameters')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--predictor-init', type=str, help='', default=None)
    parser.add_argument('--predictor-arch', type=str, default=None)
    parser.add_argument('--plot-image', type=int, default=0, help='how many images are plotted')
    parser.add_argument('--gpus', type=str)
    parser.add_argument('--eval', type=str, default=None)
    # below are only for PHYRE planning
    parser.add_argument('--start-id', default=0, type=int)
    parser.add_argument('--end-id', default=0, type=int)
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
        num_gpus = torch.cuda.device_count()
        print('Use {} GPUs'.format(num_gpus))
    else:
        assert NotImplementedError

    # --- setup config files
    C.merge_from_file(args.cfg)
    # C.RPIN.MAX_NUM_OBJS = 4
    C.freeze()

    cache_name = 'figures/' + C.DATA_ROOT.split('/')[2] + '/'
    if args.predictor_init:
        cache_name += args.predictor_init.split('/')[-2]
    output_dir = os.path.join(C.OUTPUT_DIR, cache_name)

    if args.eval == 'plan':
        assert 'reasoning' in C.DATA_ROOT
        assert num_gpus == 1, 'multi-gpu support is not avaialbe for planning tasks'
        model = eval(args.predictor_arch + '.Net')()
        model.to(torch.device('cuda'))
        model = torch.nn.DataParallel(
            model, device_ids=[0]
        )
        # load prediction model
        cp = torch.load(args.predictor_init, map_location='cuda:0')
        model.load_state_dict(cp['model'])

        tester = PlannerPHYRE(
            device=torch.device(f'cuda'),
            num_gpus=1,
            model=model,
            output_dir=output_dir,
        )
        tester.test(args.start_id, args.end_id)
        return

    # --- setup data loader
    print('initialize dataset')
    split_name = 'test'
    val_set = eval(f'{C.DATASET_ABS}')(data_root=C.DATA_ROOT, split=split_name, image_ext=C.RPIN.IMAGE_EXT)
    batch_size = 1 if C.RPIN.VAE else C.SOLVER.BATCH_SIZE * num_gpus
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16)

    model = eval(args.predictor_arch + '.Net')()
    model.to(torch.device('cuda'))
    model = torch.nn.DataParallel(
        model, device_ids=list(range(args.gpus.count(',') + 1))
    )
    cp = torch.load(args.predictor_init, map_location=f'cuda:0')
    model.load_state_dict(cp['model'])
    tester = PredEvaluator(
        device=torch.device('cuda'),
        val_loader=val_loader,
        num_gpus=num_gpus,
        model=model,
        num_plot_image=args.plot_image,
        output_dir=output_dir,
    )
    tester.test()


if __name__ == '__main__':
    main()
