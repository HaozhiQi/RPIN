import os
import time
import torch
import shutil
import random
import argparse
import numpy as np

from torch.utils.data import DataLoader

from rpin.models.dqn import ResNet18
from rpin.utils.config import _C as C
from rpin.utils.misc import tprint, pprint
from rpin.datasets.vid import VidPHYRECls
from rpin.utils.logger import setup_logger, git_diff_config


take_idx = [0, 3, 6, 9]  # take the first and 3rd, 6th, and 9th frame as input to the classifier
is_gt = [0, 0, 0, 0]


def test(data_loader, model):
    p_correct, n_correct, p_total, n_total = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(data_loader):
            gt, pred, labels = data_tuple
            labels = labels.squeeze(1).to('cuda')
            data = []
            for i, idx in enumerate(take_idx):
                data.append(gt[:, [idx]] if is_gt[i] else pred[:, [idx]])
            data = torch.cat(data, dim=1)
            data = data.long().to('cuda')
            pred = model(data)
            pred = pred.sigmoid() >= 0.5
            p_correct += ((pred == labels)[labels == 1]).sum().item()
            n_correct += ((pred == labels)[labels == 0]).sum().item()
            p_total += (labels == 1).sum().item()
            n_total += (labels == 0).sum().item()
    return p_correct / p_total, n_correct / n_total


def train(train_loader, test_loader, model, optim, scheduler, logger, output_dir):
    max_iters = C.SOLVER.MAX_ITERS
    model.train()

    losses = []
    acc = [0, 0, 0, 0]
    test_accs = []
    last_time = time.time()
    cur_update = 0
    while True and cur_update < max_iters:
        for batch_idx, data_tuple in enumerate(train_loader):
            if cur_update >= max_iters:
                break
            model.train()

            p_gt, p_pred, n_gt, n_pred = data_tuple
            labels = torch.cat([torch.ones(p_gt.shape[0]), torch.zeros(n_gt.shape[0])]).to('cuda')
            p_data = []
            n_data = []
            for i, idx in enumerate(take_idx):
                p_data.append(p_gt[:, [idx]] if is_gt[i] else p_pred[:, [idx]])
                n_data.append(n_gt[:, [idx]] if is_gt[i] else n_pred[:, [idx]])
            p_data = torch.cat(p_data, dim=1)
            n_data = torch.cat(n_data, dim=1)
            data = torch.cat([p_data, n_data])

            data = data.long().to('cuda')
            optim.zero_grad()

            pred = model(data)
            loss = model.ce_loss(pred, labels)

            pred = pred.sigmoid() >= 0.5
            acc[0] += ((pred == labels)[labels == 1]).sum().item()
            acc[1] += ((pred == labels)[labels == 0]).sum().item()
            acc[2] += (labels == 1).sum().item()
            acc[3] += (labels == 0).sum().item()

            loss.backward()
            optim.step()
            scheduler.step()
            losses.append(loss.mean().item())

            cur_update += 1
            speed = (time.time() - last_time) / cur_update
            eta = (max_iters - cur_update) * speed / 3600
            info = f'Iter: {cur_update} / {max_iters}, eta: {eta:.2f}h ' \
                   f'p acc: {acc[0] / acc[2]:.4f} n acc: {acc[1] / acc[3]:.4f}'
            tprint(info)

            if (cur_update + 1) % C.SOLVER.VAL_INTERVAL == 0:
                pprint(info)
                fpath = os.path.join(output_dir, 'last.ckpt')
                torch.save(
                    dict(
                        model=model.state_dict(), optim=optim.state_dict(), done_batches=cur_update + 1,
                        scheduler=scheduler and scheduler.state_dict(),
                    ), fpath
                )

                p_acc, n_acc = test(test_loader, model)
                test_accs.append([p_acc, n_acc])
                model.train()
                acc = [0, 0, 0, 0]
                for k in range(2):
                    info = ''
                    for test_acc in test_accs:
                        info += f'{test_acc[k] * 100:.1f} / '
                    logger.info(info)


def arg_parse():
    parser = argparse.ArgumentParser(description='PHYRE Classifier Parameters')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--gpus', type=str, default='',
                        help='specification for GPU, this model only support one GPU for now')
    parser.add_argument('--output', type=str, help='output name')
    parser.add_argument('--seed', type=int, help='set random seed use this command', default=0)
    return parser.parse_args()


def main():
    # this wrapper file contains the following procedure:
    # 1. setup training environment
    # 2. setup config
    # 3. setup logger
    # 4. setup model
    # 5. setup optimizer
    # 6. setup dataset
    args = arg_parse()
    rng_seed = args.seed
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        num_gpus = torch.cuda.device_count()
    else:
        assert NotImplementedError

    # ---- setup config files
    C.merge_from_file(args.cfg)
    C.SOLVER.BATCH_SIZE *= num_gpus
    C.SOLVER.BASE_LR *= num_gpus
    C.freeze()
    data_root = C.DATA_ROOT
    output_dir = os.path.join(C.OUTPUT_DIR, 'PHYRE_1fps_p100n400', args.output)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.cfg, os.path.join(output_dir, 'config.yaml'))

    # ---- setup logger
    logger = setup_logger('PCLS', output_dir)
    print(git_diff_config(args.cfg))

    # ---- setup model
    model = ResNet18(len(take_idx))
    model.to(torch.device('cuda'))

    # ---- setup optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=C.SOLVER.BASE_LR,
        weight_decay=C.SOLVER.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=C.SOLVER.MAX_ITERS)

    # ---- setup dataset in the last, and avoid non-deterministic in data shuffling order
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    kwargs = {'pin_memory': True, 'num_workers': 16}
    train_set = VidPHYRECls(data_root=data_root, split='train')
    test_set = VidPHYRECls(data_root=data_root, split='test')
    train_loader = DataLoader(train_set, batch_size=C.SOLVER.BATCH_SIZE // 2, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=C.SOLVER.BATCH_SIZE, shuffle=False, **kwargs)
    print(f'size: train {len(train_loader)} / test {len(test_loader)}')
    train(train_loader, test_loader, model, optim, scheduler, logger, output_dir)


if __name__ == '__main__':
    main()
