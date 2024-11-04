import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

from torch.utils.tensorboard import SummaryWriter  # TensorBoard 임포트

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    
    parser.add_argument('--aug_methods', nargs="+", default=[])

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, aug_methods=[], log_dir = './tensorboard_logs'):

    # TensorBoard 로그 설정
    writer = SummaryWriter(log_dir=log_dir)

    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        aug_methods=aug_methods
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    aug_methods_str = '_'.join(aug_methods) if aug_methods else 'no_augmentation'
    best_loss = float('inf')  # 최상의 손실을 무한대로 초기화

    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                # TensorBoard에 손실 기록
                writer.add_scalar('Loss/{aug_methods_str}/train', loss_val, epoch * num_batches + pbar.n)

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()

        mean_loss = epoch_loss / num_batches
        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            mean_loss, timedelta(seconds=time.time() - epoch_start)))

        # 현재 에폭의 손실이 최상의 손실보다 낮은지 확인
        if mean_loss < best_loss:
            best_loss = mean_loss
            
            # 최상의 모델 저장
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'{aug_methods_str}_best_model.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            print('최상의 모델이 저장되었습니다. 손실: {:.4f}'.format(best_loss))

    # 훈련이 끝난 후 최신 모델 저장, 파일 이름에 augmentation 방법 포함
    final_model_name = f'model_with_{aug_methods_str}.pth'
    final_model_path = osp.join(model_dir, final_model_name)
    torch.save(model.state_dict(), final_model_path)
    print('최종 모델이 저장되었습니다: {}'.format(final_model_path))

    # TensorBoard 종료
    writer.close()

def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)