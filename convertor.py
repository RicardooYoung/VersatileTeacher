import torch
from models.yolo import Model
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save, ema_update, non_max_suppression, generate_mask_from_pred, generate_labels, generate_mask_from_labels, cal_thres)
import yaml
from copy import deepcopy

cfg = ['models/yolov5l_da.yaml', 'models/yolov5l.yaml']
weights = ['yolov5l_cityscapes_mt.pt', 'yolov5l_cityscapes.pt']
nc = 8
resume = False
model = []
hyp = check_yaml('data/hyps/hyp.yaml')
if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
for i in range(len(cfg)):
        pretrained = weights[i].endswith('.pt')
        if pretrained:
            ckpt = torch.load(weights[i], map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
            model.append(Model(cfg[i] or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')))  # create
            exclude = ['anchor'] if (cfg[i] or hyp.get('anchors')) and not resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, model[i].state_dict(), exclude=exclude)  # intersect
            model[i].load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(model[i].state_dict())} items from {weights[i]}\n')  # report
        else:
            model.append(Model(cfg[i], ch=3, nc=nc, anchors=hyp.get('anchors')))  # create
ema_update(model[1], model[0], alpha=0.0, disp=True)
epoch = 10

ckpt = {
                        'epoch': epoch,
                        'model': deepcopy(model[1]).half(),
                        'ema': None,
                        'opt': vars(opt)
        }

torch.save(ckpt, 'test.pt')