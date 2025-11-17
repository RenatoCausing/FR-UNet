import argparse
import json
from pathlib import Path
import torch
from bunch import Bunch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from tester import Tester
from utils import losses
from utils.helpers import get_instance, get_torch_device
from utils.metrics import save_confusion_matrix


def main(data_path, weight_path, CFG, show=False, device=None, test_mode='test'):
    device = get_torch_device(device)
    checkpoint = torch.load(weight_path, map_location=device)
    CFG_ck = checkpoint['config']
    test_dataset = vessel_dataset(data_path, mode=test_mode)
    test_loader = DataLoader(test_dataset, 1,
                             shuffle=False,  num_workers=16, pin_memory=True)
    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG_ck, device=device)
    test = Tester(model, loss, CFG, checkpoint, test_loader, data_path, show, device=device)
    return test.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", default="/home/lwt/data_pro/vessel/DRIVE", type=str,
                        help="the path of dataset")
    parser.add_argument("-wp", "--wetght_path", default="pretrained_weights/DRIVE/checkpoint-epoch40.pth", type=str,
                        help='the path of wetght.pt')
    parser.add_argument("--show", help="save predict image",
                        required=False, default=False, action="store_true")
    parser.add_argument("-o", "--output_dir", default=None, type=str,
                        help="directory to dump metrics json and confusion matrix plot")
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                        help='Force testing onto a specific device (default: auto-detect).')
    args = parser.parse_args()
    yaml = YAML(typ='safe')
    yaml.pure = True
    with open('config.yaml', encoding='utf-8') as file:
        CFG = Bunch(yaml.load(file))
    device_pref = None if args.device == 'auto' else args.device
    results = main(args.dataset_path, args.wetght_path, CFG, args.show, device=device_pref)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'test_metrics.json', 'w', encoding='utf-8') as fp:
            json.dump(results, fp, indent=2)
        confusion = results.get('confusion')
        if confusion:
            save_confusion_matrix(confusion, output_dir / 'confusion_matrix.png')
            save_confusion_matrix(confusion, output_dir / 'confusion_matrix_normalized.png', normalize=True)
    print(json.dumps(results, indent=2))
