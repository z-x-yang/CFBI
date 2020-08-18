import sys
sys.path.append('.')
sys.path.append('..')
from networks.engine.eval_manager import Evaluator
import importlib

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test CFBI")
    parser.add_argument('--gpu_id', type=int, default=7)
    parser.add_argument('--config', type=str, default='configs.resnet101_cfbi')
    parser.add_argument('--ckpt_path', type=str, default='test')
    args = parser.parse_args()
    config = importlib.import_module(args.config)
    cfg = config.cfg
    
    cfg.TEST_GPU_ID = args.gpu_id
    cfg.TEST_DATASET = 'test'
    cfg.TEST_CKPT_PATH = args.ckpt_path
    cfg.TEST_MULTISCALE = [0.5, 1]
    cfg.TEST_FLIP = True

    evaluator = Evaluator(cfg=cfg)
    evaluator.evaluating()

if __name__ == '__main__':
    main()

