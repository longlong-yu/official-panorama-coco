import os
from mmengine.config import Config
from mmengine.runner import Runner
import sys

from numpy import mod
sys.path.append('.')


# change TORCH_HOME for pretrained model
os.environ['TORCH_HOME'] = '~/apps/panorama-h2rbox/.cache/torch'

def val_models(config_f, model_fs):
    """ Evaluate each model. """
    config_f = config_f.strip()
    model_fs = [item.strip() for item in model_fs.split(',')]
     
    config = Config.fromfile(config_f)
    for model in model_fs:
        config.load_from = model
        config.work_dir = os.path.dirname(model)
        runner = Runner.from_cfg(config)
        runner.val()

def val_epochs(config_f, model_dir, n):
    """ Evaluate each epoch. """
    config_f = config_f.strip()
    model_dir = model_dir.strip()
    n = int(n)
    
    config = Config.fromfile(config_f)
    runner = Runner.from_cfg(config)
    for i in range(n):
        fn = f'{model_dir}/epoch_{i+1}.pth'
        if not os.path.exists(fn):
            continue
        
        runner.load_checkpoint(
            fn,
            map_location='cpu',
        )
        runner.train_loop._epoch = i + 1
        runner.val()
  

if __name__ == '__main__':
    # mode: val_models
    mode = sys.argv[1].strip()
    args = sys.argv[2:]
    
    if mode == 'val_models':
        val_models(*args)
    elif mode == 'val_epochs':
        val_epochs(*args)
    
    exit(0)
