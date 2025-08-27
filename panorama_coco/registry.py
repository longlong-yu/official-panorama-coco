"""
@author: longlong.yu
@email:  longlong.yu@hdu.edu.cn
@date:   2023-12-20
@desc:   
"""
import mmengine.registry as mmengine_registry
from mmengine.registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner', parent=mmengine_registry.RUNNERS)
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor', parent=mmengine_registry.RUNNER_CONSTRUCTORS
)
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop', parent=mmengine_registry.LOOPS)
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', 
    parent=mmengine_registry.HOOKS,
    locations=['panorama_coco.engine.hooks'], 
)

# # manage data-related modules
DATASETS = Registry(
    'dataset', 
    parent=mmengine_registry.DATASETS,
    locations=['panorama_coco.datasets'],
)
DATA_SAMPLERS = Registry('data sampler', parent=mmengine_registry.DATA_SAMPLERS)
TRANSFORMS = Registry(
    'transform', 
    parent=mmengine_registry.TRANSFORMS, 
    locations=['panorama_coco.datasets.transforms']
)

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry(
    'model', 
    parent=mmengine_registry.MODELS, 
    locations=['panorama_coco.models']
)
# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry('model_wrapper', parent=mmengine_registry.MODEL_WRAPPERS)
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer', parent=mmengine_registry.WEIGHT_INITIALIZERS
)

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry('optimizer', parent=mmengine_registry.OPTIMIZERS)
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry('optim_wrapper', parent=mmengine_registry.OPTIM_WRAPPERS)
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer constructor', parent=mmengine_registry.OPTIM_WRAPPER_CONSTRUCTORS
)
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler', parent=mmengine_registry.PARAM_SCHEDULERS
)
# manage all kinds of metrics
METRICS = Registry(
    'metric', 
    parent=mmengine_registry.METRICS,
    locations=['panorama_coco.evaluation.metrics'] 
)

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util', 
    parent=mmengine_registry.TASK_UTILS,
    locations=['panorama_coco.models.task_modules'] 
)

# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=mmengine_registry.VISUALIZERS,
    locations=['panorama_coco.visualization'] 
)
# manage visualizer backend
VISBACKENDS = Registry('vis_backend', parent=mmengine_registry.VISBACKENDS)

# manage logprocessor
LOG_PROCESSORS = Registry('log_processor', parent=mmengine_registry.LOG_PROCESSORS)
