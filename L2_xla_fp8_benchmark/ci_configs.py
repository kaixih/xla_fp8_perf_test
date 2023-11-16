import fiddle as fdl
from paxml import experiment_registry
from paxml import tasks_lib
from paxml.contrib.gpu.scripts_gpu.configs import GPT126M
from paxml.tasks.lm.params.lm_cloud import SyntheticDataset
from praxis import layers
from praxis import pax_fiddle
from praxis.layers import transformers

@experiment_registry.register
class GPT5BSynthetic(GPT126M, SyntheticDataset):

  USE_REPEATED_LAYER = True
  ICI_MESH_SHAPE = [1, 8, 1]
  DCN_MESH_SHAPE = [1, 1, 1]
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM
  MAX_STEPS = 100

  PERCORE_BATCH_SIZE = 8

  NUM_LAYERS = 24
  NUM_HEADS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 16384
  DIMS_PER_HEAD = 128

  INIT_STD = 0.01
  SOFTMAX_INIT_STD = 0.01

  ## optimizer-related
  LEARNING_RATE = 1.6e-4

  ## lr schedule
  LR_COS_WARMUP = 115
  LR_COS_DECAY_START = LR_COS_WARMUP+1
  LR_COS_DECAY_END = 62500

  CHECKPOINT_EVERY_N_STEPS = 250
  SUMMARY_INTERVAL_STEPS = 10

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    if self.USE_FP8:
      self.CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
    task_p = super().task()

    model_p = task_p.model
    stacked_p = model_p.lm_tpl.stacked_transformer_tpl
    if issubclass(
        fdl.get_callable(stacked_p), transformers.StackedTransformerRepeated
    ):
      stacked_p = stacked_p.block

    stacked_p.input_dropout_prob = 0.1
    return task_p
  
@experiment_registry.register
class GPT175BSynthetic(GPT126M, SyntheticDataset):

  NUM_LAYERS = 96
  NUM_HEADS = 96
  MODEL_DIMS = 12288
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 50257
  USE_REPEATED_LAYER = True
  MAX_STEPS = 75000

  # Model configs
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = True
  ATTEN_LOGIT_CAP = -1.0  # Disable logits cap in atten

  # HPs
  WEIGHT_DECAY = 0.1
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  ADAM_EPSILON = 1e-8
  ADAM_CLIP_THRESHOLD = -1.0  # Disable Adam clip_threshold
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0
  LAYERNORM_EPSILON = 1e-5

  # In units of steps for BS1.5k
  LEARNING_RATE = 2e-5
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 265
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 108600
  LR_COS_MAX = 1.0
  LR_COS_MIN_RATIO = 0.1

  # Checkpoint
  EVAL_INTERVAL_STEPS = 100
  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10

  ## GPU-specific settings
  ICI_MESH_SHAPE = [1, 8, 1]
  DCN_MESH_SHAPE = [1, 1, 1]
  PERCORE_BATCH_SIZE = 6

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    return task_p

