import fiddle as fdl
from paxml import experiment_registry
from paxml import tasks_lib
from paxml.contrib.gpu.scripts_gpu.configs import GPT126MBase
from paxml.tasks.lm.params.lm_cloud import SyntheticDataset
from praxis import layers
from praxis import pax_fiddle
from praxis.layers import transformers

@experiment_registry.register
class GPT5BSynthetic(GPT126MBase, SyntheticDataset):

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
  

