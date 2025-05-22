import jax
import jax.numpy as jnp
from paxml import checkpoints, learners, tasks_lib
from praxis import base_layer, optimizers, pax_fiddle, py_utils, schedules
from vila import coca_vila, coca_vila_configs
from paxml import train_states


NestedMap = py_utils.NestedMap

_PRE_CROP_SIZE = 272
_IMAGE_SIZE = 224
_MAX_TEXT_LEN = 64
_TEXT_VOCAB_SIZE = 64000

_ZSL_QUALITY_PROMPTS = [
    ['good image', 'bad image'],
    ['good lighting', 'bad lighting'],
    ['good content', 'bad content'],
    ['good background', 'bad background'],
    ['good foreground', 'bad foreground'],
    ['good composition', 'bad composition'],
]


def load_vila_model(
    ckpt_dir,
):
  """Loads the VILA model from checkpoint directory.

  Args:
    ckpt_dir: The path to checkpoint directory

  Returns:
    VILA model, VILA model states
  """
  coca_config = coca_vila_configs.CocaVilaConfig()
  coca_config.model_type = coca_vila.CoCaVilaRankBasedFinetune
  coca_config.decoding_max_len = _MAX_TEXT_LEN
  coca_config.text_vocab_size = _TEXT_VOCAB_SIZE
  model_p = coca_vila_configs.build_coca_vila_model(coca_config)
  model_p.model_dims = coca_config.model_dims
  model = model_p.Instantiate()

  dummy_batch_size = 4  # For initialization only
  text_shape = (dummy_batch_size, 1, _MAX_TEXT_LEN)
  image_shape = (dummy_batch_size, _IMAGE_SIZE, _IMAGE_SIZE, 3)
  input_specs = NestedMap(
      ids=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.int32),
      image=jax.ShapeDtypeStruct(shape=image_shape, dtype=jnp.float32),
      paddings=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.float32),
      # For initialization only
      labels=jax.ShapeDtypeStruct(shape=text_shape, dtype=jnp.float32),
      regression_labels=jax.ShapeDtypeStruct(
          shape=(dummy_batch_size, 10), dtype=jnp.float32
      ),
  )
  prng_key = jax.random.PRNGKey(123)
  prng_key, _ = jax.random.split(prng_key)
  vars_weight_params = model.abstract_init_with_metadata(input_specs)

  # `learner` is only used for initialization.
  learner_p = pax_fiddle.Config(learners.Learner)
  learner_p.name = 'learner'
  learner_p.optimizer = pax_fiddle.Config(
      optimizers.ShardedAdafactor,
      decay_method='adam',
      lr_schedule=pax_fiddle.Config(schedules.Constant),
  )
  learner = learner_p.Instantiate()

  train_state = tasks_lib.create_state_unpadded_shapes(
      vars_weight_params, discard_opt_states=False, learners=[learner]
  )

  model_states = checkpoints.restore_checkpoint(train_state , ckpt_dir)
  assert model_states is not None, "Checkpoint restore failed"

  return model, model_states