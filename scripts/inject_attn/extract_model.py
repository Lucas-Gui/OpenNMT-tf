import tensorflow as tf
from opennmt import models, config

def get_model(dir, configpath, model_type=models.GPT2Small):
    """Get, activate and return the last model in the directory"""
    model = model_type()
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(dir)).expect_partial()  # .assert_consumed()
    cfg = config.load_config([configpath], model.auto_config())
    model.initialize(data_config=cfg["data"], params = cfg["params"])

    return model

def get_dataset(model,path, mode='inference') -> tf.data.Dataset :
    if mode == "inference":
        ds = model.examples_inputter.make_inference_dataset(
            features_file=path,
            batch_size=1,
            batch_type="examples"
        )
    elif mode == "evaluation":
        ds = model.examples_inputter.make_evaluation_dataset(
            features_file=path,
            labels_file=None,
            batch_size=1,
            batch_type="examples"
        )[0]

    else :
        raise ValueError(mode + ' is not a valid mode.')
    return ds


