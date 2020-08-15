import os
import functools
import json
import argparse

import torch #must load torch first, otherwise run this command first: export LD_PRELOAD=path/to/miniconda3/lib/libgomp.so 
import tensorflow as tf
import transformers
import t5

from hf_model import HfPyTorchModel

sqa_tsv_path = {
    "train": "data/socialiqa/data_social_iqa_train.tsv",
    "dev": "data/socialiqa/data_social_iqa_dev.tsv",
    "test": "data/socialiqa/data_social_iqa_test.tsv",
}
sqa_counts_path = "data/socialiqa/data_social_iqa_counts.json"

def socialiqa_dataset_fn(split, shuffle_files=False):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(sqa_tsv_path[split])
  # Split each "<question>\t<answer>" example into (question, answer) tuple.
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Map each tuple to a {"question": ... "answer": ...} dict.
  ds = ds.map(lambda *ex: dict(zip(["question", "answer"], ex)))
  return ds

def socialiqa_preprocessor(ds):
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    return text

  def to_inputs_and_targets(ex):
    """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs":
             tf.strings.join(
                 ["trivia question: ", normalize_text(ex["question"])]),
        "targets": normalize_text(ex["answer"])
    }
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

def register_task():
    num_sqa_examples = json.load(tf.io.gfile.GFile(sqa_counts_path))

    t5.data.TaskRegistry.add(
        "socialiqa",
        # Supply a function which returns a tf.data.Dataset.
        dataset_fn=socialiqa_dataset_fn,
        splits=["train", "dev", "test"],
        # Supply a function which preprocesses text from the tf.data.Dataset.
        text_preprocessor=[socialiqa_preprocessor],
        # Lowercase targets before computing metrics.
        postprocess_fn=t5.data.postprocessors.lower_text, 
        # We'll use accuracy as our evaluation metric.
        metric_fns=[t5.evaluation.metrics.accuracy],
        # Not required, but helps for mixing and auto-caching.
        num_input_examples=num_sqa_examples
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default=None)
    args = parser.parse_args()

    register_task() #register task to T5

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(device)
    #models/unifiedqa_trained/models_large_model.ckpt-1101200.index
    model = HfPyTorchModel(args.model_path, "models/socialiqa/", device)

    # Evaluate the pre-trained checkpoint, before further fine-tuning
    model.eval(
        "socialiqa",
        sequence_length={"inputs": 128, "targets": 10},
        batch_size=128,
        split="dev"
    )

    # Run 1000 steps of fine-tuning
    model.train(
        mixture_or_task_name="socialiqa",
        steps=1000,
        save_steps=100,
        sequence_length={"inputs": 128, "targets": 10},
        split="train",
        batch_size=32,
        optimizer=functools.partial(transformers.AdamW, lr=1e-4),
    )

    model.eval(
        "socialiqa",
        sequence_length={"inputs": 128, "targets": 10},
        batch_size=128,
        split="dev"
    )