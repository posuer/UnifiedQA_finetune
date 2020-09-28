import os
import functools
import json
import logging
import argparse

import torch #must load torch first, otherwise run this command first: export LD_PRELOAD=path/to/miniconda3/lib/libgomp.so 
import tensorflow as tf
import transformers
import t5
import tensorflow_datasets as tfds

from hf_model import HfPyTorchModel

logger = logging.getLogger(__name__)

# socialiqa task preprocessing for T5
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
  ds = ds.map(lambda *ex: dict(zip(["question", "answer"], ex))) #"answers", "context",
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
                 ["question: ", normalize_text(ex["question"])]),
                 
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
        metric_fns=[t5.evaluation.metrics.sequence_accuracy],
        # Not required, but helps for mixing and auto-caching.
        num_input_examples=num_sqa_examples
    )

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")

    parser.add_argument('--model_path', default="models/unifiedqa_trained/11B/model.ckpt-1100500.index") 
    parser.add_argument('--model_size', default="11b")
    
    parser.add_argument("--task_name", default="socialiqa", type=str, help="The name of the task to train selected in T5 registered tasks")
    parser.add_argument("--output_dir", default="output/socialiqa/", type=str, help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")

    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--max_target_length", default=10, type=int, help="The maximum total target sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",)
    
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    #parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=300, help="Save checkpoint every X updates steps.")
    parser.add_argument("--only_save_best_ckpt", action="store_true", help="only save best and last checkpoints, otherwise save all checkpoints on 'save_steps'")
    parser.add_argument("--seed", type=int, default=2020, help="random seed for initialization")
    
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    args = parser.parse_args()

    # if (
    #     os.path.exists(args.output_dir)
    #     and os.listdir(args.output_dir)
    #     and args.do_train
    #     and not args.overwrite_output_dir
    # ):
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
    #             args.output_dir
    #         )
    #     )

   
    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
   
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,# if args.local_rank in [-1, 0] else logging.WARN,
        filename=f'{args.output_dir}run.log',
        filemode='w',
    )
    logger.info("Training/evaluation parameters %s", args)
    
    # Register socialiqa task to T5
    register_task() 
    # Present data example
    nq_task = t5.data.TaskRegistry.get(args.task_name)
    ds = nq_task.get_dataset(split="test", sequence_length={"inputs": args.max_seq_length, "targets": args.max_target_length})
    logger.info("A few preprocessed validation examples...")
    for ex in tfds.as_numpy(ds.take(2)):
        logger.info(ex)

    # Load pretrained model
    model = HfPyTorchModel(args, args.model_path, args.model_size, args.output_dir, args.device)

    if args.do_train:        
        # fine-tuning
        model.train(
            args,
            mixture_or_task_name=args.task_name, # "socialiqa",
            #steps=1000,
            save_steps=args.save_steps,# 100,
            sequence_length={"inputs": args.max_seq_length, "targets": args.max_target_length},
            split="train",
            batch_size=args.batch_size,
            optimizer=functools.partial(transformers.AdamW, lr=args.learning_rate),
        )
    if args.do_eval:
        
        model.eval(
            "socialiqa",
            sequence_length={"inputs": args.max_seq_length, "targets": args.max_target_length},
            batch_size=args.batch_size,
            split="dev",
            checkpoint_steps="all"
        )

    if args.do_test:
        model.eval(
            "socialiqa",
            sequence_length={"inputs": args.max_seq_length, "targets": args.max_target_length},
            batch_size=args.batch_size,
            split="test",
            checkpoint_steps="all"
        )
    