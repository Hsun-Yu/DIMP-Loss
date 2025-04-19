#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This code is based on the script https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
# %%
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    RobertaForSequenceClassification,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import wandb
from datetime import datetime
import json
import pandas as pd

from models import MyBertForSequenceClassification, MyNewBertForSequenceClassification, MyNewRobertaForSequenceClassification

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    augmented_train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the augmented training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    # test_file2: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."}) 
    
    name: Optional[str] = field(
        default=None, metadata={"help":"wandb run name"}
    )
    
    use_wandb_dataset: Optional[bool] = field(
        default=False, metadata={"help":"use wandb dataset"}
    )
    
    wandb_dataset: Optional[str] = field(
        default=None, metadata={"help":"wandb dataset name"}
    )
    
    wandb_augmented_dataset: Optional[str] = field(
        default=None, metadata={"help":"wandb augmented dataset name"}
    )
    
    wandb_generalization_dataset: Optional[str] = field(
        default=None, metadata={"help":"wandb generalization dataset name"}
    )
    
    save_model: Optional[bool] = field(
        default=True, metadata={"help":"save model"}
    )
    
    twomodelloss_wandb_model2: Optional[str] = field(
        default=None, metadata={"help":"use wandb model"}
    )
    
    twomodelloss_wandb_model3: Optional[str] = field(
        default=None, metadata={"help":"use wandb model"}
    )
    
    contrastive_learning: Optional[bool] = field(
        default=False, metadata={"help":"use contrastive learning"}
    )
    
    temperature: Optional[float] = field(
        default=0.05, metadata={"help":"temperature for contrastive learning"}
    )
    
    contrastive_learning_weight: Optional[float] = field(
        default=0.1, metadata={"help":"contrastive learning weight"}
    )
    # def __post_init__(self):
    #     if self.dataset_name is not None:
    #         pass
    #     elif self.train_file is None or self.validation_file is None:
    #         raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
    #     else:
    #         train_extension = self.train_file.split(".")[-1]
    #         assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #         validation_extension = self.validation_file.split(".")[-1]
    #         assert (
    #             validation_extension == train_extension
    #         ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    use_wandb_model: bool = field(
        default=False,
        metadata={"help": "use wandb model"},
    )
    problem_type: Optional[str] = field(
        default="single_label_classification", metadata={"help":"classification or regression"}
    )

# %%
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    use_json_config_name = None
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        use_json_config_name = os.path.abspath(sys.argv[1])
    elif len(sys.argv) == 1:
        # Only read the config file
        model_args, data_args, training_args = parser.parse_json_file(json_file="config.json")
        use_json_config_name = "config.json"
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if len(data_args.wandb_dataset) > 64:
        cut_part = data_args.wandb_dataset[-63:]
        tags = [cut_part, model_args.model_name_or_path]
    else:
        tags = [data_args.wandb_dataset, model_args.model_name_or_path]
    
    tags.append(model_args.problem_type)
    
    # check load_best_model_at_end
    if not training_args.load_best_model_at_end:
        tags.append("end")
    
    # remove empty tags
    tags = [tag for tag in tags if tag != "" and tag is not None]
    
    run = wandb.init(
        name=data_args.name,
        project="epfl_ml_project2",
        anonymous="allow",
        tags=tags,
        job_type="run",
        save_code=True,
    )
    output_dir = os.path.join(training_args.output_dir, run.name + "_" + datetime.now().strftime("%m%d%Y%H%M"))
    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if use_json_config_name:
        # save the config file to wandb
        art = wandb.Artifact(name=run.name + "_" + datetime.now().strftime("%m%d%Y%H%M") + "_config", type="config")
        art.add_file(use_json_config_name)
        run.log_artifact(art)
    else:
        # read input arguments and save to json file
        args = parser.parse_args()
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(args.__dict__, f, indent=4)
        # save the config file to wandb
        art = wandb.Artifact(name=run.name + "_" + datetime.now().strftime("%m%d%Y%H%M") + "_config", type="config")
        art.add_file(config_path)
        run.log_artifact(art)
        
    if data_args.use_wandb_dataset:
        # download the dataset from wandb
        dataset = run.use_artifact(data_args.wandb_dataset)
        dataset_dir = dataset.download()
        if data_args.train_file is not None:
            if type(data_args.train_file) == list:
                data_args.train_file = [os.path.join(dataset_dir, f) for f in data_args.train_file]
            else:
                data_args.train_file = os.path.join(dataset_dir, data_args.train_file)
        else:
            # find the file name of the directory and include 'train'
            data_args.train_file = os.path.join(dataset_dir, [f for f in os.listdir(dataset_dir) if 'train' in f][0])
        
        if data_args.wandb_augmented_dataset:
            augmented_dataset = run.use_artifact(data_args.wandb_augmented_dataset)
            augmented_dataset_dir = augmented_dataset.download()
            if data_args.augmented_train_file is not None:
                if type(data_args.augmented_train_file) == list:
                    data_args.augmented_train_file = [os.path.join(augmented_dataset_dir, f) for f in data_args.augmented_train_file]
                else:
                    data_args.augmented_train_file = os.path.join(augmented_dataset_dir, data_args.augmented_train_file)
            else:
                # find the file name of the directory and include 'generated', 'augmented'
                da_list = [f for f in os.listdir(augmented_dataset_dir) if ('generated' in f) or ('augmented' in f)]
                if len(da_list) == 0:
                    data_args.augmented_train_file = os.path.join(augmented_dataset_dir, [f for f in os.listdir(augmented_dataset_dir) if 'train' in f][0])
                else:
                    data_args.augmented_train_file = os.path.join(augmented_dataset_dir, da_list)

        
        
        if data_args.validation_file is not None:
            if type(data_args.validation_file) == list:
                data_args.validation_file = [os.path.join(dataset_dir, f) for f in data_args.validation_file]
            else:
                data_args.validation_file = os.path.join(dataset_dir, data_args.validation_file)
        else:
            # find the file name of the directory and include 'dev' or 'valid'
            data_args.validation_file = os.path.join(dataset_dir, [f for f in os.listdir(dataset_dir) if 'dev' in f or 'valid' in f or 'val' in f][0])
        if data_args.test_file is not None:
            if type(data_args.test_file) == list:
                data_args.test_file = [os.path.join(dataset_dir, f) for f in data_args.test_file]
            else:
                data_args.test_file = os.path.join(dataset_dir, data_args.test_file)
        else:
            # find the file name of the directory and include 'test'
            data_args.test_file = os.path.join(dataset_dir, [f for f in os.listdir(dataset_dir) if 'test' in f][0])
        
        if data_args.wandb_generalization_dataset:
            gen_dataset = run.use_artifact(data_args.wandb_generalization_dataset)
            gen_dataset_dir = gen_dataset.download()
            # find all the files in the directory
            gen_files = [os.path.join(gen_dataset_dir, f) for f in os.listdir(gen_dataset_dir)]
            # add generalization files to the data_args.test_file
            data_args.test_file = [data_args.test_file] + gen_files
    
    if model_args.use_wandb_model:
        # download the model from wandb
        model = run.use_artifact(model_args.model_name_or_path)
        model_dir = model.download()
        model_args.model_name_or_path = model_dir
    
    if data_args.twomodelloss_wandb_model2:
        model2 = run.use_artifact(data_args.twomodelloss_wandb_model2)
        model2_dir = model2.download()
        data_args.twomodelloss_wandb_model2 = model2_dir
        
    if data_args.twomodelloss_wandb_model3:
        model3 = run.use_artifact(data_args.twomodelloss_wandb_model3)
        model3_dir = model3.download()
        data_args.twomodelloss_wandb_model3 = model3_dir

    run.config.update(model_args)
    run.config.update(data_args)
    run.config.update(training_args)
    run.log_code(".")
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    if data_args.wandb_augmented_dataset:
        data_files = {"train": data_args.train_file, 
                    "augmented_train": data_args.augmented_train_file,
                    "validation": data_args.validation_file}
    else:
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            if type(data_args.test_file) == str:
                test_extension = data_args.test_file.split(".")[-1]
            else:
                test_extension = data_args.test_file[0].split(".")[-1]
            assert (
                test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            
            if type(data_args.test_file) == str:
                data_files["test"] = data_args.test_file
            else:
                data_files["test"] = data_args.test_file[0]
                for i in range(1, len(data_args.test_file)):
                    data_files[f"test_{i}"] = data_args.test_file[i]
            
        else:
            raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")
    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from local json files
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    # Labels
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique("label") + raw_datasets["validation"].unique("label")
        label_list = list(set(label_list))
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
    logger.info(f"num_labels: {num_labels}")
    logger.info(f"label_list: {label_list}")

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.problem_type = model_args.problem_type
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # check if the model is a bert model or roberta model
    if config.model_type == "bert":
        model = MyNewBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        model = MyNewRobertaForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

    model.tokenizer = tokenizer

    # for contrastive learning
    model.contrastive_learning = data_args.contrastive_learning
    model.temperature = data_args.temperature
    model.contrastive_learning_weight = data_args.contrastive_learning_weight
    # Model 2
    if data_args.twomodelloss_wandb_model2:
        model2_config = AutoConfig.from_pretrained(
            data_args.twomodelloss_wandb_model2,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model2_tokenizer = AutoTokenizer.from_pretrained(
            data_args.twomodelloss_wandb_model2,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        print(data_args.twomodelloss_wandb_model2)
        
        if model2_config.model_type == "bert":
            model2 = MyBertForSequenceClassification.from_pretrained(
                data_args.twomodelloss_wandb_model2,
                from_tf=bool(".ckpt" in data_args.twomodelloss_wandb_model2),
                config=model2_config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )
        else:
            model2 = RobertaForSequenceClassification.from_pretrained(
                data_args.twomodelloss_wandb_model2,
                from_tf=bool(".ckpt" in data_args.twomodelloss_wandb_model2),
                config=model2_config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )
        
        model.model2 = model2
        model.model2_tokenizer = model2_tokenizer
        
    # Model 3
    if data_args.twomodelloss_wandb_model3:
        model3_config = AutoConfig.from_pretrained(
            data_args.twomodelloss_wandb_model3,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model3_tokenizer = AutoTokenizer.from_pretrained(
            data_args.twomodelloss_wandb_model3,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        print(data_args.twomodelloss_wandb_model3)
        
        if model3_config.model_type == "bert":
            model3 = MyBertForSequenceClassification.from_pretrained(
                data_args.twomodelloss_wandb_model3,
                from_tf=bool(".ckpt" in data_args.twomodelloss_wandb_model3),
                config=model3_config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )
        else:
            model3 = RobertaForSequenceClassification.from_pretrained(
                data_args.twomodelloss_wandb_model3,
                from_tf=bool(".ckpt" in data_args.twomodelloss_wandb_model3),
                config=model3_config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )
        
        model.model3 = model3
        model.model3_tokenizer = model3_tokenizer
    sentence1_key, sentence2_key = "sentence1", None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        # if we have augmented dataset, we need to combine the original dataset and augmented dataset
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        
        if data_args.wandb_augmented_dataset:
            # combine the original dataset and augmented dataset
            train_dataset = datasets.concatenate_datasets([raw_datasets["train"], raw_datasets["augmented_train"]])
        else:
            train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.test_file is not None:
        generalization_dataset = []
        if type(data_args.test_file) == str:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                predict_dataset = predict_dataset.select(range(max_predict_samples))
        else:
            predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                predict_dataset = predict_dataset.select(range(max_predict_samples))
            
            for i in range(1, len(data_args.test_file)):
                generalization_dataset.append(raw_datasets["test_" + str(i)])
                if data_args.max_predict_samples is not None:
                    max_predict_samples = min(len(generalization_dataset[i-1]), data_args.max_predict_samples)
                    generalization_dataset[i-1] = generalization_dataset[i-1].select(range(max_predict_samples))
        
        
    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function    
    acc_metrics = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
    f1_metric = evaluate.load("f1")
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = acc_metrics .compute(predictions=preds, references=p.label_ids)
        f1_results = f1_metric.compute(predictions=preds, references=p.label_ids, average="macro")
        
        # cross entropy
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else preds
        # softmax
        preds = np.exp(preds) / np.exp(preds).sum(axis=1, keepdims=True)
        labels = p.label_ids
        cross_entropy = 0
        for i in range(len(labels)):
            cross_entropy += -np.log(preds[i][labels[i]])
        cross_entropy = cross_entropy / len(labels)
        
        result["f1"] = f1_results["f1"]
        result["cross_entropy"] = cross_entropy
        return result
    
    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        if data_args.save_model:
            model_dir = data_args.name
            model_art = wandb.Artifact(name=data_args.name, type="model")
            trainer.save_model(model_dir)
            model_art.add_dir(model_dir)
            run.log_artifact(model_art)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_datasets = [predict_dataset]
        if len(generalization_dataset) > 0:
            predict_datasets.extend(generalization_dataset)
        for index, predict_dataset in enumerate(predict_datasets):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict")
            metrics = predictions.metrics
            label_ids = predictions.label_ids
            predictions = np.squeeze(predictions.predictions) if is_regression else np.argmax(predictions.predictions, axis=1)
            labels = label_ids
            # map label and prediction to their string value
            predictions_text = [label_list[pred] for pred in predictions]
            labels_text = [label_list[label] for label in labels]
            
            # save predictions results to data frame
            result_df = pd.DataFrame({'input_text': predict_dataset[sentence1_key],
                               'label': labels_text, 
                               'prediction': predictions_text})
            if index == 0:
                result_df.to_csv(os.path.join(output_dir, f"predict_results.csv"))
                # save csv to wandb
                result_art = wandb.Artifact(name=f"{data_args.name}_predict_results", type="predict_results")
                result_art.add_file(os.path.join(output_dir, f"predict_results.csv"))
                run.log_artifact(result_art)
                # save to wandb
                run.log({"predict_results": wandb.Table(dataframe=result_df)})
                # save metric to wandb
                run.log(metrics)
                
                output_predict_file = os.path.join(output_dir, f"predict_results.txt")
                    
                # output submission file (only Id,Prediction)
                if data_args.test_file is not None:
                    output_test_file = os.path.join(output_dir, f"test_submission.txt")
                    with open(output_test_file, "w") as f:
                        f.write("Id,Prediction\n")
                        for i, pred in enumerate(predictions_text):
                            f.write(f"{i+1},{pred}\n")
                run.log_artifact(output_test_file)
                
                
            else:
                result_df.to_csv(os.path.join(output_dir, f"predict_results_{index}.csv"), index=False)
                # save csv to wandb
                result_art = wandb.Artifact(name=f"{data_args.name}_predict_results_{index}", type="predict_results")
                result_art.add_file(os.path.join(output_dir, f"predict_results_{index}.csv"))
                run.log_artifact(result_art)
                # save to wandb
                run.log({f"predict_results_{index}": wandb.Table(dataframe=result_df)})
                # save metric to wandb
                # modify metric key
                metrics = {k + f"_{index}": v for k, v in metrics.items()}
                run.log(metrics)
                
                output_predict_file = os.path.join(output_dir, f"predict_results_{index}.txt")
                
                # calculate accuracy for each class
                for i in range(len(label_list)):
                    class_text = label_list[i]
                    class_label_index = [i for i, x in enumerate(labels_text) if x == class_text]
                    class_prediction = [predictions[i] for i in class_label_index if predictions_text[i] == class_text]
                    class_label = [labels[i] for i in class_label_index if labels_text[i] == class_text]
                    if len(class_label) == 0:
                        class_accuracy = -1
                    else:
                        class_accuracy = len(class_prediction) / len(class_label)
                    key = 'accuracy_on_' + class_text + f"_{index}"
                    run.log({key: class_accuracy})


if __name__ == "__main__":
    main()
# %%
