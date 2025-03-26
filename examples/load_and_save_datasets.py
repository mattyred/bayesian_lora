
import os
import sys
import hydra
import torch as t
import logging
from tqdm import tqdm
from typing import Any
from omegaconf import DictConfig

from utils import dsets
from utils.loggers import setup_loggers
from utils.setup_llm import setup_llm

@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="example_usage",
)
def main(cfg: DictConfig):
    setup_loggers(cfg)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    model, tokenizer, gen_cfg = setup_llm(**cfg.llm)
    dset_class: dsets.ClassificationDataset = getattr(dsets, cfg.dset.name)
    dset = dset_class(tokenizer, add_space=cfg.llm.add_space)
    logging.info(f"Loading validation split")
    val_loader = dset.loader(
        is_sc=cfg.llm.is_sc,
        batch_size=cfg.dset.eval_bs,
        split=cfg.dset.eval_split,
        subset_size=-1, #MR: cfg.dset.eval_subset,
    )

    raw_samples = []
    tokenized_samples_matching_inputs = []
    tokenized_samples_model = []
    labels_all = []

    # Iterate over the validation loader
    logging.info(f"Saving validation split")
    for batch in tqdm(val_loader, disable=not cfg.use_tqdm, file=sys.stdout):
        # Assuming your batch is structured as (prompts, labels, extra)
        prompts, labels, _ = batch  
        raw_samples.extend(prompts)
        
        # Tokenize the raw samples again (if needed) using the same tokenizer settings
        batch_inputs = tokenizer(prompts, **cfg.tokenizer_run_kwargs)
        # Iterate over the batch dimension
        for i in range(len(prompts)):
            # Create a dictionary for the i-th sample
            sample_tokens = {key: val[i] for key, val in batch_inputs.items()}
            tokenized_samples_matching_inputs.append(sample_tokens)
        tokenized_samples_model.append(batch_inputs)

        if isinstance(labels, t.Tensor):
            labels_all.extend(labels.tolist())
        else:
            labels_all.extend(labels)

    # Save both raw and tokenized samples in a single dictionary
    data_to_save = {
        "raw": raw_samples,
        "tokenized_matched": tokenized_samples_matching_inputs,
        "tokenized": tokenized_samples_model,
        "labels": labels_all,
    }
    
    output_file = os.path.join(cfg.paths.output_dir, "val_samples.pth")
    t.save(data_to_save, output_file)
    logging.info(f"Validation split saved correctly")


if __name__ == "__main__":
    main()
