# Copyright (C) 2023-24 Maxime Robeyns <dev@maximerobeyns.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example usage for Bayesian LoRA.
"""

import os
import sys
import peft
import hydra
import logging
import importlib
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor
from typing import Any
from omegaconf import DictConfig
from torch.func import jacrev, functional_call
from torchmetrics import Accuracy, CalibrationError
from transformers.modeling_outputs import ModelOutput

from bayesian_lora import (
    calculate_kronecker_factors,
    cholesky_decompose_small_factors,
    model_evidence,
    variance,
    stable_cholesky,
)
from utils import dsets
from utils.loggers import setup_loggers
from utils.setup_llm import setup_llm
from bayesian_lora.main import jacobian_mean


@hydra.main(
    version_base="1.3",
    config_path="configs",
    config_name="example_usage",
)
def main(cfg: DictConfig):
    #
    # 1. Load configuration from Hydra
    #
    device = "cuda:0"
    setup_loggers(cfg)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    map_param_path = f"{cfg.paths.output_dir}/MAP_params.pth"
    ll_path = f"{cfg.paths.output_dir}/ll.pth"
    kfac_path = f"{cfg.paths.output_dir}/kronecker_factors.pth"
    prior_path = f"{cfg.paths.output_dir}/prior_params.pth"

    # Make linearized predictions
    #del model
    t.cuda.empty_cache()
    logging.info("Doing linearized prediction")

    cfg.llm.use_quant = False  # because our gradient calcs don't support bnb
    cfg.llm.use_peft = False  # due to the quirk in loading PEFT models
    # cfg.llm.model_kwargs.attn_implementation = "sdpa"
    model, tokenizer, gen_cfg = setup_llm(**cfg.llm)
    model = peft.PeftModel.from_pretrained(model, map_param_path, is_trainable=True)
    model = model.to(device)
    dset_class: dsets.ClassificationDataset = getattr(dsets, cfg.dset.name)
    dset = dset_class(tokenizer, add_space=cfg.llm.add_space)
    val_loader = dset.loader(
        is_sc=cfg.llm.is_sc,
        batch_size=cfg.dset.eval_bs,
        split=cfg.dset.eval_split,
        subset_size=cfg.dset.eval_subset,
    )

    pred_mu = []
    pred_var = []
    pred_logits = []
    pred_samples = []

    total_loss = 0
    metric_kwargs = {"task": "multiclass", "num_classes": dset.n_labels}
    acc_metric = Accuracy(**metric_kwargs).to(device)
    ece_metric = CalibrationError(**metric_kwargs).to(device)

    def output_callback(outputs: ModelOutput) -> Tensor:
        """Post process model outputs.

        This function will be passed the results of model(**batch_inputs), and
        should return the relevant logits. For multiple-choice tasks, this is
        the class logits, but for full next-token prediction, this would just
        be all the logits.
        """
        # Get the last token for CausalLM
        logits = outputs.logits if cfg.llm.is_sc else outputs.logits[:, -1]
        # Select the logits corresponding to our target classes
        target_logits = logits[:, dset.target_ids.squeeze(-1)]
        return target_logits

    kfactors = t.load(kfac_path)
    factors = kfactors["factors"]
    priors = t.load(prior_path)
    s2 = priors["s2"]
    with t.no_grad():
        for batch in tqdm(val_loader, disable=not cfg.use_tqdm, file=sys.stdout):
            prompts, classes, _ = batch
            classes = classes.to(device)

            batch_inputs = tokenizer(prompts, **cfg.tokenizer_run_kwargs).to(device) 

            # Predict the output logit locations
            jacobian, f_mu = jacobian_mean(
                model, batch_inputs, output_callback=output_callback
            )
            pred_mu.append(f_mu.clone().cpu())

            # Predict the output logit variances
            f_var = variance(
                batch_inputs,
                jacobian,
                factors,
                s2,
                dset.n_labels,
                cfg.llm.peft.r,
                cfg.n_kfac,
                device,
            )
            pred_var.append(f_var.clone().cpu())

            # Sample logits from a Gaussian parametrised by f_mu, f_var
            L = stable_cholesky(f_var)
            samples = 100_000
            f_mu = f_mu.expand(samples, *f_mu.shape)
            L = L.to(f_mu.dtype) # MR: added conversion to f_mu dytpe
            eps = t.randn_like(f_mu).unsqueeze(-1).to(f_mu.dtype) # MR: added conversion to f_mu dype
            L = L.expand(samples, *L.shape) 
            eps = t.randn_like(f_mu).unsqueeze(-1) 
            logits = f_mu[..., None] + L @ eps
            # print('logits shape: ', logits.squeeze(-1).shape) # 100_000 x 4 x 5
            pred_samples.append(logits.squeeze(-1).cpu()) # 100_000 x 4 x 5
            logits = logits.squeeze(-1).mean(0)

            pred_logits.append(logits.cpu())
            total_loss += F.cross_entropy(logits, classes).item()
            acc_metric(logits, classes)
            ece_metric(logits, classes)

    loss = total_loss / len(val_loader)
    acc = acc_metric.compute().item()
    ece = ece_metric.compute().item()

    logging.info(f"NLL: {loss:.5f}, ACC: {acc:.5f}, ECE: {ece:.5f}")

    output_path = f"{cfg.paths.output_dir}/predicted_logits.pth"
    t.save(
        {"pred_mu": pred_mu, "pred_var": pred_var, "pred_logits": pred_logits, "pred_samples": pred_samples},
        output_path,
    )


if __name__ == "__main__":
    main()
