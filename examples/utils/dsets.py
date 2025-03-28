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
Convenience wrappers around classification datasets
"""
import torch as t

from abc import abstractmethod
from enum import Enum
from typing import Optional
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
import numpy as np

# List of datasets available in this module
dsets = [
    "boolq",
    "obqa",
    "arc",
    "winogrande",
    "cqa",
    "cola",
    "mnli",
    "mrpc",
    "qnli",
    "qqp ",
    "rte ",
    "sst2",
    "wnli",
]


class ClassificationDataset:
    """
    An abstract base dataset for sequence classification problems. Multiple
    choice QA problems could also be made a subclass of this class with an
    appropriate collation / formatting.
    """

    def __init__(
        self,
        dset,
        tokenizer,
        n_labels: int,
        preamble: str = "",
        add_space: bool = False,
        numerical: bool = True,
        boolean: bool = False,
        few_shot: bool = False,
        max_len: int = 1024,
    ):
        """
        Args:
            dset: The loaded Dataset
            tokenizer: The model tokenizer
            n_labels: The number of labels / classes for each question
            preamble: Preamble for general pre-trained / 'CausalLM' models
            add_space: Add an explicit space suffix between preamble and answer tokens.
            numerical: whether labels are numerical (0, 1, etc.) or alphabetical (A, B, etc.)
            boolean: whether the labels are boolean (0, 1)
            few_shot: whether to use few-shot prompting (if available)
            max_len: the matximum length of the prompt.
        """
        self.dset = dset
        self.n_labels = n_labels
        self.preamble = preamble
        self.add_space = add_space
        self.tokenizer = tokenizer
        self.numerical = numerical
        self.few_shot = few_shot
        self.max_len = max_len

        spc = " " if self.add_space else ""

        # 1. Build up the token IDS of the class labels.
        if numerical and boolean:
            raise ValueError("Question type cannot be both numerical and boolean")
        if boolean:
            labels = [f"{spc}True", f"{spc}False"]
        elif numerical:
            labels = [f"{spc}{i}" for i in range(self.n_labels)]
        else:  # alphabetical
            labels = [f"{spc}{chr(ord('A')+i)}" for i in range(self.n_labels)]
        self.target_ids = tokenizer(
            labels, return_tensors="pt", add_special_tokens=False
        ).input_ids[:, -1:]
        assert (
            self.target_ids.unique().numel() == self.target_ids.numel()
        ), "Target label IDS are not unique! Try changing add_space or numerical."

        # 2. Get a mapping from the label indices (e.g. 0, 1, 2, etc.) to the
        # target token ids from above (e.g. 345, 346, etc.).
        # That is; {(0, 345), (1, 346), etc}
        self.label_idx2target_id = OrderedDict(
            [(i, self.target_ids[i]) for i in range(n_labels)]
        )
        self.target_id2label_idx = OrderedDict(
            [(self.target_ids[i], i) for i in range(n_labels)]
        )

    @abstractmethod
    def sc_collate_fn(self, batch):
        """Collate function for sequence classification models"""
        raise NotImplementedError

    def sc_loader(self, dset: Dataset, *args, **kwargs) -> DataLoader:
        """Returns the dataloader for sequence classification models"""
        return t.utils.data.DataLoader(
            dset, collate_fn=self.sc_collate_fn, *args, **kwargs
        )

    @abstractmethod
    def clm_collate_fn(self, batch):
        """Collate function for causal language models"""
        raise NotImplementedError

    def clm_loader(self, dset: Dataset, *args, **kwargs) -> DataLoader:
        """Returns the dataloader for causal language models"""
        return t.utils.data.DataLoader(
            dset, collate_fn=self.clm_collate_fn, *args, **kwargs
        )

    def loader(
        self,
        *args,
        is_sc: bool = False,
        split: str = "train",
        subset_size: int = -1,
        subset_seed: Optional[int] = 42,
        subset_balanced: bool = False,
        grad_acc_steps: int = 1,
        drop_last: bool = True,
        **kwargs,
    ):
        if subset_size > 0:
            """
            subset_size = (
                len(self.dset[split])
                if len(self.dset[split]) < subset_size
                else subset_size
            )
            dset = self.dset[split].shuffle(seed=subset_seed).select(range(subset_size))
            """
            if subset_balanced:
                # Group indices by class label
                answer_to_indices = {}
                for idx, example in enumerate(self.dset[split]):
                    label = example["answer"]
                    answer_to_indices.setdefault(label, []).append(idx)

                unique_labels = list(answer_to_indices.keys())
                num_classes = len(unique_labels)
                # Calculate the number of samples per class
                samples_per_class = subset_size // num_classes
                print(f'[dset subsample info] Samples per class: {samples_per_class}')
                balanced_indices = []
                rng = np.random.RandomState(subset_seed)
                for label in unique_labels:
                    indices = answer_to_indices[label]
                    rng.shuffle(indices)
                    # Select up to the desired number, but if a class has fewer examples, take all of them
                    selected = indices[:min(samples_per_class, len(indices))]
                    balanced_indices.extend(selected)

                # If subset_size isn’t exactly divisible by num_classes, fill the remainder randomly
                remainder = subset_size - len(balanced_indices)
                print(f'[dset subsample info] remainder: {remainder}')
                if remainder > 0:
                    remaining_indices = []
                    for label in unique_labels:
                        # Add indices that weren’t already selected
                        remaining = [idx for idx in answer_to_indices[label] if idx not in balanced_indices]
                        remaining_indices.extend(remaining)
                    rng.shuffle(remaining_indices)
                    balanced_indices.extend(remaining_indices[:remainder])

                # Select the balanced indices from the dataset
                dset = self.dset[split].select(balanced_indices)
            else:
                dset = self.dset[split].shuffle(seed=subset_seed).select(range(subset_size))
        else:
            dset = self.dset[split]

        kwargs = {"batch_size": 32, "drop_last": drop_last} | kwargs
        assert (
            kwargs["batch_size"] % grad_acc_steps == 0
        ), "batch size must be divisible by gradient accumulation steps"
        kwargs["batch_size"] = kwargs["batch_size"] // grad_acc_steps

        if is_sc:
            return self.sc_loader(dset, *args, **kwargs)
        else:
            return self.clm_loader(dset, *args, **kwargs)


class BoolQDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 256,
    ):
        dset = load_dataset("boolq")

        prompt = """Read the passage below and answer the question with the words 'True' or 'False'.

Passage: {passage}
Question: {question}
Answer (True or False):"""

        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=False,
            boolean=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    def clm_collate_fn(self, batch):
        prompts = [
            self.preamble.format(
                passage=e["passage"][-self.max_len :], question=e["question"]
            )
            for e in batch
        ]
        classes = t.tensor([int(e["answer"]) for e in batch])
        targets = t.cat([self.label_idx2target_id[c.item()] for c in classes])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [
            self.preamble.format(
                passage=e["passage"][-self.max_len :], question=e["question"]
            )
            for e in batch
        ]
        classes = t.tensor([int(e["answer"]) for e in batch])
        return prompts, classes, None


boolq = BoolQDataset


class OBQADataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 1024,
    ):
        dset = load_dataset("openbookqa", "main")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            4,
            prompt,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """Return the label of the correct answer for each question below.

The sun is responsible for
Choices:
A) puppies learning new tricks
B) children growing up and getting old
C) flowers wilting in a vase
D) plants sprouting, blooming and wilting
Answer: D

What doesn't eliminate waste?
A) plants
B) robots
C) mushrooms
D) bacteria
Answer: B

{question}
Choices:
{choices}
Answer:"""

    zero_shot_preamble = """Return the label of the correct answer for the question below.

Question: {question}
Chioces:
{choices}
Answer:"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            choices = "\n".join(
                [
                    f"{l}) {c}"
                    for c, l, in zip(e["choices"]["text"], e["choices"]["label"])
                ]
            )
            prompts.append(
                self.preamble.format(question=e["question_stem"], choices=choices)
            )
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        targets = t.cat([self.label_idx2target_id[c.item()] for c in classes])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        return prompts, classes, None


obqa = OBQADataset


class ArcSplit(Enum):
    C = "ARC-Challenge"
    E = "ARC-Easy"


class ARCDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        name: ArcSplit = ArcSplit.E,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("ai2_arc", name.value)
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            5,
            prompt,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """Return the label of the correct answer for each question below.

Which two body systems are directly involved in movement?
Choices:
A) muscular and skeletal
B) digestive and muscular
C) skeletal and respiratory
D) respiratory and digestive
Answer: A

{question}
Choices:
{choices}
Answer:"""

    zero_shot_preamble = """Return the label of the correct answer for the question below.

Question: {question}
Choices:
{choices}
Answer:"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            choices = "\n".join(
                [
                    f"{l}) {c}"
                    for c, l in zip(e["choices"]["text"], e["choices"]["label"])
                ]
            )
            prompts.append(
                self.preamble.format(question=e["question"], choices=choices)
            )
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes_alpha = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        classes_num = []
        for e in batch:
            try:
                classes_num.append(int(e["answerKey"]) - 1)
            except:
                classes_num.append(-1)
        # classes_num = t.tensor([int(e["answerKey"]) - 1 for e in batch])
        classes = t.where(classes_alpha < 0, t.tensor(classes_num), classes_alpha)
        targets = t.cat([self.label_idx2target_id[c.item()] for c in classes])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        return prompts, classes, None


arc = ARCDataset


class WinograndeSplit(Enum):
    XS = "winogrande_xs"
    S = "winogrande_s"
    M = "winogrande_m"
    L = "winogrande_l"
    XL = "winogrande_xl"

    # Check sizes at https://huggingface.co/datasets/allenai/winogrande

class WinograndeDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        name: WinograndeSplit = WinograndeSplit.XL,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("winogrande", name.value)
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """Return the label of the correct answer for each question below.

Adam put handwash only clothes in the washer but Aaron washed them by hand as _ was lazy.
Choices:
A) Adam
B) Aaron
Answer: A

Steven proudly showed Michael the mangoes he grew himself all this summer. _ is astonished.
Choices:
A) Stephen
B) Michael
Answer: B

{question}
Choices:
{choices}
Answer:"""

    zero_shot_preamble = """Return the label of the correct answer for the question below.

Question: {question}
Choices:
{choices}
Answer:"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            choices = f"A) {e['option1']}\nB) {e['option2']}"
            prompts.append(
                self.preamble.format(question=e["sentence"], choices=choices)
            )
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([int(e["answer"]) - 1 for e in batch])
        targets = t.cat([self.label_idx2target_id[c.item()] for c in classes])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = [e["sentence"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([int(e["answer"]) - 1 for e in batch])
        return prompts, classes, None


winogrande = WinograndeDataset


class CommonsenseQADataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = True,
        max_len=4096,
    ):
        dset = load_dataset("commonsense_qa")
        super().__init__(
            dset,
            tokenizer,
            5,
            self.few_shot_preamble if few_shot else self.zero_shot_preamble,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )

    # few-shot preamble
    few_shot_preamble = """Answer the questions below correctly.

Question: What do people aim to do at work?
Choices:
A) complete job
B) learn from each other
C) kill animals
D) wear hats
E) talk to each other
Answer: A

Question: Where do adults use glue sticks?
Choices:
A) classroom
B) desk drawer
C) at school
D) office
E) kitchen draw
Answer: D

Question: {question}
Choices:
{choices}
Answer:"""

    zero_shot_preamble = """Answer the multiple choice question below by returning the answer label (A to E)

Question: {question}
Choices:
{choices}
Answer:"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            choices = "\n".join(
                [
                    f"{l}) {c}"
                    for l, c in zip(e["choices"]["label"], e["choices"]["text"])
                ]
            )
            prompts.append(
                self.preamble.format(question=e["question"], choices=choices)
            )
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        targets = t.cat([self.label_idx2target_id[c.item()] for c in classes])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        return prompts, classes, None


cqa = CommonsenseQADataset


class CoLADataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "cola")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each sentence below, indicate whether it is grammatically acceptable (1) or unacceptable (0).

Sentence: If you had eaten more, you would want less.
Answer: 1

Sentence: As you eat the most, you want the least.
Answer: 0

Sentence: {sentence}
Answer:"""

    zero_shot_preamble = """For each sentence below, indicate whether it is grammatically acceptable (1) or unacceptable (0).

Sentence: {sentence}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [self.preamble.format(sentence=e["sentence"]) for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["sentence"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


cola = CoLADataset


class MNLIDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "mnli")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            3,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each premise below, indicate whether the hypothesis entails (0), is neutral towards (1) or contradicts (2) the premise.

Hypothesis: Buffet and a la carte available.
Premise: It has a buffet.
Answer: 0

Hypothesis: He had never felt better.
Premise: The medicine he had taken had worked well.
Answer: 1

Hypothesis: Oh, what a fool I feel!
Premise: I am beyond proud
Answer: 2

Hypothesis: {hypothesis}
Premise: {premise}
Answer:"""

    zero_shot_preamble = """For each premise below, indicate whether the hypothesis entails (0), is neutral towards (1) or contradicts (2) the premise.

Hypothesis: Buffet and a la carte available.
Premise: It has a buffet.
Answer: 0

Hypothesis: He had never felt better.
Premise: The medicine he had taken had worked well.
Answer: 1

Hypothesis: Oh, what a fool I feel!
Premise: I am beyond proud
Answer: 2

Hypothesis: {hypothesis}
Premise: {premise}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(hypothesis=e["hypothesis"], premise=e["premise"])
            for e in batch
        ]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["hypothesis"] + " " + e["premise"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


mnli = MNLIDataset


class MRPCDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "mrpc")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each pair of sentences below, indicate whether the Sentence 1 is equivalent (1) or not equivalent (2) to the Sentence 2.

Sentence 1: Yucaipa owned Dominick's before selling the chain to Safeway in 1998 for $2.5 billion.
Sentence 2: Yucaipa bought Dominick's in 1995 for $693 million and sold it to Safeway for $1.8 billion in 1998.
Answer: 0

Sentence 1: Amrozi accused his brother, whom he called "the witness", of deliberately distorting his evidence.
Sentence 2: Referring to him as only "the witness", Amrozi accused his brother of deliberately distorting his evidence.
Answer: 1

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    zero_shot_preamble = """For each pair of sentences below, indicate whether the Sentence 1 is equivalent (1) or not equivalent (2) to the Sentence 2.

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(sentence_1=e["sentence1"], sentence_2=e["sentence2"])
            for e in batch
        ]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["sentence1"] + " " + e["sentence2"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


mrpc = MRPCDataset


class QNLIDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "qnli")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each sentence below, indicate whether it entails (0) or does not entail (1) the associated question.

Question: Which collection of minor poems are sometimes attributed to Virgil?
Sentence: A number of minor poems, collected in the Appendix Vergiliana, are sometimes attributed to him.
Answer: 0

Question: What was the highest order of species n land?
Sentence: The climate was much more humid than the Triassic, and as a result, the world was very tropical.
Answer: 1

Question: {question}
Sentence: {sentence}
Answer:"""

    zero_shot_preamble = """For each sentence below, indicate whether it entails (0) or does not entail (1) the associated question.

Question: {question}
Sentence: {sentence}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(question=e["question"], sentence=e["sentence"])
            for e in batch
        ]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["question"] + " " + e["sentence"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


qnli = QNLIDataset


class QQPDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "qqp")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each pair of questions below, indicate whether the first is a duplicate (1) or not a duplicate (0) of the first.

Question 1: How is air traffic controlled?
Question 2: How do you become an air traffic controller?
Answer: 0

Question 1: What are the coolest Android hacks and tricks you know?
Question 2: What are some cool hacks for Android phones?
Answer: 1

Question 1: {question_1}
Question 2: {question_2}
Answer:"""

    zero_shot_preamble = """For each pair of questions below, indicate whether the first is a duplicate (1) or not a duplicate (0) of the first.

Question 1: {question_1}
Question 2: {question_2}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(question_1=e["question1"], question_2=e["question2"])
            for e in batch
        ]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["question1"] + " " + e["question2"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


qqp = QQPDataset


class RTEDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "rte")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each pair of sentences below, indicate whether the second entails (0) or does not entail (1) the first.

Sentence 1: Edward VIII became King in January of 1936 and abdicated in December.
Sentence 2: King Edward VIII abdicated in December 1936.
Answer: 0

Sentence 1: No Weapons of Mass Destruction Found in Iraq Yet.
Sentence 2: Weapons of Mass Destruction Found in Iraq.
Answer: 1

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    zero_shot_preamble = """For each pair of sentences below, indicate whether the second entails (0) or does not entail (1) the first.

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(sentence_1=e["sentence1"], sentence_2=e["sentence2"])
            for e in batch
        ]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["sentence1"] + " " + e["sentence2"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


rte = RTEDataset


class SST2Dataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "sst2")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each sentence below, indicate whether the sentiment is negative (0) or positive (1).

Sentence: a depressed fifteen-year-old 's suicidal poetry
Answer: 0

Sentence: the greatest musicians
Answer: 1

Sentence: {sentence}
Answer:"""

    zero_shot_preamble = """For each sentence below, indicate whether the sentiment is negative (0) or positive (1).

Sentence: {sentence}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [self.preamble.format(sentence=e["sentence"]) for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["sentence"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


sst2 = SST2Dataset


class WNLIDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "wnli")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each pair of sentences below, indicate whether the second entails (1) or does not entail (0) the first.

Sentence 1: Steve follows Fred's example in everything. He influences him hugely.
Sentence 2: Steve influences him hugely.
Answer: 0

Sentence 1: The police arrested all of the gang members. They were trying to stop the drug trade in the neighborhood.
Sentence 2: The police were trying to stop the drug trade in the neighborhood.
Answer: 1

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    zero_shot_preamble = """For each pair of sentences below, indicate whether the second entails (1) or does not entail (0) the first.

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(sentence_1=e["sentence1"], sentence_2=e["sentence2"])
            for e in batch
        ]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["sentence1"] + " " + e["sentence2"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


wnli = WNLIDataset
