"""
step2_dataset.py
=================
Multimodal dataset loader for IDRiD.
Additions over original:
  - Mixup augmentation support (alpha blending between two samples)
  - Returns sample index for TTA (test-time augmentation) tracking
  - Handles missing captions gracefully after leakage stripping
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as transforms


class IDRiDMultimodalDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        tokenizer_name='emilyalsentzer/Bio_ClinicalBERT',
        transform=None,
        max_length=128,
        use_mixup=False,
        mixup_alpha=0.4,
    ):
        self.data_frame   = pd.read_csv(csv_file)
        self.image_dir    = image_dir
        self.transform    = transform
        self.max_length   = max_length
        self.use_mixup    = use_mixup
        self.mixup_alpha  = mixup_alpha
        self.num_classes  = 5

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.data_frame)

    def _load_single(self, idx):
        """Load one (image, input_ids, attention_mask, label) sample."""
        img_name = str(self.data_frame.iloc[idx]['Image name'])
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        caption = str(self.data_frame.iloc[idx]['Captions'])
        # Safety fallback for empty captions after grade stripping
        if caption.strip() == "" or caption == "nan":
            caption = "fundus image with retinal findings"

        encoding = self.tokenizer(
            caption,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        label = int(self.data_frame.iloc[idx]['Retinopathy grade'])
        return (
            image,
            encoding['input_ids'].squeeze(0),
            encoding['attention_mask'].squeeze(0),
            label,
        )

    def __getitem__(self, idx):
        image, input_ids, attention_mask, label = self._load_single(idx)

        # ── Mixup augmentation ──────────────────────────────────────
        # Blends two samples' images and creates soft labels.
        # This forces the model to learn smooth decision boundaries
        # and significantly reduces overconfidence / overfitting.
        if self.use_mixup and np.random.random() < 0.5:
            idx2 = np.random.randint(0, len(self))
            image2, _, _, label2 = self._load_single(idx2)

            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            image = lam * image + (1 - lam) * image2

            # Soft one-hot labels for Mixup
            soft_label = torch.zeros(self.num_classes)
            soft_label[label]  += lam
            soft_label[label2] += (1 - lam)
        else:
            soft_label = torch.zeros(self.num_classes)
            soft_label[label] = 1.0

        return {
            'image':          image,
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'label':          torch.tensor(label, dtype=torch.long),
            'soft_label':     soft_label,   # used by Mixup-aware loss
        }
