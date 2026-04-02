import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as transforms

class IDRiDMultimodalDataset(Dataset):
    def __init__(self, csv_file, image_dir, tokenizer_name='emilyalsentzer/Bio_ClinicalBERT', transform=None, max_length=128):
        """
        Args:
            csv_file (string): Path to the csv file with annotations and captions.
            image_dir (string): Directory with all the images.
            tokenizer_name (string): HuggingFace model string for the text tokenizer.
            transform (callable, optional): Optional transform to be applied on an image.
            max_length (int): Maximum token length for the captions.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        
        # We use ClinicalBERT tokenizer as it understands medical terminology better than standard BERT
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Process the Image
        img_name = str(self.data_frame.iloc[idx]['Image name'])
        # Handle potential missing extensions in CSV
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'
            
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 2. Process the Text Caption
        caption = str(self.data_frame.iloc[idx]['Captions'])
        
        # Tokenize the text for the Transformer
        encoding = self.tokenizer(
            caption,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Remove the batch dimension added by the tokenizer
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 3. Get the Label (Retinopathy grade 0-4)
        label = int(self.data_frame.iloc[idx]['Retinopathy grade'])

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }
    