import os
import pandas as pd
from PIL import Image
import re
from tqdm import tqdm

def clean_caption(text):
    """Basic text cleaning for clinical captions."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_dataset(raw_csv, raw_img_dir, proc_csv, proc_img_dir, img_size=(256, 256)):
    """Resizes images and cleans text, saving them to a new processed directory."""
    os.makedirs(proc_img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(proc_csv), exist_ok=True)
    
    df = pd.read_csv(raw_csv)
    print(f"Cleaning text in {raw_csv}...")
    df['Captions'] = df['Captions'].apply(clean_caption)
    
    print(f"Resizing images and saving to {proc_img_dir}...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_name = str(row['Image name'])
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'
            
        raw_path = os.path.join(raw_img_dir, img_name)
        proc_path = os.path.join(proc_img_dir, img_name)
        
        if os.path.exists(raw_path):
            try:
                with Image.open(raw_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(img_size, Image.Resampling.BILINEAR)
                    img.save(proc_path, quality=95)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
        else:
            print(f"Warning: {raw_path} not found.")

    # Save the cleaned dataframe
    df.to_csv(proc_csv, index=False)
    print(f"Processed dataset saved to {proc_csv}\n")

if __name__ == "__main__":
    # Define Kaggle input (Read-Only) and working (Writeable) directories
    # If running locally, change these to your local paths (e.g., 'data/raw/idrid/...')
    

    # LOCAL PATHS:
    RAW_DIR = 'data/raw/idrid/'
    PROC_DIR = 'data/processed/idrid/'
    
    # Train Data
    process_dataset(
        raw_csv=os.path.join(RAW_DIR, 'train_labels.csv'),
        raw_img_dir=os.path.join(RAW_DIR, 'train_images/'),
        proc_csv=os.path.join(PROC_DIR, 'train_labels.csv'),
        proc_img_dir=os.path.join(PROC_DIR, 'train_images/')
    )
    
    # Test Data
    process_dataset(
        raw_csv=os.path.join(RAW_DIR, 'test_labels.csv'),
        raw_img_dir=os.path.join(RAW_DIR, 'test_images/'),
        proc_csv=os.path.join(PROC_DIR, 'test_labels.csv'),
        proc_img_dir=os.path.join(PROC_DIR, 'test_images/')
    )
    
    print("All preprocessing complete. Data is ready for training.")