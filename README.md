# NLP Fact Checker

This project is an NLP-based fact-checking system using BERT.

## Features
- Claim classification (Support / Refute / Neutral)
- Evidence retrieval using SciFact dataset
- Validity scoring system

## Tech Stack
- Python
- Transformers (BERT)
- PyTorch
- SciFact dataset

## Setup Instructions

1. Clone the repository:
   git clone <repo-url>

2. Create virtual environment:
   python -m venv venv

3. Activate environment:
   source venv/bin/activate

4. Install dependencies:
   pip install -r requirements.txt

5. Download datasets:
   python data/download_datasets.py

6. Train model:
   python training/train_bert.py

7. Run prediction:
   python predict.py

## Note
Model files are not included due to size limits.