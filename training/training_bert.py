import json
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

print("Loading training data...")

# loading the datset
with open("data/training_data.json") as f:
    data = json.load(f)

print("Total samples:", len(data))

# label mapping
label_map = {
    "SUPPORT": 0,
    "REFUTE": 1,
    "NEUTRAL": 2
}

texts = []
labels = []

for row in data:
    claim = row["claim"]
    texts.append(claim)
    labels.append(label_map[row["label"]])

# train and validation split of the data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.1,
    random_state=42
)

print("Training samples:", len(train_texts))
print("Validation samples:", len(val_texts))

# tokenizing the data
print("Loading tokenizer...")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=False,   
        max_length=128   
    )

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# converting to dataset objects
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

val_dataset = Dataset.from_dict({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": val_labels
})

# laoding the model
print("Loading BERT model...")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

# training arguments
training_args = TrainingArguments(
    output_dir="models/fact_checker_model",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    evaluation_strategy="epoch",
    save_strategy="epoch",

    logging_steps=200,

    learning_rate=2e-5,
    weight_decay=0.01,

    load_best_model_at_end=True,

    fp16=False,
    dataloader_num_workers=0
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# loading the trainder with the model, training arguments, datasets, tokenizer and data collator and starting the training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("Starting training...")

trainer.train()

print("Training completed.")

# Save model
trainer.save_model("models/fact_checker_model")
tokenizer.save_pretrained("models/fact_checker_model")

print("Model saved successfully!")