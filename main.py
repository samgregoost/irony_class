model_checkpoint = "/path/to/mode/data"
batch_size = 16

import csv

# read tab-delimited text and store data to lists. 0th index is ignored to avoid header
def read_data(path):
  with open(test_path) as f:
    reader = csv.reader(f, delimiter="\t")
    data = list(reader)
  return data[1:]

test_path = "./SemEval2018-Task3/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt"
train_path = "./SemEval2018-Task3/datasets/train/SemEval2018-T3-train-taskA_emoji.txt"

test_data = read_data(test_path)
train_data = read_data(train_path)
train_texts = [el[2] for el in train_data]
val_texts = [el[2] for el in test_data]

train_labels = [int(el[1]) for el in train_data]
val_labels = [int(el[1]) for el in test_data]

#Tokenize the sentences
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('/flush5/ram095/iccv2021/oracle/model/')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create dataset objects with data
import torch

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

#load pretrained model
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

#Configure training params
metric_name =  "accuracy"



args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=10,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# metrics definitions
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

#define trainer
trainer = Trainer(
    model=model,                         
    args=args,                  
    train_dataset=train_dataset,      
    eval_dataset=val_dataset,   
    compute_metrics=compute_metrics
)

#train the model
trainer.train()
trainer.evaluate()
