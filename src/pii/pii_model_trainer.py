from typing import Tuple

import evaluate
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    Trainer,
    DataCollatorForTokenClassification,
    TrainingArguments,
)

from src.pii.pii_label import PIILabel


def _split_dataset(
    dataset: Dataset, test_size: float = 0.2, val_size: float = 0.1
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Splits dataset into train/validation/test sets.

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    dataset_dict = dataset.train_test_split(test_size=test_size + val_size, seed=42)
    remaining = dataset_dict["train"]
    temp_split_size = val_size / (test_size + val_size)
    val_test_split = remaining.train_test_split(test_size=temp_split_size, seed=42)
    train_dataset = val_test_split["train"]
    val_dataset = val_test_split["test"]
    test_dataset = dataset_dict["test"]
    return train_dataset, val_dataset, test_dataset


class PIIModelTrainer:
    """
    Fine-tunes a BERT model for PII token classification.

    Automatically splits dataset into train/validation/test sets.
    """

    def __init__(self, model_name: str = "boltuix/NeuroBERT-Mini"):
        self.model_name = model_name
        self.num_labels = len(PIILabel.LABEL_LIST)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(
            model_name, num_labels=self.num_labels
        )

    def train(self, full_dataset: Dataset, output_dir: str = "./pii_neurobert_model"):
        """
        Fine-tunes the model on PII-labeled data with train/val/test split.

        Args:
            full_dataset: Tokenized dataset containing the 'labels' field.
            output_dir: Directory to save model and tokenizer.

        Returns:
            Trainer instance.
        """
        train_dataset, val_dataset, test_dataset = _split_dataset(full_dataset)

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, padding=True
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            eval_strategy="steps",
            learning_rate=3e-5,
            fp16=False,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

        def compute_metrics(p):
            labels = p.label_ids.flatten()
            preds = p.predictions.argmax(-1).flatten()
            metric = evaluate.load("f1")
            mask = labels != -100
            return metric.compute(
                predictions=preds[mask], references=labels[mask], average="weighted"
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print("Starting PII detection fine-tuning...")
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

        print("Evaluating on test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        print("Test metrics:", test_metrics)

        return trainer
