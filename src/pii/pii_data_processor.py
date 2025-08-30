import json
from typing import Dict, List

from datasets import Dataset
from transformers import AutoTokenizer

from src.pii.pii_label import PIILabel


def load_bio_data(data_path: str) -> List[Dict]:
    """
    Loads BIO-tagged data from a JSON file.

    Args:
        data_path: Path to the JSON file.

    Returns:
        List of data samples each with 'text' and 'labels'.
    """
    with open(data_path, "r") as f:
        return json.load(f)


class PIIDataProcessor:
    """
    Processes raw text and BIO labels for PII token classification.

    Attributes:
        tokenizer: Hugging Face tokenizer instance.
        label_map: Mapping from label string to label index.
        max_length: Max token sequence length.
    """

    def __init__(
        self, tokenizer_name: str = "boltuix/NeuroBERT-Mini", max_length: int = 128
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label_map = PIILabel.get_label_map()
        self.max_length = max_length

    def tokenize_and_align_labels(self, examples: Dict) -> Dict:
        """Tokenizes text and aligns BIO labels to subword tokens.

        Args:
            examples: Batch dict with 'text' and 'labels'.

        Returns:
            Tokenized inputs dict with aligned label ids.
        """
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=self.max_length,
        )

        labels = []
        for i, label_list in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_map[label_list[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def prepare_dataset(self, data_path: str) -> Dataset:
        """
        Loads data and prepares tokenized Dataset for training.

        Args:
            data_path: Path to BIO JSON data.

        Returns:
            Hugging Face Dataset with aligned labels.
        """
        raw_data = load_bio_data(data_path)
        texts = [item["text"] for item in raw_data]
        labels = [item["labels"] for item in raw_data]
        dataset = Dataset.from_dict({"text": texts, "labels": labels})

        return dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset.column_names,
        )
