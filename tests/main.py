import torch
from transformers import AutoTokenizer, BertForTokenClassification

from src.pii.pii_label import PIILabel


def main():
    """Test the fine-tuned model on sample text."""
    tokenizer = AutoTokenizer.from_pretrained("../pii_neurobert_mini_model")
    model = BertForTokenClassification.from_pretrained("../pii_neurobert_mini_model")
    model.eval()

    # Sample text for inference
    text = """
    Hey team! Just a quick reminder that our trip is scheduled for Nov 29, 1990. 
    Please send me your updated contact info \u2014 Michele Richardson, katrinahowe@example.net,
    +1-957-474-0667x68212, and 140 Craig Neck Suite 273, East Robertfurt, MN 05192. 
    By the way, I ran a quick check using our internal tools and saw your SSN: 571-86-4108.
    Looking forward to finalizing everything
    """

    # Tokenize input
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=128
    )

    # Predict labels
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Convert IDs to labels
    id_to_label = PIILabel.get_id_to_label()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Print results
    print("PII Detection Results:")
    print("-" * 50)
    for token, pred_id in zip(tokens, predictions[0]):
        if token not in ["[CLS]", "[SEP]", "[PAD]"] and pred_id != -100:
            label = id_to_label[pred_id.item()]
            if label != "O":
                print(f"{token:15} -> {label}")


if __name__ == "__main__":
    main()
