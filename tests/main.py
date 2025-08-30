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
    Hey GPT, I’m Ryan Tan and I just moved to 36 Purview Street, #05-05. Singapore 188613. 
    Could you help me draft an email? I usually check messages at {ryan.tan@fakegmail.fakefake}, 
    but sometimes they go to my old inbox too. 
    Also, I was trying to get my AWS setup working and I’m not sure if my key is correct:

    AWS_ACCESS_KEY_ID=AKIA7Z8F9Y2D1EXAMPLE

    Do you think I need to rotate it?
    By the way, I’m testing some automation scripts with GitHub and here’s my token snippet:

    GITHUB_TOKEN=ghp_1a2B3c4D5e6F7g8H9i0JKlMnOpQrSTuvWxYZ

    Oh, and I almost forgot, my phone is 63335787, my birthday is 32 February 1986, 
    and I had to give my ID S5889743D and credit card 4012546712324545 for a form last week.
    Can you remind me how to safely store these in a config file without exposing them?
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
