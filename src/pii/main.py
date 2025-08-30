import argparse

from src.pii.pii_data_processor import PIIDataProcessor
from src.pii.pii_model_trainer import PIIModelTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train a PII token classification model."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the BIO-tagged JSON data file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pii_neurobert_mini_model",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="boltuix/NeuroBERT-Mini",
        help="Pretrained BERT model name.",
    )
    args = parser.parse_args()

    print("Preparing dataset...")
    processor = PIIDataProcessor(tokenizer_name=args.model_name)
    dataset = processor.prepare_dataset(args.data_path)

    print("Initializing trainer...")
    trainer = PIIModelTrainer(model_name=args.model_name)
    trainer.train(dataset, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
