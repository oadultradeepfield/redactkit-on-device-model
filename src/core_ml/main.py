import argparse

from src.core_ml.core_ml_converter import CoreMLConverter


def main():
    parser = argparse.ArgumentParser(
        description="Convert a fine-tuned BERT model to CoreML format."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned BERT model directory or Hugging Face model ID.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="PIIDetectionModel.mlpackage",
        help="File path to save the converted CoreML model.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum input sequence length for the model.",
    )
    args = parser.parse_args()

    print("Initializing CoreML converter...")
    converter = CoreMLConverter(model_path=args.model_path, max_length=args.max_length)
    converter.convert_to_coreml(output_path=args.output_path)


if __name__ == "__main__":
    main()
