import argparse
import json

from src.bio.bio_generator import BIOGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic NER data using OpenAI API"
    )

    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output", type=str, default="synthetic_ner_data.json", help="Output file path"
    )

    args = parser.parse_args()

    generator = BIOGenerator(openai_api_key=args.api_key)

    print(f"Generating {args.num_samples} synthetic NER data samples...")
    dataset = generator.generate_dataset(args.num_samples)

    print(f"Saving dataset to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    print("Data generation complete.")


if __name__ == "__main__":
    main()
