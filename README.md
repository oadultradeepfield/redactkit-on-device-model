# RedactKit On-device Model

This repository contains Python scripts used to train and develop the **PII detection model** for RedactKit, a mobile
app that redacts PII and potential API key leaks in text to ensure security before users submit text to an LLM chatbot.

This project was developed as part of the **TikTok TechJam 2025 Hackathon**, Track #7: _Privacy Meets AI: Building a
Safer Digital Future_.

## Getting Started

### Prerequisites

1. Python >= 3.11 installed.
2. Install required packages:

```bash
pip install -r requirements.txt
```

### Generating the Data

This project uses the [boltuix/NeuroBERT-Mini](https://huggingface.co/boltuix/NeuroBERT-Mini) model, fine-tuned
for **Name Entity Recognition (NER)** on PII.

To generate training data:

1. Create template messages for an LLM with placeholders for PII (e.g., `{NAME}`, `{EMAIL}`).
2. Replace placeholders with realistic variations using Python's [Faker](https://faker.readthedocs.io/en/master/)
   library.

You need an **OpenAI API key** to generate data. Run:

```bash
python -m src.bio.main --api_key <YOUR_API_KEY> --output ./data.json --num_samples 3000
```

* `--num_samples` can be adjusted based on API usage and desired dataset size.
* In this project, we generate **3,000 samples**, each containing 2–3 sentences to balance quality and speed.

**Example generated entry:**

> "Hi {NAME}, welcome to the team! Please review the onboarding instructions below. You can reach HR at {EMAIL} or call
> {PHONE} if you have questions. First, complete your profile by visiting {ADDRESS}."

The output `data.json` is an array of objects containing:

* `raw`: original text with placeholders.
* `text`: array of tokenized text using HuggingFace's tokenizer.
* `labels`: NER classification for each token.
  See [`src/pii/pii_label.py`](src/pii/pii_label.py) for details.

### Training the Model

The fine-tuning logic is abstracted. To train the model:

```bash
python -m src.pii.main --data_path ./data.json
```

* The model will be saved in `pii_neurobert_mini_model` with checkpoints and performance logs.
* Our testing dataset achieves **nearly 100% F1-Score**, demonstrating high accuracy.

Verify inference with an example in [`tests`](./tests/main.py):

```bash
python -m tests.main
```

### Converting to CoreML

CoreML allows machine learning models to run on Apple devices (iPhone, iPad, etc.). We chose CoreML due to our expertise
in Swift.

* The model can also be converted to other formats like ONNX for Android, but this project focuses on iOS for fast
  prototyping.

Convert the trained model to CoreML:

```bash
python -m src.core_ml.main --model_path ./pii_neurobert_mini_model
```

* This creates `PIIDetectionModel.mlpackage`, compatible with Xcode and Swift development.

## Libraries Used

As specified in `requirements.txt`:

```txt
Faker~=37.6.0
openai~=1.102.0
evaluate~=0.4.5
datasets~=4.0.0
transformers~=4.56.0
coremltools~=8.3.0
numpy~=2.3.2
torch~=2.8.0
```

More information on using the exported model is available in the [frontend repository](). The open-sourced model file
can be accessed [here](/PIIDetectionModel.mlpackage).

