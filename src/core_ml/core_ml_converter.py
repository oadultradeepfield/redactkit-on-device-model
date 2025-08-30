import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import BertForTokenClassification, AutoTokenizer


class ArgmaxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        argmax_tensor = torch.argmax(outputs.logits, dim=-1)
        return argmax_tensor.flatten()


class CoreMLConverter:
    """
    Converts a fine-tuned BERT token classification model to CoreML format.

    Attributes:
        model_path: Path or identifier of the pretrained model.
        max_length: Fixed input sequence length for model tracing.
        tokenizer: BERT tokenizer instance.
        model: BERT token classification model in eval mode.
    """

    def __init__(self, model_path: str, max_length: int = 128):
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.model.eval()

    def create_traced_model(self) -> torch.jit.ScriptModule:
        """
        Creates a TorchScript traced model for CoreML conversion using dummy inputs.

        Returns:
            Traced TorchScript model.
        """
        dummy_input_ids = torch.randint(
            0, self.tokenizer.vocab_size, (1, self.max_length)
        )
        dummy_attention_mask = torch.ones((1, self.max_length), dtype=torch.long)

        wrapped_model = ArgmaxWrapper(self.model)
        traced_model = torch.jit.trace(
            wrapped_model, (dummy_input_ids, dummy_attention_mask)
        )
        return traced_model

    def convert_to_coreml(self, output_path: str = "PIIDetectionModel.mlpackage"):
        """
        Converts the traced model to CoreML format and saves it.

        Args:
            output_path: File path to save the converted CoreML model.

        Returns:
            The converted CoreML model instance.
        """
        print("Creating traced model...")
        traced_model = self.create_traced_model()

        print("Converting to CoreML...")
        coreml_model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input_ids", shape=(1, self.max_length), dtype=np.int32
                ),
                ct.TensorType(
                    name="attention_mask", shape=(1, self.max_length), dtype=np.int32
                ),
            ],
            outputs=[ct.TensorType(name="predictions", dtype=np.int32)],
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.ALL,
            debug=True,
        )

        # Metadata
        coreml_model.short_description = "PII Detection using fine-tuned NeuroBERT-Mini"
        coreml_model.author = "PII Detection System"
        coreml_model.version = "1.0"

        # Input/Output Descriptions
        coreml_model.input_description["input_ids"] = (
            "Tokenized input text (BertTokenizer)"
        )
        coreml_model.input_description["attention_mask"] = (
            "Attention mask for input tokens"
        )
        coreml_model.output_description["predictions"] = (
            "Predicted class IDs for each token"
        )

        coreml_model.save(output_path)
        print(f"CoreML model saved to {output_path}")

        return coreml_model
