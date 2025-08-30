import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from faker import Faker
from openai import OpenAI, OpenAIError

SYSTEM_PROMPT = """You are simulating user-generated text for NER training, focusing on natural dialogues or messages 
where some users might inadvertently expose PII or sensitive API keys. Task: Generate natural-looking conversational or 
textual snippets containing placeholders for PII and realistic fake API keys.
Rules:
- PII placeholders must be wrapped in curly braces exactly as: {FULL_NAME}, {EMAIL}, {PHONE}, {ADDR}, {SSN}, {CC}, {DATE}.
- Generate realistic fake API keys directly in text (not placeholders), using only letters (a-z, A-Z) and digits (0-9),
 respecting these formats:
  * AWS keys: AKIA + 16 uppercase alphanumeric chars (A-Z, 0-9), e.g., AKIAIOSFODNN7EXAMPLE
  * OpenAI keys: sk- + 48 alphanumeric chars (a-z, A-Z, 0-9), e.g., sk-1234abcd5678EFGH9012ijkl3456MNOP7890qrst1234UVWX
  * GitHub tokens: ghp_ + 36 alphanumeric chars, e.g., ghp_1234abcd5678EFGH9012ijkl3456MNOP7890
  * Stripe keys: sk_live_ or pk_test_ + 24 alphanumeric chars, e.g., sk_live_1234abcd5678EFGH9012ij
  * Generic API keys: random 32–64 alphanumeric chars
- Each text must include at least one instance (or all) of each PII placeholder somewhere within the text.
- Include 1–2 API keys in natural locations such as code snippets, config files, logs, error messages, or conversation context.
- Write 4–5 sentences that follow the example I give you below.
- The tone should simulate real user behavior—unintentional disclosure, mundane conversation, or troubleshooting context.
- Do not include explanations, tags, or metadata. Output only the generated text snippet.
"""

EXAMPLES = [
    "During today’s project sync, {FULL_NAME} shared his new social security number {SSN} to expedite background checks. Also, Jane mentioned her passport number X12345678 for travel approval. Please verify these against the database.",
    "Hello support, my name is {FULL_NAME}, and my account number is 987654321. I noticed unauthorized transactions on my bank card ending {CC}. My full billing address is {ADDR}. Kindly freeze my account ASAP.",
    "Dear HR, I’m forwarding my onboarding docs. My employee ID is EID202589, and attached is a scanned copy of my driver’s license number D1234567890. Also, my phone is {PHONE} for any urgent contact.",
    "Recent leaks show a database of users with detailed info: names, DOBs, and credit card numbers like {CC} exposed, alongside emails such as {EMAIL}.",
    "Alice: I can’t find my tax documents! Bob: Did you check your last tax return? I remember your SSN was {SSN} there. Alice: Yes, and it’s also in my email: {EMAIL}. I need to update my direct deposit info with account number 123456789.",
    "{\n  'api_key': 'sk_live_51Hfdjd9f3o87sxEXAMPLE',\n  'user_email': '{EMAIL}',\n  'user_phone': '{PHONE}',\n  'address': '{ADDR}'\n}",
    "I recently returned the product. My order number is ORD123456, and you can refund to the card ending {CC}. Contact me at {EMAIL} or {PHONE} if you need more details.",
    "Excited to share my vacation plans! Flights booked under passport number P98765432 and hotel reservations tied to my corporate email {EMAIL}.",
    "My full legal name is {FULL_NAME}, reachable at {EMAIL}. My LinkedIn profile contains my phone number {PHONE} and my current address is {ADDR}.",
    "Got my new health insurance card today: ID HI-123456789. Also got my new bank card ending in {CC}. Feeling nervous about sharing my full SSN {SSN} in the forms.",
]


def call_openai_to_generate_text(
    client: OpenAI,
    placeholder_prompt: str,
    retries: int = 3,
    delay: int = 2,
) -> str:
    """
    Uses OpenAI GPT to generate text containing PII placeholders.

    Args:
        client: OpenAI GPT client.
        placeholder_prompt: User prompt guiding text generation.
        retries: Number of times to retry generating text.
        delay: Delay between retries.

    Returns:
        Generated text string with placeholders.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Adjust your text based on this example, make sure it is not the same, focus on making diversity"
                        + placeholder_prompt,
                    },
                ],
                max_tokens=200,
                temperature=0.7,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            print(f"OpenAI API error on attempt {attempt+1}: {e}")
            time.sleep(delay)
    raise RuntimeError("Failed to generate text after retries")


def inject_pii(text_with_placeholders: str, pii: Dict[str, str]) -> str:
    """
    Replaces PII placeholders in text with actual fake data.

    Args:
        text_with_placeholders: Text containing placeholders such as {FULL_NAME}.
        pii: Dict mapping placeholders to fake values.

    Returns:
        Text with placeholders replaced by real values.

    Raises:
        ValueError if any placeholder is missing in the text.
    """
    text = text_with_placeholders
    for key, value in pii.items():
        placeholder = "{" + key + "}"
        if placeholder in text:
            text = text.replace(placeholder, value)
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes text into words, punctuation, and emails for annotation alignment.

    Args:
        text: Raw text string.

    Returns:
        List of token strings.
    """
    return re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\w+|[.,@-]", text
    )


def label_tokens(tokens: List[str], annotations: Dict[str, str]) -> List[str]:
    """
    Assigns BIO labels to tokens based on annotated PII entities.

    Args:
        tokens: Tokenized text.
        annotations: Dict of PII types mapped to their string values.

    Returns:
        List of BIO label strings aligned with tokens.
    """
    labels = ["O"] * len(tokens)
    label_map = {
        "FULL_NAME": "PER",
        "EMAIL": "EMAIL",
        "PHONE": "PHONE",
        "ADDR": "ADDR",
        "SSN": "SSN",
        "CC": "CC",
        "DATE": "DATE",
    }
    for label_name, entity in annotations.items():
        entity_tokens = tokenize_text(entity)
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i : i + len(entity_tokens)] == entity_tokens:
                labels[i] = "B-" + label_map[label_name]
                for j in range(i + 1, i + len(entity_tokens)):
                    labels[j] = "I-" + label_map[label_name]
    return labels


class BIOGenerator:
    """
    Generates synthetic NER training samples with realistic PII content.

    Methods:
        generate_pii(): Generates fake PII values using Faker.
        generate_sample(prompt): Creates one annotated sample from an OpenAI prompt.
        generate_dataset(prompt, n): Generates multiple annotated samples.
    """

    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.faker = Faker()

    def generate_pii(self) -> Dict[str, str]:
        """
        Generates realistic fake PII data for all required placeholder types.

        Returns:
            Dictionary mapping placeholder names to generated fake values.
        """
        return {
            "FULL_NAME": self.faker.name(),
            "EMAIL": self.faker.email(),
            "PHONE": self.faker.phone_number(),
            "ADDR": self.faker.address().replace("\n", ", "),
            "SSN": self.faker.ssn(),
            "CC": self.faker.credit_card_number(),
            "DATE": self.faker.date(),
        }

    def generate_sample(self, openai_prompt: str) -> Dict:
        """
        Generates a single NER training sample with tokens and BIO labels.

        Args:
            openai_prompt: Prompt string to instruct OpenAI text generation.

        Returns:
            Dictionary with tokenized raw text ('raw'), tokenized texts ('text'), and corresponding BIO labels ('labels').
        """
        pii = self.generate_pii()
        text_with_placeholders = call_openai_to_generate_text(
            self.client, openai_prompt
        )
        text_with_pii = inject_pii(text_with_placeholders, pii)
        tokens = tokenize_text(text_with_pii)
        labels = label_tokens(tokens, pii)
        return {"raw": text_with_pii, "text": tokens, "labels": labels}

    def generate_dataset(self, num_samples: int) -> List[Dict]:
        """
        Generates multiple annotated NER training samples.

        Args:
            openai_prompt: Prompt for generating each sample's placeholder text.
            num_samples: Number of samples to generate.

        Returns:
            List of sample dictionaries each with 'text' and 'labels'.
        """

        samples = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self.generate_sample, random.choice(EXAMPLES))
                for _ in range(num_samples)
            ]
            for future in as_completed(futures):
                try:
                    samples.append(future.result())
                except Exception as e:
                    print(f"Skipping failed sample: {e}")
        return samples
