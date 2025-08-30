import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from faker import Faker
from openai import OpenAI, OpenAIError

SYSTEM_PROMPT = """
You generate natural-looking text snippets for NER training that include personal and sensitive information.

Rules:
1. Include all PII placeholders exactly as: {FULL_NAME}, {EMAIL}, {PHONE}, {ADDR}, {SSN}, {CC}, {DATE}.
2. Include 1–2 realistic API keys naturally in the text. Use these formats:
   - AWS: AKIA + 16 uppercase letters/digits
   - OpenAI: sk- + 48 letters/digits
   - GitHub: ghp_ + 36 letters/digits
   - Stripe: sk_live_ or pk_test_ + 24 letters/digits
   - Generic: 32–64 letters/digits
3. Placeholders and API keys must appear in context (code, chat, logs, config files, emails, etc.).
4. Tone: natural and conversational.
5. Output only the generated snippet. No explanations, tags, or metadata.
"""

EXAMPLES = [
    "A developer shares a snippet in a team chat containing code, accidentally leaving API keys and placeholders like {FULL_NAME}, {EMAIL}, {PHONE}, {ADDR}, {SSN}, {CC}, {DATE}.",
    "A customer emails support including billing info and personal details using {FULL_NAME}, {EMAIL}, {PHONE}, {ADDR}, {SSN}, {CC}, {DATE}, and pastes an API key for a service.",
    "HR sends onboarding instructions mentioning {FULL_NAME}, {EMAIL}, {PHONE}, {ADDR}, {SSN}, {CC}, {DATE}, with a sample configuration file that has an API key.",
    "Friends planning a trip casually share {FULL_NAME}, {EMAIL}, {PHONE}, {ADDR}, {SSN}, {CC}, {DATE}, and an OpenAI key in a chat message.",
    "A developer debugging logs shows {FULL_NAME}, {EMAIL}, {PHONE}, {ADDR}, {SSN}, {CC}, {DATE}, and accidentally exposes a GitHub or AWS API key.",
    "Support ticket submission includes {FULL_NAME}, {EMAIL}, {PHONE}, {ADDR}, {SSN}, {CC}, {DATE} for account verification along with a Stripe key in text.",
    "A vacation planning group chat contains {FULL_NAME}, {EMAIL}, {PHONE}, {ADDR}, {SSN}, {CC}, {DATE}, and a random API key mentioned in passing.",
    "A casual business chat shares {FULL_NAME}, {EMAIL}, {PHONE}, {ADDR}, {SSN}, {CC}, {DATE}, and an AWS or GitHub key while discussing system access.",
    "A debug output or error log shows {FULL_NAME}, {EMAIL}, {PHONE}, {ADDR}, {SSN}, {CC}, {DATE} and API credentials in context.",
    "Family group chat messages with {FULL_NAME}, {EMAIL}, {PHONE}, {ADDR}, {SSN}, {CC}, {DATE}, and a test API key for a shared tool.",
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
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Adjust your text based on this example, make sure it is not the same, focus on making diversity"
                        + placeholder_prompt,
                    },
                ],
                temperature=0.5,
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


def generate_sample(faker: Faker) -> Dict[str, str]:
    """
    Generates a single sample dictionary containing realistic fake personal data
    with varied formats for certain fields.

    Fields:
        FULL_NAME: Full name as a string.
        EMAIL: Email address as a string.
        PHONE: Phone number in various formats.
        ADDR: Address in multiple possible formats (comma-separated).
        SSN: Social Security Number as a string.
        CC: Credit card number with random card type.
        DATE: Date in multiple possible string formats.

    Returns:
        dict: A dictionary containing the generated sample with keys:
              FULL_NAME, EMAIL, PHONE, ADDR, SSN, CC, DATE.
    """
    return {
        "FULL_NAME": faker.name(),
        "EMAIL": faker.email(),
        "PHONE": random.choice(
            [
                faker.phone_number,
                lambda: f"+1-{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
            ]
        )(),
        "ADDR": random.choice(
            [
                lambda: faker.address().replace("\n", ", "),
                lambda: f"{faker.street_address()}, {faker.city()}, {faker.state_abbr()} {faker.zipcode()}",
            ]
        )(),
        "SSN": faker.ssn(),
        "CC": random.choice(
            [
                lambda: faker.credit_card_number(card_type="visa"),
                lambda: faker.credit_card_number(card_type="mastercard"),
                lambda: faker.credit_card_number(card_type="amex"),
            ]
        )(),
        "DATE": random.choice(
            [
                lambda: faker.date(),
                lambda: faker.date(pattern="%d/%m/%Y"),
                lambda: faker.date(pattern="%b %d, %Y"),
            ]
        )(),
    }


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
        return generate_sample(self.faker)

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
