from typing import Dict


class PIILabel:
    """
    Maintains BIO tag labels for PII entity recognition.

    Attributes:
        LABEL_LIST: List of BIO labels for various PII categories.

    Class Methods:
        get_label_map(): Returns a label-to-index mapping.
        get_id_to_label(): Returns an index-to-label mapping.
    """

    LABEL_LIST = [
        "O",  # Outside
        "B-PER",  # Begin Person
        "I-PER",  # Inside Person
        "B-EMAIL",  # Begin Email
        "B-PHONE",  # Begin Phone
        "I-PHONE",  # Inside Phone
        "B-ADDR",  # Begin Address
        "I-ADDR",  # Inside Address
        "B-SSN",  # Begin SSN
        "I-SSN",  # Inside SSN
        "B-CC",  # Begin Credit Card
        "B-DATE",  # Begin Date
        "I-DATE",  # Inside Date
    ]

    @classmethod
    def get_label_map(cls) -> Dict[str, int]:
        """Returns a dict mapping each label to its index."""
        return {label: i for i, label in enumerate(cls.LABEL_LIST)}

    @classmethod
    def get_id_to_label(cls) -> Dict[int, str]:
        """Returns a dict mapping each index to its label."""
        return {i: label for i, label in enumerate(cls.LABEL_LIST)}
