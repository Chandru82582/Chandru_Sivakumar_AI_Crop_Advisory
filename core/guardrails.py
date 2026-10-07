"""Safety guardrails and validation for the advisory system."""


class AdvisoryGuardrails:
    """Implements safety checks and validation for advisory generation."""

    @staticmethod
    def validate_features(features: dict) -> tuple[bool, list[str]]:
        """Validate input features for safety.

        Args:
            features: Dictionary of feature values

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        critical_fields = ["N", "P", "K", "pH", "avg_temp_c", "total_rainfall_mm"]
        missing = [f for f in critical_fields if f not in features]
        if missing:
            warnings.append(f"Missing critical fields: {missing}")

        if "pH" in features:
            pH = features["pH"]
            if not (0 <= pH <= 14):
                warnings.append(f"pH value {pH} outside valid range [0, 14]")

        if "avg_humidity_percent" in features:
            h = features["avg_humidity_percent"]
            if not (0 <= h <= 100):
                warnings.append(f"Humidity {h}% outside valid range [0, 100]")

        numeric_fields = ["N", "P", "K", "total_rainfall_mm", "area"]
        for field in numeric_fields:
            if field in features and features[field] < 0:
                warnings.append(f"{field} cannot be negative")

        return (len(warnings) == 0, warnings)

    @staticmethod
    def sanitize_advisory_text(text: str) -> str:
        """Remove potentially harmful content from advisory text.

        Args:
            text: Generated advisory text

        Returns:
            Sanitized text
        """
        lines = text.split('\n')
        sanitized = []

        for line in lines:
            if line.strip() and not _contains_harmful_content(line):
                sanitized.append(line)

        return '\n'.join(sanitized)


def _contains_harmful_content(text: str) -> bool:
    """Check if text contains potentially harmful content.

    Args:
        text: Text to check

    Returns:
        True if harmful content detected
    """
    harmful_patterns = [
        "toxic",
        "poison",
        "kill",
        "death",
        "banned",
        "illegal"
    ]

    text_lower = text.lower()
    return any(pattern in text_lower for pattern in harmful_patterns)
