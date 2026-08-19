import re
from pydantic import BaseModel
from detoxify import Detoxify
from transformers import pipeline


class SecurityReport(BaseModel):
    is_safe: bool
    reason: str | None = None
    anonymized_prompt: str | None = None


class GuardrailGateway:

    def __init__(self):
        # 1. Protect AI Prompt Injection Model
        self.injection_classifier = pipeline(
            "text-classification",
            model="protectai/deberta-v3-base-prompt-injection-v2",
            device=-1,
        )
        # 2. Detoxify Model
        self.toxicity_model = Detoxify("original")
        self.TOXICITY_THRESHOLD = 0.70

    def _mask_pii(self, text: str) -> str:
        text = re.sub(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "[REDACTED_EMAIL]",
            text,
        )
        text = re.sub(r"\b\+?[0-9]{10,15}\b", "[REDACTED_PHONE]", text)
        text = re.sub(
            r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", "[REDACTED_IBAN]", text
        )
        return text

    def verify_input(self, user_prompt: str) -> SecurityReport:
        # Step 1: Prompt Injection Check
        injection_result = self.injection_classifier(user_prompt)[0]
        if (
            injection_result["label"].upper() == "INJECTION"
            and injection_result["score"] > 0.50
        ):
            return SecurityReport(
                is_safe=False,
                reason=(
                    "SECURITY_ALERT: Prompt injection payload detected"
                    f" (Confidence: {injection_result['score']:.2f})."
                ),
            )

        # Step 2: Toxicity Check
        tox_scores = self.toxicity_model.predict(user_prompt)
        if any(
            score > self.TOXICITY_THRESHOLD for score in tox_scores.values()
        ):
            return SecurityReport(
                is_safe=False,
                reason=(
                    "POLICY_VIOLATION: Input contains vulgarity or unsafe"
                    " content."
                ),
            )

        # Step 3: PII Masking
        cleaned_prompt = self._mask_pii(user_prompt)
        return SecurityReport(is_safe=True, anonymized_prompt=cleaned_prompt)
