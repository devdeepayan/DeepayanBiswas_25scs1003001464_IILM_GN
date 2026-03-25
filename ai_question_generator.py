
"""
AI Question Generator using T5 (Text-to-Text Transfer Transformer)

Author: Deepayan Biswas
Description:
Generates questions from a given paragraph using a pre-trained T5 model.
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_model(model_name="valhalla/t5-small-qg-prepend"):
    """Load tokenizer and model."""
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def generate_questions(text, tokenizer, model, num_questions=5):
    """Generate questions from input text."""
    input_text = "generate questions: " + text

    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    outputs = model.generate(
        inputs,
        max_length=64,
        num_beams=5,
        num_return_sequences=num_questions,
        early_stopping=True
    )

    questions = [
        tokenizer.decode(o, skip_special_tokens=True).strip()
        for o in outputs
    ]

    # Remove duplicates
    seen = set()
    final_qs = [q for q in questions if q not in seen and not seen.add(q)]

    return final_qs


def main():
    print("=== AI Question Generator (T5 Model) ===\n")

    paragraph = input("Enter your paragraph:\n")

    tokenizer, model = load_model()

    questions = generate_questions(paragraph, tokenizer, model)

    print("\nGenerated Questions:\n")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")


if __name__ == "__main__":
    main()
