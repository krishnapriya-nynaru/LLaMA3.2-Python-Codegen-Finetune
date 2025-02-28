import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import nltk
import nltk
nltk.download('punkt_tab')


# Load evaluation results
with open("evaluation_results.json", "r") as file:
    data = json.load(file)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Smoothing function for BLEU (useful for short texts)
smoothie = SmoothingFunction().method1

# Store scores
results = []

for sample in data:
    reference = sample["output"]  # Ground truth
    fine_tuned_response = sample["model response"]  # From fine-tuned LLaMA
    original_response = sample["original model response"]  # From original LLaMA

    # Tokenize
    ref_tokens = nltk.word_tokenize(reference)
    ft_tokens = nltk.word_tokenize(fine_tuned_response)
    orig_tokens = nltk.word_tokenize(original_response)

    # Compute BLEU scores
    bleu_ft = sentence_bleu([ref_tokens], ft_tokens, smoothing_function=smoothie) * 100
    bleu_orig = sentence_bleu([ref_tokens], orig_tokens, smoothing_function=smoothie) * 100

    # Compute ROUGE scores
    rouge_ft = scorer.score(reference, fine_tuned_response)["rougeL"].fmeasure * 100
    rouge_orig = scorer.score(reference, original_response)["rougeL"].fmeasure * 100

    # Append results
    results.append({
        "instruction": sample["instruction"],
        "bleu_fine_tuned": bleu_ft,
        "bleu_original": bleu_orig,
        "rouge_fine_tuned": rouge_ft,
        "rouge_original": rouge_orig
    })

# Save validation results
with open("validation_results.json", "w") as file:
    json.dump(results, file, indent=4)

# Print summary
avg_bleu_ft = sum(r["bleu_fine_tuned"] for r in results) / len(results)
avg_bleu_orig = sum(r["bleu_original"] for r in results) / len(results)
avg_rouge_ft = sum(r["rouge_fine_tuned"] for r in results) / len(results)
avg_rouge_orig = sum(r["rouge_original"] for r in results) / len(results)

print("\n **Evaluation Summary** ")
print(f" Average BLEU Score - Fine-tuned Model: {avg_bleu_ft:.2f}")
print(f" Average BLEU Score - Original Model: {avg_bleu_orig:.2f}")
print(f" Average ROUGE Score - Fine-tuned Model: {avg_rouge_ft:.2f}")
print(f" Average ROUGE Score - Original Model: {avg_rouge_orig:.2f}")

# Print result for first sample
print("\n **Sample Evaluation**")
print(f" Instruction: {results[0]['instruction']}")
print(f" Fine-tuned BLEU: {results[0]['bleu_fine_tuned']:.2f} | ROUGE: {results[0]['rouge_fine_tuned']:.2f}")
print(f" Original BLEU: {results[0]['bleu_original']:.2f} | ROUGE: {results[0]['rouge_original']:.2f}")
