from adaptive_consistency import AdaptiveConsistency, BetaStoppingCriteria


def extract_final_answer(model_output: str) -> str:
    # Replace this with the answer extraction used in your experiment.
    return model_output.strip().split()[-1]


def sample_llm(prompt: str) -> str:
    # Replace this with an actual LLM call.
    del prompt
    return "The answer is 42"


ac = AdaptiveConsistency(
    max_samples=40,
    stopping_criteria=BetaStoppingCriteria(confidence=0.95),
    answer_key=extract_final_answer,
)

result = ac.run(sample_llm, "Solve the problem.")

print("winner:", result.winner)
print("samples:", result.num_samples)
print("confidence:", round(result.confidence, 4))
print("counts:", result.counts)
