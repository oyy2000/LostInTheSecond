from sc_unified import Generation, extract_last_number, run_consistency


generations = [
    "Path A. The answer is 42.",
    "Path B. The answer is 42.",
    "Path C. The answer is 41.",
    "Path D. The answer is 42.",
    "Path E. The answer is 42.",
    "Path F. The answer is 42.",
]

for method in ["sc", "ac", "esc", "dsc"]:
    result = run_consistency(
        method,
        generations=generations,
        answer_extractor=extract_last_number,
        max_samples=6,
        window_size=2,
        confidence=0.95,
    )
    print(method, result.answer, result.samples_used, result.stop_reason, result.counts)


rasc_generations = [
    Generation("Short rationale. The answer is 41.", answer="41", score=0.30),
    Generation("Good rationale. The answer is 42.", answer="42", score=0.80),
    Generation("Another good rationale. The answer is 42.", answer="42", score=0.75),
    Generation("Weak rationale. The answer is 43.", answer="43", score=0.20),
    Generation("Best rationale. The answer is 42.", answer="42", score=0.90),
]

rasc = run_consistency(
    "rasc",
    generations=rasc_generations,
    max_samples=10,
    rasc_threshold=0.7,
    rasc_buffer_size=3,
)
print("rasc", rasc.answer, rasc.samples_used, rasc.stop_reason, rasc.confidence)
