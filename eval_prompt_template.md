# Video Knowledge Graph QA Evaluation

## Instructions

You are given:
1. A **Video Knowledge Graph (VKG)** summary for video `Cm73ma6Ibcs`
2. **18 multiple-choice questions** about the video

Your task: Answer each question using ONLY the VKG data provided. For each question:
- Identify which VKG nodes/edges provide evidence
- State your answer letter (A/B/C/D)
- Explain your reasoning in 1-2 sentences

**Do NOT guess.** If the VKG doesn't contain enough evidence, say "insufficient evidence" and make your best inference.

## Output Format

For each question, output:
```
Q[uid]: [Your answer letter]
Evidence: [Which VKG nodes/edges you used]
Reasoning: [Why you chose this answer]
```

## VKG Context

Load `vkg_summary_for_eval.json` for the graph data.

## Questions

Load `questions_no_answers.json` for the 18 questions.
