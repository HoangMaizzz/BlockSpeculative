# Scientific limitations

1. The factorized Fast-dLLM marginal score is a surrogate, not necessarily the exact probability of its iterative diffusion trajectory.
2. Residual sampling is approximate truncated block-level residual sampling over shared Top-K support.
3. The acceptance/residual construction is exact only for normalized truncated distributions `p_S` and `q_S`.
4. Approximation to the full AR target depends on retained probability mass; logged masses do not prove full-space coverage.
5. Tokenizer mismatch invalidates direct block-probability ratios without a principled alignment layer; strict mode stops instead of using the text bridge.
6. Width schedules and drafter confidence are performance heuristics, not correctness proofs.
7. The optional bonus token does not determine residual correctness.
8. Tree drafting currently recomputes remote-code forwards rather than sharing Fast-dLLM cache objects across divergent branches.
9. No quantization is enabled; comparisons must use the same precision.

