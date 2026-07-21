# Algorithm

For each verified prefix, Fast-dLLM produces per-position marginals for a fixed-size block. Heap best-first search finds the best joint tuples without enumerating the vocabulary Cartesian product. Their summed marginal log probabilities define surrogate `q(B|C)` scores.

An asymmetric tree expands candidate paths according to a global width schedule. At depth 2, every active parent receives a minimum child quota before remaining slots are filled globally. All candidates remain in their parent's shared support even if they are not expanded.

Qwen teacher-forces every candidate in a node, yielding autoregressive `p(B|C)`. Both score vectors are normalized with `logsumexp` over that same finite support. A proposal sampled from `q_S` is accepted in log space with `min(0, log p_S - log q_S)`. Rejections sample from normalized positive `p_S-q_S`; the `p_S` fallback is used only for a numerically degenerate residual. If the committed residual candidate has a child, verification continues down that existing subtree.

A single Qwen token may be added only after reaching the end of an expanded path. Every run uses one fixed block size.

