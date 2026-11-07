# Sources and verification notes (2026-11-07)

Primary sources:

1. Moonshot AI Kimi K3.1 Pro official model card: https://huggingface.co/moonshotai/Kimi-K3.1-Pro
2. Kimi K3.1 Pro license: https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/main/LICENSE
3. Mistral model documentation: https://docs.mistral.ai/models/mistral-large-4-0
4. Hugging Face Mistral architecture and release breakdown: https://huggingface.co/blog/mistral-large4
5. Tencent full Hy4 announcement: https://www.tencent.com/zh-hk/tencent-launches-hy4-joining-the-ranks-of-the-worlds-leading-open-source-ai-models/
6. Tencent Hy4 preview code / model card: https://github.com/Tencent-Hunyuan/Hy4-preview
7. Thinking Machines Inkling 2 Small launch: https://thinkingmachines.ai/news/inkling-2-small/
8. Thinking Machines Inkling 2 Small model card: https://thinkingmachines.ai/model-card/inkling-2-small/
9. Artificial Analysis Index v5 official methodology update: https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v5
10. vLLM expert parallel docs: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/

## Caveats

- Kimi and Mistral published weights by verification date. Full Tencent Hy4 weights were announced for future release; Hy4 *preview* checkpoint was already public. Inkling 2 Small had hosted access; Apache 2.0 weights were promised for the following week.
- The 49B vs 52B Mistral active parameter figures use different counting conventions. The 52B figure includes embedding and output layers as described in Hugging Face docs.
- The four parameter totals in this lab are provider-reported rounded numbers. Weight memory is in decimal GB (10^9 bytes), not GiB (2^30 bytes).
- 4-bit numerical results are theoretical lower bounds. Mixed precision, scales, tensor layout and MTP/vision modules can change actual on-disk and resident footprints.
- Artificial Analysis index scores are version-specific and can vary by provider / effort setting. This lab does not reproduce scores.

- NVIDIA NVFP4 architecture note: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference . Native accelerated NVFP4 is a Blackwell Tensor Core feature; comparing ideal 4-bit storage to H100 memory is not a claim of H100 NVFP4 support.
