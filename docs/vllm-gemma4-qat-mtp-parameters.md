# vLLM + Gemma 4 QAT/MTP Parameter Rationale

Created: 2026-06-06

This document summarizes the purpose, rationale, tradeoffs, and validation points for the Gemma 4 QAT/MTP parameters used by this vLLM deployment. The main target files are `variables.tf`, `app/docker-compose.yml`, and `templates/cloud-init.yaml.tftpl`.

## Target Files

| Area | Files |
|---|---|
| Terraform variables | `variables.tf` |
| Docker Compose runtime | `app/docker-compose.yml` |
| EC2 bootstrap | `templates/cloud-init.yaml.tftpl` |
| SearXNG settings | `app/searxng-settings.yml` |

## Current Settings

| Item | Value |
|---|---|
| Instance | `g6e.xlarge` |
| GPU | 1 x NVIDIA L40S, 44 GiB accelerator memory |
| CPU/RAM | 4 vCPU / 32 GiB memory |
| Target model | `google/gemma-4-31B-it-qat-w4a16-ct` |
| MTP assistant | `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` |
| vLLM image | `vllm/vllm-openai:v0.22.1` |
| `max_model_len` | `131072` |
| `num_speculative_tokens` | `4` |
| `gpu_memory_utilization` | `0.95` |
| `max_num_seqs` | `1` |
| `kv_cache_dtype` | `fp8` |
| `limit_mm_per_prompt` | `{"image": 1, "audio": 0}` |
| `speculative_config` | `{"method":"mtp","model":"google/gemma-4-31B-it-qat-q4_0-unquantized-assistant","num_speculative_tokens":4}` |
| `chat_template` | `examples/tool_chat_template_gemma4.jinja` |
| `async_scheduling` | enabled |
| Attention backend | Auto-selected by vLLM; do not rely on `VLLM_ATTENTION_BACKEND=FLASHINFER` |
| Web Search | SearXNG search provider, Firecrawl scraper provider, no reranker |

## Assumptions And Operating Policy

| Viewpoint | Details |
|---|---|
| Goal | Run Gemma 4 31B QAT with MTP on a single `g6e.xlarge` instance for long-context private demos |
| Tuning focus | Allocate more GPU memory to KV cache, enable MTP speculative decoding, and allow a 131,072-token context window |
| Primary workload | Text chat and Web Search, with one image per prompt allowed for multimodal demos |
| Main constraint | Fit the 31B QAT model, MTP assistant model, KV cache, speculative decoding overhead, and image-input headroom into a 44 GiB GPU |
| Stability policy | Prefer one long active session with `max_num_seqs=1` instead of maximizing concurrent throughput |
| Rollback style | Reduce context length first, then lower GPU memory utilization or speculative depth if needed |

## Parameter Rationale

| Parameter | Why It Was Chosen | Tradeoff / Rollback | Main Basis |
|---|---|---|---|
| `instance_type=g6e.xlarge` | `g6.xlarge` has an NVIDIA L4 with 22 GiB, which leaves little room for a 31B QAT model, assistant model, and large KV cache. `g6e.xlarge` provides an NVIDIA L40S with 44 GiB while keeping the same 4 vCPU class. | CPU/RAM are still limited to 4 vCPU / 32 GiB, so concurrency and modality use must stay constrained. | AWS accelerated instance specs |
| `model_id=google/gemma-4-31B-it-qat-w4a16-ct` | Google and Hugging Face provide this checkpoint as a compressed-tensors QAT model suitable for vLLM. The 31B dense model favors output quality among Gemma 4 dense choices. | Prioritizes quality over speed. If memory pressure is too high, move to a smaller Gemma 4 model. | Google QAT announcement, HF model card, Unsloth guide |
| `w4a16-ct` | The weight 4-bit / activation 16-bit compressed-tensors format is intended for native optimized vLLM inference. | Do not set `--quantization` unless detection fails; vLLM should read the model config's `quantization_config`. | vLLM engine args, LLM Compressor docs |
| `mtp_assistant_model_id` | vLLM's Gemma 4 MTP path expects a Gemma 4 assistant checkpoint. Because the target model is 31B, the assistant checkpoint is also from the 31B family. | Treat it as `method=mtp`, not as a generic draft model. | vLLM MTP docs |
| `--speculative-config.method=mtp` | Uses Gemma 4's native multi-token prediction capability through vLLM speculative decoding. | If startup logs show the assistant handled as `draft_model`, upgrade or change the vLLM image. | vLLM MTP docs |
| `num_speculative_tokens=4` | vLLM's Gemma 4 recipe uses small MTP depths such as `4` in examples, and this profile keeps the existing MTP assistant checkpoint. | It may not improve speed for every workload. If speed does not improve or memory pressure rises, reduce to `2`, then `1`. | vLLM Gemma 4 recipe, vLLM MTP docs |
| `max_model_len=131072` | The official Gemma 4 recipe often shows smaller launch values such as 16K or 32K, but this deployment keeps `131072` because it has worked in this stack and supports the intended long-context private demo. | On OOM, reduce to `98304` or `65536`. This remains more aggressive than the official recipe's typical examples. | HF model card, vLLM Gemma 4 recipe, vLLM engine args |
| `gpu_memory_utilization=0.95` | The official Gemma 4 recipe lists `0.85-0.95` as the recommended range and uses `0.90` in full-featured examples. This deployment keeps `0.95` because it has worked with the current long-context profile. | Leaves less room for CUDA graphs, speculative decoding, long prefill, and image input. First reduce `max_model_len`, then roll back to `0.92` or `0.90`. | vLLM Gemma 4 recipe, vLLM engine args |
| `max_num_seqs=1` | Prioritizes one long demo session over concurrent multi-user throughput, making memory and scheduling behavior easier to reason about. | Concurrent throughput drops. For multi-user demos, compare `2` with `max_model_len=65536`. | vLLM scheduling behavior |
| `kv_cache_dtype=fp8` | KV cache memory strongly scales with context length and concurrency, so FP8 is a major memory-saving lever on a 44 GiB GPU. | May affect quality or stability. Compare with `auto` only if the context length can be reduced substantially. | vLLM engine args |
| `limit_mm_per_prompt={"image":1,"audio":0}` | Gemma 4 31B supports image-text-to-text. Allow one image for multimodal demos and explicitly disable audio, which is not the main target for the 31B model. | Image input increases memory footprint. Use `image:0` if text-only stability is more important. | Google QAT announcement, vLLM engine args, HF model card |
| `--chat-template=examples/tool_chat_template_gemma4.jinja` | vLLM's Gemma 4 recipe recommends the Gemma 4 tool chat template for reasoning and tool calling with vLLM. | If the Docker image changes and the relative `examples/` path is missing, mount or copy the template explicitly. | vLLM Gemma 4 recipe |
| `--async-scheduling` | vLLM's Gemma 4 recipe recommends async scheduling for throughput by overlapping scheduling with decoding. | Disable only if the selected vLLM image reports an incompatibility at startup. | vLLM Gemma 4 recipe |
| Attention backend auto | vLLM should choose the actual attention backend. `VLLM_ATTENTION_BACKEND=FLASHINFER` is not relied on here, and Gemma 4 may force TRITON_ATTN because of heterogeneous head dimensions. | If an attention backend is specified later, use a supported vLLM option and verify the effective backend in startup logs. | vLLM runtime behavior |
| `vllm_image=vllm/vllm-openai:v0.22.1` | Pinning avoids behavior changes from `latest`. Gemma 4 QAT/MTP depends on relatively new vLLM functionality. | If issues appear, switch to a Gemma 4-specific tag or nightly image. | vLLM Gemma 4 recipe |
| LibreChat Web Search | `SearXNG + Firecrawl + reranker none` keeps Web Search self-hosted and follows LibreChat's search-provider / scraper / reranker structure. SearXNG JSON output is enabled in `app/searxng-settings.yml`. | Without a reranker, result ordering depends more heavily on the search provider and scraper output. Add a reranker only after the base self-hosted path is stable. | LibreChat Web Search docs, local SearXNG settings |

## Parameters Intentionally Left Unset

| Parameter | Why It Is Not Explicitly Set | When To Consider It |
|---|---|---|
| `--quantization` | vLLM checks the model config's `quantization_config` when this is unset. The selected `w4a16-ct` checkpoint is a compressed-tensors QAT model, so model-config detection is preferred. | Consider setting it only if detection fails or model loading breaks. |
| `--tensor-parallel-size` | `g6e.xlarge` has 1 GPU, so the default tensor parallel size of 1 is sufficient. | Consider it only when moving to a multi-GPU instance. |
| CPU offload | The profile assumes the model fits in the 44 GiB GPU. CPU offload can virtually increase available GPU memory, but it adds CPU-GPU transfer on each forward pass. | Use it only as a late fallback if smaller model/context choices are not acceptable. |
| `--enforce-eager` | Useful for isolating CUDA graph memory issues, but it can reduce speed. | Consider it if `0.95` / `131072` / `image:1` triggers CUDA graph-related OOM. |
| `--max-num-batched-tokens` | Can influence long prefill behavior, but defaults depend on the vLLM version and chunked prefill behavior. | Tune it only after first trying `max_model_len`, `gpu_memory_utilization`, and `max_num_seqs`. |

## Logs To Check During Startup And Validation

| Check | What To Look For |
|---|---|
| Model load | `google/gemma-4-31B-it-qat-w4a16-ct` is loaded as a compressed-tensors model |
| Speculative config | Startup logs show `method='mtp'` |
| Assistant handling | The assistant checkpoint is not treated as a generic draft model |
| Chat template | Startup succeeds with `examples/tool_chat_template_gemma4.jinja`, or the template is mounted explicitly if the image path is missing |
| Async scheduling | Startup logs accept `--async-scheduling` without incompatibility warnings |
| Memory | No OOM appears during startup or generation |
| KV cache | Startup logs show KV cache capacity and maximum concurrency |
| Model API | `/v1/models` includes `google/gemma-4-31B-it-qat-w4a16-ct` |
| Chat API | `/v1/chat/completions` succeeds with a short text-only prompt |
| Web Search | LibreChat can call SearXNG and Firecrawl successfully |
| Readiness | `Application startup complete.` appears |

If an older vLLM release treats the Gemma 4 assistant checkpoint as a `draft_model`, upgrade to a vLLM version that supports Gemma 4 MTP.

## Rollback Order For Failures

| Symptom | First Action | Next Action |
|---|---|---|
| Startup OOM | Set `max_model_len=98304` | Set `max_model_len=65536` |
| Generation OOM | Reduce `max_model_len` | Set `gpu_memory_utilization=0.92` |
| MTP does not improve speed | Set `num_speculative_tokens=2` | Set `num_speculative_tokens=1` |
| MTP is treated as `draft_model` | Update the vLLM image | Consider a Gemma 4-specific tag or nightly image |
| Image input is unstable | Set `image:0` | Revalidate in text-only mode |
| Quality regression is concerning | Compare `kv_cache_dtype=auto` | Reduce context length |
| Web Search fails | Check SearXNG JSON output and Firecrawl health | Temporarily validate chat without Web Search |

## Source Notes

| Source | Information Used In This Configuration |
|---|---|
| Google: Gemma 4 with QAT | Compressed tensors for vLLM, modality memory footprint |
| Hugging Face: `google/gemma-4-31B-it-qat-w4a16-ct` | 31B total parameters, 256K context, compressed-tensors tag, image/audio modality notes |
| Hugging Face: assistant checkpoint | Gemma 4 31B MTP assistant model |
| vLLM MTP docs | `method=mtp`, Gemma 4 assistant checkpoint, `num_speculative_tokens`, older-release caveat for `draft_model` |
| vLLM engine args | `max_model_len`, `gpu_memory_utilization`, `kv_cache_dtype`, `limit_mm_per_prompt`, `quantization_config`, CPU offload |
| vLLM / LLM Compressor | W4A16 compressed-tensors scheme |
| vLLM Gemma 4 recipe | Gemma 4 MTP image/version caveats |
| Unsloth Gemma 4 guide | 31B quality-oriented positioning, 32K context as a practical starting point |
| AWS accelerated instance specs | `g6.xlarge` L4 22 GiB and `g6e.xlarge` L40S 44 GiB comparison |
| LibreChat Web Search docs | Search provider, scraper provider, and reranker structure |
| vLLM runtime validation | Effective attention backend behavior and startup log checks |

## Reference URLs

- Google: Gemma 4 with quantization-aware training: https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/
- Unsloth: Gemma 4 QAT: https://unsloth.ai/docs/models/gemma-4/qat
- Unsloth: Gemma 4: https://unsloth.ai/docs/models/gemma-4
- Hugging Face: `google/gemma-4-31B-it-qat-w4a16-ct`: https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct
- Hugging Face: `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant`: https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized-assistant
- vLLM MTP docs: https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/mtp.md
- vLLM engine args: https://docs.vllm.ai/en/v0.10.0/configuration/engine_args.html
- vLLM / LLM Compressor compression schemes: https://docs.vllm.ai/projects/llm-compressor/en/stable/steps/choosing-scheme/
- vLLM Gemma 4 recipe: https://recipes.vllm.ai/Google/gemma-4-E2B-it
- AWS accelerated instance specs: https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html
