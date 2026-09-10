При твоём текущем наборе реализованных моделей следующий приоритет такой:

```text
1. Generic MoE + Qwen3-MoE
2. Quantization / memory path
3. Gemma 3/4 text path
4. GPT-OSS-20B
5. Phi-4 adapter
6. Seed-OSS-36B / GLM-4-32B adapters
7. Mistral Small 3.x / VLM path
8. Qwen3.6-35B-A3B hybrid path
```

Главное: **не DeepSeek-native**. При лимите 35–37B релевантные DeepSeek-модели — это в основном distill-чекпойнты на Qwen/Llama, а они у тебя уже почти закрыты через Qwen/Llama dense. DeepSeek-R1-Distill-Qwen-32B прямо основан на Qwen2.5-32B, так что отдельная DeepSeek-V3/R1 MLA/MoE-архитектура сейчас не даёт нормального ROI. ([Hugging Face][1])

## 1. Первый крупный приоритет: Generic MoE, начиная с Qwen3-30B-A3B

Это самый логичный следующий шаг. У тебя уже есть Qwen3 dense, поэтому **Qwen3-MoE — минимальное архитектурное расширение с максимальным приростом покрытия**.

Целевые модели:

| Модель                           | Почему именно она                                                                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Qwen3-30B-A3B-Instruct-2507**  | 30.5B total / 3.3B active, 48 layers, GQA 32Q/4KV, 128 experts, 8 active, 262K context. Почти идеальный следующий тест после Qwen3 dense. ([Hugging Face][2]) |
| **Qwen3-Coder-30B-A3B-Instruct** | Та же MoE-линия, но coding/agentic target.                                                                                                                    |
| **Gemma 4 26B-A4B**              | Позже, когда MoE уже есть: 25.2B total / 3.8B active, 128 experts, 8 active + 1 shared. ([Hugging Face][3])                                                   |
| **GPT-OSS-20B**                  | Позже: MoE есть, но там ещё MXFP4, harmony format и локально-бандированное attention.                                                                         |

Что реализовать в `SparseMoEBlock`:

```text
router: Linear(hidden_size, num_experts)
top_k selection per token
router softmax / optional renormalization
dispatch tokens by expert
expert MLP: gate/up/down SwiGLU-style
weighted combine
optional shared expert
optional expert bias / scaling hooks
```

Сначала делай **корректный медленный MoE**, потом grouped GEMM/fused dispatch. Не начинай с оптимизации, иначе отладка router/expert mapping будет болезненной.

Минимальный тестовый критерий: logits parity с HF/Transformers на коротком batch, потом prefill/decode parity на 2–3 фиксированных prompt. Для MoE особенно важно проверять не только итоговые logits, но и top-k expert ids/scores на нескольких слоях.

## 2. Второй приоритет: memory/quantization path

Если ты реально хочешь запускать 30B–36B модели, BF16-only быстро станет ограничением. 32B dense в BF16 — это примерно 64GB только веса, без KV cache и runtime buffers. Поэтому после MoE или параллельно с ним стоит делать:

```text
sharded safetensors loading
lazy/streamed loading
weight tying correctness
INT4/AWQ/GPTQ path
FP8 path, если целишься в серверные GPU
paged KV cache
sliding-window KV eviction
tensor parallel hooks, хотя бы skeleton
```

Для MoE это особенно важно: total parameters всё равно надо загрузить, даже если active parameters маленькие. Google прямо подчёркивает это для Gemma 4 26B-A4B: активируется 4B, но все 26B параметров должны быть в памяти для быстрого routing/inference. ([Google AI for Developers][4])

## 3. Третий приоритет: Gemma 3/4 text path

После Qwen3-MoE я бы делал Gemma. Причина: это популярная локальная ветка, но она **не является простым Llama/Qwen clone**.

Целевые модели:

| Модель              |                  Размер | Что нужно                                                        |
| ------------------- | ----------------------: | ---------------------------------------------------------------- |
| **Gemma 3 27B**     |               27B dense | local/global attention, sliding window, QK-norm, Gemma tokenizer |
| **Gemma 4 31B**     |             30.7B dense | hybrid local/global attention, p-RoPE, 256K context              |
| **Gemma 4 26B-A4B** | 25.2B MoE / 3.8B active | Gemma text path + generic MoE + shared expert                    |

Gemma 3 использует повторяющийся паттерн из **5 local attention layers со sliding window 1024 + 1 global layer**, а также GQA и QK-norm. Это отдельная attention-mask/KV-cache задача, а не просто новый mapper. ([developers.googleblog.com][5])

Gemma 4 ещё интереснее: есть dense 31B и MoE 26B-A4B, medium-модели поддерживают 256K context, а архитектура использует interleaved local sliding attention + global attention, unified K/V на global layers и p-RoPE. ([Hugging Face][3])

Рекомендация: сначала **text-only Gemma**. Multimodal frontend отложить.

## 4. Четвёртый приоритет: GPT-OSS-20B

GPT-OSS-20B очень важен по популярности и практическому спросу, но это не лучший первый MoE-target. Его надо делать после generic MoE.

Почему сложнее, чем Qwen3-MoE:

```text
MoE: 21B total / 3.6B active
32 experts / 4 active
alternating dense + locally banded sparse attention
grouped multi-query attention
RoPE
128K context
o200k_harmony tokenizer / harmony response format
MXFP4 quantized MoE weights
```

OpenAI описывает gpt-oss-20b как Transformer MoE с 21B total / 3.6B active, 32 experts, 4 active experts per token, alternating dense and locally banded sparse attention, grouped multi-query attention and 128K context. ([OpenAI][6]) HF model card отдельно указывает, что модели используют harmony response format и MXFP4 quantization для MoE weights; без harmony format модель может работать некорректно. ([Hugging Face][7])

Практический порядок:

```text
1. Реализовать GPT-OSS в BF16/upcast mode для correctness.
2. Проверить harmony template/tokenizer.
3. Добавить locally banded sparse attention.
4. Потом уже MXFP4 fast path.
```

MXFP4 не надо делать первым. Сначала докажи, что forward корректен.

## 5. Пятый приоритет: Phi-4 adapter как быстрый win

Phi-4 — не архитектурно самый важный шаг, но это дешёвое расширение покрытия. Phi-4 — 14B dense decoder-only Transformer с 16K context и MIT license. ([Hugging Face][8])

Почему стоит добавить:

```text
маленький размер;
хороший тест для adapter abstraction;
популярная локальная ветка;
скорее всего, большая часть core уже готова;
полезен для CI/regression tests.
```

Что может отличаться:

```text
tokenizer / chat format
vocab size
qkv layout
embedding tying
Phi-specific config fields
```

Если твой dense core действительно generic, Phi-4 должен быть относительно простым.

## 6. Шестой приоритет: Seed-OSS-36B и GLM-4-32B

Это уже не архитектурные “разблокировщики”, а хорошие coverage adapters.

### Seed-OSS-36B

Seed-OSS-36B прямо попадает в твой лимит и архитектурно выглядит как понятный dense decoder: RoPE, GQA, RMSNorm, SwiGLU, 36B parameters, 64 layers, 80/8/8 QKV heads, 512K context. ([Hugging Face][9])

Почему не раньше:

```text
он не открывает новый класс архитектур;
но хорошо проверяет long-context, large vocab, RoPE base, GQA flexibility.
```

Я бы добавил его после Gemma/GPT-OSS или параллельно с Phi-4, если нужен ещё один сильный dense target около верхней границы 37B.

### GLM-4-32B / GLM-Z1-32B

GLM-4-32B-0414 — 32B модель с сильным фокусом на coding, function calling, agent tasks и локальное deployment; GLM-Z1-32B — reasoning-вариант на той же базе. ([Hugging Face][10])

Почему не раньше:

```text
скорее всего, это отдельный mapper + tokenizer/chat/tool format;
не даёт такого архитектурного прироста, как MoE или Gemma attention;
но полезна для китайского/agentic/coding сегмента.
```

## 7. Седьмой приоритет: Mistral Small 3.x, но только если нужен VLM/conditional path

Ты проверил Mistral dense на `Mistral-7B-v0.1`. Это хорошо, но **не означает автоматическую поддержку Mistral Small 3.x**.

Mistral Small 3.1 — 24B, multimodal, multilingual, с context window до 128K, image understanding и function calling. ([mistral.ai][11])

Если твой scope пока text-only, Mistral Small 3.x можно отложить. Если хочешь первым зайти в multimodal, это хороший кандидат вместе с Gemma 3/4 VLM.

Что понадобится:

```text
Mistral common tokenizer/template handling
conditional generation wrapper
vision encoder/projector path
image token packing
multimodal attention masks
128K context handling
```

## 8. Восьмой приоритет: Qwen3.6-35B-A3B / hybrid state path

Qwen3.6-35B-A3B входит в твой лимит, но это **не “ещё один Qwen3-MoE”**.

Он имеет 35B total / 3B active, vision encoder, 40 layers и layout:

```text
10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))
```

Также там 256 experts, 8 routed + 1 shared expert, MTP и context 262K с расширением до 1,010,000 tokens. ([Hugging Face][12])

Почему поздно:

```text
нужен state cache, не только KV cache;
нужен Gated DeltaNet / linear attention path;
нужен scheduler, который понимает разные типы state;
нужен MoE;
нужен vision path, если хочешь полную модель.
```

Это интересная архитектура, но плохой следующий шаг, если цель — максимальное покрытие при ограниченном времени.

## Что не делать сейчас

### Native DeepSeek

При лимите 35–37B не надо делать DeepSeek MLA/DSA. Большие native DeepSeek-модели вне твоего ограничения, а distill-модели уже закрываются через Qwen/Llama dense. ([Hugging Face][1])

### Mixtral 8x7B

Даже несмотря на 12.9B active, Mixtral 8x7B имеет 46.7B total parameters, то есть выходит за твой total-parameter лимит. Если считать active parameters — другое дело, но ты ранее ограничивал именно размер модели, и для inference engine total parameters важны из-за загрузки весов. ([mistral.ai][13])

### Qwen3.6 / Falcon-H1 / Mamba-like hybrid как следующий шаг

Это отдельный execution model. Не делай до MoE, paged cache и хорошего validation harness.

## Самый практичный порядок

Я бы делал так:

```text
0. Logit-parity harness + ModelSpec/weight-mapper registry
1. SparseMoEBlock
2. Qwen3-30B-A3B-Instruct-2507
3. Qwen3-Coder-30B-A3B
4. Paged/sliding KV + INT4/AWQ/GPTQ baseline
5. Gemma 3 text path
6. Gemma 4 31B dense
7. Gemma 4 26B-A4B MoE
8. GPT-OSS-20B
9. Phi-4 / Phi-4-mini
10. Seed-OSS-36B
11. GLM-4-32B / GLM-Z1-32B
12. Mistral Small 3.x multimodal
13. Qwen3.6-35B-A3B hybrid state path
```

Самый сильный инженерный следующий шаг: **Qwen3-MoE через generic MoE primitive**. Он максимально использует то, что у тебя уже есть, и открывает путь к Qwen3-30B-A3B, Qwen3-Coder-30B-A3B, Gemma4-26B-A4B и GPT-OSS-20B.

[1]: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B?utm_source=chatgpt.com "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
[2]: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507 "Qwen/Qwen3-30B-A3B-Instruct-2507 · Hugging Face"
[3]: https://huggingface.co/google/gemma-4-26B-A4B "google/gemma-4-26B-A4B · Hugging Face"
[4]: https://ai.google.dev/gemma/docs/core "Gemma 4 model overview  |  Google AI for Developers"
[5]: https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/ "
            
            Gemma explained: What’s new in Gemma 3
            
            
            \- Google Developers Blog
            
        "
[6]: https://openai.com/index/introducing-gpt-oss/ "Introducing gpt-oss | OpenAI"
[7]: https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b · Hugging Face"
[8]: https://huggingface.co/microsoft/phi-4 "microsoft/phi-4 · Hugging Face"
[9]: https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct "ByteDance-Seed/Seed-OSS-36B-Instruct · Hugging Face"
[10]: https://huggingface.co/zai-org/GLM-4-32B-0414 "zai-org/GLM-4-32B-0414 · Hugging Face"
[11]: https://mistral.ai/news/mistral-small-3-1 "Mistral Small 3.1 | Mistral AI"
[12]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B "Qwen/Qwen3.6-35B-A3B · Hugging Face"
[13]: https://mistral.ai/news/mixtral-of-experts "Mixtral of experts | Mistral AI"
