# Inference recipes

Конфиги сгруппированы по движку, модели и особенностям запуска.
Все самостоятельные recipes включают SMG и Collector с фильтрацией содержимого.
Локальные варианты также включают Jaeger; большие отправляют traces во внешнюю
инфраструктуру. OWUI и LiteLLM для больших моделей уже должны быть настроены.

| Движок / рецепт | Телеметрия | Особенности |
| --- | --- | --- |
| [sglang/deepseek-v4-flash-vision-hgx-h200](sglang/deepseek-v4-flash-vision-hgx-h200/README.md) | внешний OTLP | 8×H200, DPA8/TP8, SSD HiCache |
| [sglang/glm-5.1-nvfp4-hicache](sglang/glm-5.1-nvfp4-hicache/README.md) | внешний OTLP | TP8/DP8, HiCache RAM, MTP; legacy flags |
| [sglang/glm-5.1-nvfp4-mooncake-b200](sglang/glm-5.1-nvfp4-mooncake-b200/README.md) | внешний OTLP | TP8/DP8, Mooncake SSD, B200; legacy flags |
| [sglang/glm-5.1-nvfp4-mooncake-host](sglang/glm-5.1-nvfp4-mooncake-host/README.md) | внешний OTLP | TP8, Mooncake SSD, host networking; legacy flags |
| [sglang/glm-5.1-nvfp4-mooncake-separate](sglang/glm-5.1-nvfp4-mooncake-separate/README.md) | внешний OTLP | отдельные Mooncake master/store, большой RAM/SSD budget |
| [sglang/glm-5.1-nvfp4-mooncake-separate-small](sglang/glm-5.1-nvfp4-mooncake-separate-small/README.md) | внешний OTLP | отдельные Mooncake master/store, меньший RAM budget |
| [sglang/glm-5.1-nvfp4-smg](sglang/glm-5.1-nvfp4-smg/README.md) | внешний OTLP | TP8/DP8, manual affinity через SMG |
| [sglang/kimi-k2.6-eagle3-hicache](sglang/kimi-k2.6-eagle3-hicache/README.md) | внешний OTLP | TP8/DP8, HiCache RAM, EAGLE3 |
| [sglang/qwen3-8b-dpa](sglang/qwen3-8b-dpa/README.md) | локальный Jaeger | TP2/DP2, FP8 KV; нужны две GPU |
| [vllm/deepseek-v4-flash-vision-hgx-h200-lmcache](vllm/deepseek-v4-flash-vision-hgx-h200-lmcache/README.md) | внешний OTLP | 8×H200, DEP8, общий LMCache RAM/SSD |
| [vllm/diffusiongemma-26b-a4b-int8-tiered-kv](vllm/diffusiongemma-26b-a4b-int8-tiered-kv/README.md) | локальный Jaeger | A100 и отдельный 4090 canary, INT8, RAM/SSD KV |
| [vllm/gemma4-26b-a4b-tiered-kv](vllm/gemma4-26b-a4b-tiered-kv/README.md) | локальный Jaeger | A100 80 GB, RAM/SSD KV, base и MTP |
| [vllm/kimi-k2.6-dep8-eagle3](vllm/kimi-k2.6-dep8-eagle3/README.md) | внешний OTLP | TP1/DP8/EP8, RAM KV offload, EAGLE3, text-only |
| [vllm/kimi-k2.6-tp8](vllm/kimi-k2.6-tp8/README.md) | внешний OTLP | TP8, prefix cache, multimodal |
| [sglang/qwen3.8-27b-fp8-hicache](sglang/qwen3.8-27b-fp8-hicache/README.md) | локальный Jaeger | одна GPU 48 GiB, FP8 KV, SSD HiCache, MTP, gRPC/KV events |

## Общие компоненты и миграция

- [Телеметрия и интеграция с существующей инфраструктурой](misc/telemetry/README.md).
- [Sticky routing probes](misc/sticky-sessions/README.md), [нагрузка на cache](misc/prefix-cache-load/README.md), [фильтр OWUI](misc/openwebui-filter/README.md).
- [Исторические заметки](misc/notes/README.md); embedded примеры в заметках сохраняют исторический статус.
- [Таблица старых и новых путей](misc/layout-migration.json).

Серверные `/data/...` и `/opt/...` каталоги автоматически не перемещаются.
Перед запуском из новой папки сохраните исходный Compose project name и задайте
абсолютные пути для существующих bind mounts. Новая директория с относительным
`./cache` создаёт другой cache; старые данные при этом остаются на месте.
Проверенный стенд bx по-прежнему находится в `/data/scratch/qwen38-sglang-smg`.
Для выборочного копирования recipes учитывайте общую папку `misc/telemetry`;
проще клонировать весь репозиторий. Публичный default restart — `unless-stopped`;
локальный эксперимент использует `.env` с `RESTART_POLICY=no`.

GLM recipes в этой ветке содержат GLM-5.1. Рекомендации GLM-5.2 из исторических
заметок не означают наличие проверенного GLM-5.2 Compose в этом каталоге.
