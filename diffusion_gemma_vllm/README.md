# DiffusionGemma vLLM на одной GPU

vLLM `0.26.0`, OpenAI-compatible text/vision API, INT8 W8A8 и cache hierarchy
GPU → 128 GiB RAM → NVMe. Default рассчитан на A100 80 GB.

## Выбор checkpoint

| Профиль | Вес | A100 | Решение |
| --- | ---: | --- | --- |
| `aidendle94/...-INT8-dynamic` | 25.3 GiB | Ampere Triton INT8 MoE | default: лучший доступный баланс качества и HBM |
| `cyankiwi/...-AWQ-INT4` | 16.0 GiB | W4A16, group 32 | A/B после text+vision quality gate |
| `pixelkaiser/...-AWQ-MLP-W4A16-G64-S32-L1024` | 16.8 GiB | проверен на RTX A6000 | A/B после quality gate |
| `google/diffusiongemma-26B-A4B-it` | 48.1 GiB | A100 80 GB | контроль качества |
| FP8 dynamic | 25.3 GiB | fallback kernels, известный Ampere MoE blocker | исключён |
| NVFP4 | 16.8–17.5 GiB | Hopper/Blackwell path | исключён |

INT8 checkpoint оставляет vision tower, routers, embeddings и
self-conditioning в BF16. Опубликованные cosine/error значения измеряют
восстановление весов; итоговое качество подтверждает GPU canary.

## Запуск

1. Примите лицензию Gemma и скачайте весь snapshot INT8 checkpoint.
2. Укажите его корень в `MODEL_DIR`. В корне должны лежать `config.json`,
   tokenizer/processor files, safetensors index и все shards. Compose монтирует
   этот каталог read-only как `/models`.
3. Создайте `VLLM_CACHE_DIR` и `KV_CACHE_DIR`. Разместите `KV_CACHE_DIR` на
   отдельном quota-bounded NVMe filesystem.
4. Подготовьте env и проверьте host. `.env.example` — A100 80 GB:

```bash
cp .env.example .env
$EDITOR .env
./preflight.sh
docker compose config
docker compose pull
docker compose up -d
docker compose logs -f diffusiongemma
```

Runtime работает с `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1`; checkpoint
через сеть не загружается. Первый запуск компилирует kernels, поэтому healthcheck
даёт до 30 минут. API слушает `127.0.0.1:8000` и требует `VLLM_API_KEY`.
Ключ передаётся через environment: CLI argv и startup log его не содержат.

```bash
export VLLM_API_KEY='значение-из-.env'
./smoke.py
./smoke.py --image /path/to/screenshot.png
```

Smoke повторяет text и image запросы и выводит дельты prefix, external KV,
RAM/filesystem transfer и multimodal processor cache metrics.

Порядок vision content: изображение, затем текст. Для screenshot/OCR подходит
`MM_MAX_SOFT_TOKENS=1120`; обычным изображениям часто хватает `560`.
Структурированные outputs/json schema пока недоступны для DiffusionGemma.
Субагенту стоит возвращать компактный Markdown: наблюдения, читаемый текст,
ошибки, следующие проверки и уверенность по каждому OCR-фрагменту. Нечитаемые
фрагменты помечаются явно; наблюдения отделяются от выводов.

## Профиль A100 80 GB

Default:

```dotenv
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.85
```

Diffusion state резервирует крупный FP32 buffer на каждую concurrent sequence.
Поднимайте concurrency по одной после проверки пикового HBM и p99.

Аварийный low-HBM профиль: `16384/2/4096/0.80`.

## Профиль RTX 4090 48 GB

Canary-профиль учитывает меньшие RAM/disk budgets:

```bash
cp .env.4090.example .env
$EDITOR .env
mkdir -p model cache kv-cache
./preflight.sh
```

```dotenv
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=4096
GPU_MEMORY_UTILIZATION=0.82
```

Cache hierarchy: GPU → 24 GiB RAM → local NVMe. В `/data/scratch` требуется
минимум 64 GiB свободного места. Короткий canary ограничивает число запросов;
длительная эксплуатация требует quota для `KV_CACHE_DIR`.

Проверено на RTX 4090 48 GB с vLLM 0.26.0: compressed-tensors W8A8 выбрал
Cutlass INT8 linear и Triton INT8 MoE; weights заняли 25.83 GiB, GPU KV —
12.1 GiB/166k tokens, steady HBM — 41.6 GiB. Text cold/repeat — 5.02/3.10 с;
vision cold/filesystem-cached/MM-hot — 4.11/1.82/1.55 с. После restart NVMe
вернул 239.9 MB KV: 1152 external hits из 1171 queries.

Для W4 A/B замените `VLLM_MODEL`. vLLM читает `compressed-tensors` из checkpoint.
Сравните минимум 100 реальных text prompts и 100 screenshots: task success,
OCR exactness, hallucination rate, TTFT, E2E latency и peak HBM.

## KV и disk cache

Default использует native `OffloadingConnector`: GPU prefix cache → 128 GiB
pinned host RAM → filesystem tier `/kv-cache`. LRU и prompt-only offload полезны
для повторных system/repository prefixes и многократного анализа одной картинки.
`MM_PROCESSOR_CACHE_GB=4` кэширует image preprocessing.

`KV_TRANSFER_CONFIG` уже включает NVMe tier:

```json
{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":137438953472,"eviction_policy":"lru","offload_prompt_only":true,"secondary_tiers":[{"type":"fs","root_dir":"/kv-cache","n_read_threads":32,"n_write_threads":16}]}}
```

`preflight.sh` требует минимум `KV_DISK_MIN_FREE_GB=512`. Сам FS connector не
задаёт capacity limit, поэтому обязательны filesystem quota и disk-space alert.
Стартовая quota — 2 TiB. Следите за hit rate, promotion latency, CPU-cache usage
и NVMe writes через `/metrics`. Фиксированный `PYTHONHASHSEED=0` сохраняет
стабильные block hashes между рестартами.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` запрещён: CUDA VMM может
инвалидировать pinned KV pages OffloadingConnector.

## Диагностика и откат

В startup log ожидаются DiffusionGemma, compressed-tensors INT8 и Triton
attention. Ошибки FP8 Marlin означают неверный checkpoint. При OOM уменьшите
`MAX_NUM_SEQS`, затем `MAX_MODEL_LEN` и `GPU_MEMORY_UTILIZATION`.

```bash
docker compose ps
docker compose logs --tail=300 diffusiongemma
curl -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:8000/metrics
docker compose down
```

`down` сохраняет внешний model directory, compile cache и KV cache. Для внешнего
доступа поставьте TLS reverse proxy. Data URI для изображений уменьшает
SSRF-поверхность относительно произвольных remote URLs.
