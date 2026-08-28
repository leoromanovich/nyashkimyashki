# Gemma 4 26B-A4B на одной A100 80 GB

Локальный vLLM `v0.26.0-cu129` profile для mixed chat, Bash/code, документов и
изображений. Runtime работает offline, API слушает localhost и требует key.

## Профиль

| Параметр | Default | Причина |
| --- | ---: | --- |
| Checkpoint | `google/gemma-4-26B-A4B-it`, BF16 | Native A100 Tensor Core path и quality control |
| Context | 32768 | Баланс длинных документов и concurrent KV capacity |
| Active sequences | 24 | Continuous batching для нескольких десятков пользователей |
| Batched tokens | 8192 | Chunked prefill без монополизации scheduler длинным документом |
| HBM fraction | 0.90 | Веса около 48 GiB плюс GPU KV и runtime buffers |
| Images/request | 4 | Несколько страниц документа или screenshots |
| Image budget | 1120 soft tokens | OCR и мелкий текст; снижайте до 280/560 для throughput |
| RAM KV | 128 GiB | Pinned prefix tier |
| NVMe floor | 512 GiB free | Filesystem tier startup gate |

Модель содержит 25.2B total / 3.8B active parameters. Все веса находятся в
HBM, MoE routing активирует около 4B на token. A100/SM80 выполняет BF16 нативно.
FP8/NVFP4 checkpoints рассчитаны на Hopper/Blackwell. W4A16 Gemma 4 MoE
остаётся quality/kernel canary из-за GELU expert path.

Модель поддерживает text, image, thinking, tool calling и JSON schema.
Audio отсутствует. PDF передавайте как извлечённый text или изображения страниц.

## Подготовка

Примите Gemma license на Hugging Face и заранее материализуйте оба checkpoints:

```bash
hf download google/gemma-4-26B-A4B-it \
  --local-dir /data/models/gemma-4-26B-A4B-it
hf download google/gemma-4-26B-A4B-it-assistant \
  --local-dir /data/models/gemma-4-26B-A4B-it-assistant
```

Download выполняется отдельно от runtime. Контейнер получает локальные модели
через read-only mounts; `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1` запрещают
сетевой fallback.

```bash
cd gemma4_a4b_vllm
cp .env.example .env
mkdir -p /data/vllm-cache /data/vllm-kv-cache
chmod 700 /data/vllm-cache /data/vllm-kv-cache
python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
```

Запишите сгенерированное значение в `VLLM_API_KEY`. Проверьте пути, RAM, NVMe,
GPU, driver, target checkpoint:

```bash
./preflight.sh .env base
```

## Base service

```bash
docker compose --env-file .env config --quiet
docker compose --env-file .env up -d gemma4
docker compose --env-file .env logs -f gemma4
python3 smoke.py --structured --tool --image screenshot.png
```

Base service — production candidate. `async-scheduling`, chunked prefill и
continuous batching оптимизируют aggregate throughput. Thinking выключен по
default и включается per request:

```json
{"chat_template_kwargs":{"enable_thinking":true}}
```

Vision content размещайте перед text. Для OCR оставляйте 1120 soft tokens;
обычная классификация изображений обычно укладывается в 280–560.

Те же действия доступны через executable Control Center app:

```bash
cd /path/to/llm_engines
export VLLM_GEMMA4_ENV_FILE=/absolute/path/to/gemma4_a4b_vllm/.env
./cc feature gemma4-a100 run nyashkimyashki-vllm-gemma4-a4b-control preflight

export VLLM_GEMMA4_CONFIRM=mutate-vllm-gemma4-a4b
./cc feature gemma4-a100 run nyashkimyashki-vllm-gemma4-a4b-control up -d
```

Для MTP добавьте `VLLM_GEMMA4_MODE=mtp`. Runtime mutations требуют отдельный
confirmation value; `config`, `preflight`, `smoke` и `benchmark` read-only.

## MTP speculative canary

`google/gemma-4-26B-A4B-it-assistant` — официальный BF16 drafter около 0.4B.
vLLM подключает его через Gemma 4 MTP path, target и drafter используют общий
KV. Google заявляет ускорение до 3x; фактический gain задают acceptance,
concurrency, output length и A100 kernels.

MTP выделен в profile. Это сохраняет простой rollback при startup/runtime
ошибках OffloadingConnector или слабом throughput gain.

```bash
./preflight.sh .env mtp
docker compose --env-file .env stop gemma4
docker compose --env-file .env --profile mtp up -d gemma4-mtp
docker compose --env-file .env --profile mtp logs -f gemma4-mtp
python3 smoke.py --structured --tool --image screenshot.png
```

Проверка throughput выполняется последовательно на одном GPU:

```bash
# Base запущен, cache прогрет.
python3 benchmark.py --label base-c24 --concurrency 24 --requests 96 \
  --output base-c24.json

# После переключения на gemma4-mtp и прогрева.
python3 benchmark.py --label mtp-c24 --concurrency 24 --requests 96 \
  --output mtp-c24.json
```

Повторите concurrency `1`, `4`, `16`, `24` с реальным JSONL corpus через
`--prompt-file`. Каждая строка содержит `{"messages":[...]}`.

Production gate для MTP:

- output tokens/s выше base минимум на 10% в целевом concurrency;
- p95 E2E не ухудшается больше чем на 10%;
- acceptance rate стабильна и составляет минимум 0.5;
- text, vision, tool и structured smoke проходят;
- часовой burst canary проходит без engine restart/OOM.

При провале gate верните base:

```bash
docker compose --env-file .env --profile mtp stop gemma4-mtp
docker compose --env-file .env up -d gemma4
```

## KV hierarchy

Путь: GPU prefix cache → 128 GiB pinned host RAM → filesystem tier `/kv-cache`.
GPU↔RAM transfers идут asynchronously; NVMe transfers staging проходят через
RAM. LRU и `offload_prompt_only=true` сохраняют повторные system prompts,
репозитории, документы и chat prefixes. Decode KV активного ответа остаётся в
GPU pool.

Filesystem tier vLLM не имеет capacity limit. Разместите `KV_CACHE_DIR` на
отдельной filesystem с quota, стартово 2 TiB. Нужны alerts по free space,
p95/p99 latency, NVMe writes и external prefix hit rate. Очистка cache требует
отдельного maintenance action при остановленном container.

`PYTHONHASHSEED=0` стабилизирует block keys после restart. Не задавайте
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: CUDA VMM может нарушить
pinned KV pages OffloadingConnector.

## Проверка Compose argv

После любого изменения `docker-compose.yaml`:

```bash
docker compose --env-file .env --profile mtp config --quiet
docker compose --env-file .env --profile mtp config --format json \
  | jq '.services | with_entries(.value = .value.command)'
```

Каждый flag должен быть отдельным argv без окружающих пробелов. Service command
записан через folded scalar `command: >-`, один flag на строку.

## Наблюдаемость и tuning

Основные метрики:

- `vllm:prompt_tokens_cached_total`;
- `vllm:external_prefix_cache_hits_total`;
- `vllm:kv_offload_load_bytes_total` и `store_bytes_total`;
- `vllm:mm_cache_hits_total`;
- `vllm:spec_decode_num_draft_tokens_total` и `accepted_tokens_total`;
- running/waiting requests, prompt/output throughput, GPU KV usage;
- host `MemAvailable`, pinned memory и NVMe free/write latency.

Начните с defaults. Для коротких chat prompts сравните `MAX_MODEL_LEN=16384` и
`MAX_NUM_SEQS=32`. Для 64K документов установите `MAX_MODEL_LEN=65536`; ожидайте
меньшую effective concurrency. Высокий vision budget увеличивает prefill cost.

RAM gate требует `KV_CPU_GIB + 32 GiB` свободной памяти. Уменьшение RAM tier до
64 GiB допустимо после проверки cache hit rate. NVMe помогает только повторным
prefixes; уникальные документы создают writes без latency gain.

## Security

- API bind: `127.0.0.1`.
- Auth: `VLLM_API_KEY` в environment; key отсутствует в argv/startup command.
- Model mounts: read-only.
- Runtime: offline, HF token отсутствует.
- Request/output logging выключен default vLLM-конфигурацией.
- Production exposure требует TLS/auth gateway, rate limits и request-size caps.
