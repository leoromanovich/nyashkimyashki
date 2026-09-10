# Gemma 4 26B-A4B на одной A100 80 GB

## Запуск и трассировка

Состав: **vllm + SMG + Collector+ Jaeger**.
A100 80 GB, RAM/SSD KV, base и MTP. Параметры модели и cache budgets сохранены при переносе.

```bash
cp .env.example .env
# Настройте пути, API keys в .env.
# Для экспериментов добавьте RESTART_POLICY=no.
docker compose config --quiet
docker compose up -d --build
```

Внешний клиент обращается к SMG на `http://<host>:30000/v1`; укажите одинаковый
inference API key в `.env` и LiteLLM. Публичный bind по умолчанию — localhost;
для доступа с другого сервера задайте `SMG_BIND=<LAN IP>`.
Jaeger: http://localhost:16686. История хранится в RAM и теряется при перезапуске.

[Интеграция OWUI/LiteLLM, просмотр запроса и проверочный клиент](../../misc/telemetry/README.md).
[Запуск через Nix](../../misc/telemetry/README.md#nix).


Локальный vLLM `v0.26.0-cu129` profile для mixed chat, Bash/code, документов и
изображений. Runtime работает offline, API слушает localhost и требует key.

## Профиль

| Параметр | Default | Причина |
| --- | ---: | --- |
| Checkpoint | `google/gemma-4-26B-A4B-it`, BF16 | Native A100 Tensor Core path и quality control |
| Context | 131072 | Нативный 128K checkpoint contract |
| Active sequences | 24 | Continuous batching для нескольких десятков пользователей |
| Batched tokens | 8192 | Chunked prefill без монополизации scheduler длинным документом |
| HBM fraction | 0.90 | Веса около 48 GiB плюс GPU KV и runtime buffers |
| Images/request | 4 | Несколько страниц документа или screenshots |
| Image budget | 1120 soft tokens | OCR и мелкий текст; снижайте до 280/560 для throughput |
| RAM KV | 128 GiB | Pinned prefix tier |
| Private `/dev/shm` | 144 GiB | RAM KV mmap плюс startup headroom |
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
cd vllm/gemma4-26b-a4b-tiered-kv
cp .env.example .env
mkdir -p /data/vllm-cache /data/vllm-kv-cache results
chmod 700 /data/vllm-cache /data/vllm-kv-cache results
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
set -a
source .env
set +a
python3 smoke.py --structured --tool
```

Smoke всегда отправляет сгенерированную PNG-картинку `red | blue` и принимает
успех только при корректном structured vision-ответе. `--image screenshot.png`
добавляет OCR/analysis реального изображения.

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
export VLLM_GEMMA4_ENV_FILE=/absolute/path/to/nyashkimyashki/vllm/gemma4-26b-a4b-tiered-kv/.env
./cc feature recipes-layout-tracing run nyashkimyashki-vllm-gemma4-a4b-control preflight

export VLLM_GEMMA4_CONFIRM=mutate-vllm-gemma4-a4b
./cc feature recipes-layout-tracing run nyashkimyashki-vllm-gemma4-a4b-control up -d
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
set -a
source .env
set +a
python3 smoke.py --structured --tool
```

Короткий прикладной A/B probe выполняется последовательно на одном GPU:

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

RAM tier vLLM 0.26 создаёт mmap `/dev/shm/vllm_offload_<engine>.mmap`. Рецепт
использует container-private IPC и `VLLM_SHM_GIB=144`; аварийное удаление
контейнера уничтожает private tmpfs. `serve.sh` удаляет scoped mmap перед
стартом и после завершения engine. Host `/dev/shm` не монтируется. Файлы
filesystem KV в `KV_CACHE_DIR` сохраняются между restart согласно cache policy.

`PYTHONHASHSEED=0` стабилизирует block keys после restart. Не задавайте
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: CUDA VMM может нарушить
pinned KV pages OffloadingConnector.

## Полная матрица `vllm bench serve`

При запущенном base service:

```bash
./run-bench-serve-matrix.sh .env base
```

Для MTP:

```bash
./run-bench-serve-matrix.sh .env mtp
```

Host wrapper вызывает `/opt/gemma4/bench-serve-matrix.sh` внутри работающего
server container. Default grid содержит 15 cells:

- concurrency: `2`, `4`, `8`, `16`, `32`;
- input: `8192`, `16384`, `32768` tokens;
- output: `1024` tokens с `ignore_eos`;
- prompts/cell: `concurrency × 2`, один warmup.

Каждый cell использует `/v1/completions`, infinite request rate и заданный
`max-concurrency`. Это измеряет saturated aggregate throughput при точной
synthetic token length. Результаты появляются в
`BENCH_RESULTS_DIR/<mode>-<UTC timestamp>/`: raw JSON/log для каждого cell,
`summary.json`, `summary.csv`. Любой request failure, отсутствующий cell или
ненулевой `vllm bench serve` завершает matrix с exit 1.

Один полный прогон измеряет `372 × 1024 = 380928` output tokens и добавляет
15360 warmup tokens. Он может идти долго. Менять defaults можно через
`BENCH_*` в `.env`; production comparison использует одинаковые значения для
base и MTP.

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

`MAX_MODEL_LEN=131072` объявляет доступный per-request context. Реальная
одновременная длина ограничена GPU KV pool; scheduler применяет preemption при
перегрузке. Для throughput tuning сохраняйте 131K contract и меняйте
`MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, admission limits gateway. Высокий
vision budget увеличивает prefill cost.

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
