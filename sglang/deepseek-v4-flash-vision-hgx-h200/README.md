# DeepSeek V4 Flash Vision — HGX 8×H200

## Запуск и трассировка

Состав: **sglang + SMG + Collector**.
8×H200, DPA8/TP8, SSD HiCache. Параметры модели и cache budgets сохранены при переносе.

```bash
cp .env.example .env
# Настройте пути, API keys и OTLP_UPSTREAM_ENDPOINT в .env.
# Для экспериментов добавьте RESTART_POLICY=no.
docker compose config --quiet
docker compose build sglang smg
docker compose up -d --build
```

Внешний клиент обращается к SMG на `http://<host>:30000/v1`; укажите одинаковый
inference API key в `.env` и LiteLLM. Публичный bind по умолчанию — localhost;
для доступа с другого сервера задайте `SMG_BIND=<LAN IP>`.
Collector отправляет только очищенные traces в обязательный внешний OTLP/gRPC endpoint; TLS включён по умолчанию. OWUI, LiteLLM и Jaeger этим Compose не создаются.

[Интеграция OWUI/LiteLLM, просмотр запроса и проверочный клиент](../../misc/telemetry/README.md).
[Запуск через Nix](../../misc/telemetry/README.md#nix).


DPA throughput-профиль для **200 пользователей × 2 сессии**, контекста **200000**,
**2 ТБ RAM / 17 ТБ SSD**. Source/Compose проверены 2026-09-09. GPU smoke,
восстановление Vision KV из RAM/SSD и нагрузочный прогон на HGX ещё требуются.

## Модель и версия

- `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, revision
  `6821d6ad3681a4b137b066b76094fa82ebd0a380`.
- Оригинальные FP4 experts + FP8 dense/attention; Hopper использует
  `flashinfer_mxfp4` → SM90 CUTLASS W4A16. `--quantization modelopt_fp4`
  относится к другому формату checkpoint.
- `lmsysorg/sglang:dev-dsv4-flash-vision`, закреплённый linux/amd64 digest
  `sha256:44a113290011bf87fcfeabc2ed94966bfc90f91b2dbe6239a69de3f35e686cba`.
- Image source `40b3e15ddbd9a1067e181283d9900dd3f4d76ed7`, CUDA 13.0.3,
  FlashInfer 0.6.18. Требуется совместимый NVIDIA R580+ driver, Docker Compose
  и NVIDIA Container Toolkit. `ipc: host` использует host `/dev/shm`.

Vision support находится в [PR #37253](https://github.com/sgl-project/sglang/pull/37253);
на 2026-09-07 latest release — v0.5.19, PR был открыт. Cookbook требует preview.
Его H200 Vision balanced cell использует **TP4** и помечен `verified: false`.
Основной профиль здесь — **TP8/DP8/DPA/EP1**, A2A `none`: восемь независимых
attention/KV ranks, tensor-sharded experts. Pinned source содержит этот путь
и синхронизацию Vision routing между ranks. Throughput и latency на HGX
требуют измерений; DPA не требует DeepEP.

## Профиль

| Параметр | Значение | Назначение |
| --- | --- | --- |
| Parallelism | TP8, DP8, DPA, EP1, A2A none | независимые attention/KV ranks, TP experts |
| Context | 200000 | общий бюджет prompt + reasoning + answer |
| Running | 256 CLI → 32/rank | до 256 active суммарно, с учётом KV capacity |
| Queued | 64/rank → до 512 суммарно | отдельная waiting queue каждого scheduler |
| DP balancing | total_tokens | учитывает длину текущих и поступающих запросов |
| Scheduler | FCFS | порядок поступления при общей нагрузке |
| Chunked prefill | 32768 CLI → 4096/rank | engine делит CLI budget на DP8 |
| Static memory | 0.85 | запас для vision activations и runtime |
| CUDA graphs decode | до batch 32/rank | соответствует per-rank running cap |
| KV | FP8, page 256 | DeepSeek V4 compressed attention |
| RAM cache | ratio 1.25, direct/page_first_direct | повторное использование холодных префиксов |
| SSD cache | file, 10T/index, 3T min free, LRU 0.9 | общая filesystem quota ≤12 TB обязательна |
| Reasoning | high | default при контексте 200k |

Контекст 200k включает выход. Практический клиентский бюджет:
prompt ≤184000, `max_tokens=16000`; image/template tokens входят в prompt.
Клиент должен сохранять историю и tool messages. KV cache может вытесняться.

400 сессий могут порождать запросы с паузами. `max-running-requests=256`
делится на DP8: 32/rank, фактический batch ограничен KV blocks. Значение
`max-queued-requests=64` применяется к каждому DP scheduler без деления:
до 512 waiting на узле. Внешний gateway задаёт общий admission limit, если
нужен жёсткий предел запросов на весь endpoint. Переполнение локальной
очереди даёт HTTP 503; используйте bounded retry с jitter.

DPA распределяет уникальные контексты между attention ranks. В прежнем
TP8/DP1 compressed KV реплицировался. Увеличение полезной KV capacity
ограничивают также веса, vision activations и runtime buffers; восьмикратный
выигрыш throughput обещать нельзя. 400 ×200k — 80 млн уникальных токенов:
HBM/RAM residency и SLA такого workload подтверждаются benchmark.

HiCache ratio 1.25 применяется к host pools каждого rank. Измерьте сумму
`Allocating ... host memory` всех восьми ranks: целевой host KV ≤1.2–1.3 ТБ,
MemAvailable ≥400 ГБ. При нехватке RAM уменьшайте ratio, затем mem fraction.
`--hicache-size` для V4 в этом image вызывает ValueError.
`direct` означает GPU↔RAM I/O; file backend использует buffered filesystem I/O.

Под `/hicache` нужен новый namespace с суффиксом `-dpa8`. В DPA каждый worker
получает storage `attn_tp_rank=0` и становится writer. Восемь процессов
используют общий каталог с независимыми LRU indexes: `10T` ограничивает
учёт одного evictor и не обеспечивает точный общий cap. Межпроцессной
координации eviction нет. Atomic replace защищает целостность отдельной
записи; cache misses при конкурирующем eviction допустимы.

**Общая filesystem/project quota ≤12 ТБ обязательна.** `3T` min-free смотрит
на весь filesystem, но конкурирующие проверки не являются общей резервацией.
Контролируйте bytes/inodes и оставляйте место для весов и Docker layers.
После смены checkpoint, engine или KV layout используйте новый namespace.
Docker restart policy сама по себе не перезапускает `unhealthy` контейнер.

Для agent sessions trusted gateway может назначать
`X-Data-Parallel-Rank: SHA256(tenant + session) % 8` и удалять такой header
из внешнего запроса. Это сохраняет GPU/RAM prefix locality. Без header
работает `total_tokens`. Affinity может перегрузить отдельный rank; измеряйте
очереди и TTFT по ranks. Общий SSD допускает reuse между ranks после backup.

## Отличия от GLM-5.2 NVFP4

- Собственные parsers: `deepseek-v4` для reasoning, `deepseekv4` для tools.
- Hopper FP4 профиль использует TP8/DP8/DPA с TP для экспертов. Text-only
  `sgl-project/DeepSeek-V4-Flash-FP8` теряет требуемый Vision checkpoint.
- Shared-expert fusion явно отключена; chunked prefill и prefix cache включены.
- DSPARK выключен для стартового target-only профиля. Для отдельного A/B
  использовать DSpark, проверить image batches, acceptance rate и p95 ITL.
  EAGLE flags из GLM не подходят к bundled DSpark head.
- FP4 indexer и prefill context parallelism исключены из этого профиля.

## Управление через Control Center

Nix app: `nyashkimyashki-sglang-dsv4-vision-h200-control`.
В локальной разработке source выбирается feature override:

```bash
./cc feature recipes-layout-tracing run nyashkimyashki-sglang-dsv4-vision-h200-control config
```

Скопировать `.env.example` во внешний env-файл, задать два разных ключа,
создать каталоги `MODEL_CACHE_DIR` и `HICACHE_DIR`, проверить quota и bind IP.
При переходе со старого TP-only env перенесите новые defaults: running 256,
chunk 32768, graph 32, `MAX_QUEUED_REQUESTS_PER_DP=64`, новый cache namespace.
Старое имя `MAX_QUEUED_REQUESTS` больше не используется.
По умолчанию endpoint доступен с localhost; `BIND_IP` должен быть адресом,
доступным существующему gateway. Auth и health используют порт 30000.

```bash
export DSV4_ENV_FILE=/absolute/path/dsv4-vision.env
./cc feature recipes-layout-tracing run nyashkimyashki-sglang-dsv4-vision-h200-control config

# Явное разрешение runtime side effects на целевом HGX.
export DSV4_CONFIRM=mutate-dsv4-vision-h200
./cc feature recipes-layout-tracing run nyashkimyashki-sglang-dsv4-vision-h200-control pull
./cc feature recipes-layout-tracing run nyashkimyashki-sglang-dsv4-vision-h200-control preflight
./cc feature recipes-layout-tracing run nyashkimyashki-sglang-dsv4-vision-h200-control up -d
./cc feature recipes-layout-tracing run nyashkimyashki-sglang-dsv4-vision-h200-control logs --tail 100 -f
./cc feature recipes-layout-tracing run nyashkimyashki-sglang-dsv4-vision-h200-control smoke
```

Первый запуск скачивает pinned checkpoint в `/models` при доступе к Hugging Face.
`HF_TOKEN` нужен только при требованиях доступа/rate limit. После загрузки
snapshot сохраняется в `MODEL_CACHE_DIR`. `config` и Nix check работают без GPU,
Docker daemon и весов. Production launch в рамках подготовки recipe не выполнялся.
`preflight` читает GPU/RAM/filesystem и проверяет уже скачанный image.
`up` и `restart` повторяют preflight.

Во время экспериментов отключайте restart через **локальный, игнорируемый Git**
Compose override с `restart: "no"`. Перед публикацией проверьте штатный
`unless-stopped`; основной `docker-compose.yaml` сохраняет его всегда.

## Приёмка на HGX

1. Smoke: text, SSE, tool round trip, red/blue/red images на ranks 0/1/7,
   concurrent text/image requests с idle peers на остальных ranks.
   Скрипт печатает только статусы; ответы и внутренние рассуждения не сохраняет.
2. Отдельно проверить reasoning content с `reasoning_effort=high`, JSON/tool
   streaming, длинный многошаговый coding transcript, изображения после
   длинного текста, повтор одного изображения и смену изображения при том же тексте.
3. Проверить 32k / 128k / ~184k prompt с запасом на 16k output; границу
   200k и поведение превышения лимита. Затем нагрузку 32/64/128/256 активных
   запросов, 400 сессий, cold и warm prefixes, реальную долю vision.
4. Подтвердить RAM loadback, затем SSD loadback после вытеснения и restart:
   сравнить результаты с cold path и метрики `sglang:prefetched_tokens_total`,
   `sglang:backuped_tokens_total`, `usage.prompt_tokens_details.cached_tokens`.
   Простое наличие файлов не доказывает корректность восстановления.
5. Измерить p50/p95/p99 TTFT и ITL, output tok/s на активный запрос, queue wait,
   503, GPU/host memory и retractions по каждому DP rank. В `/get_server_info`
   проверить `effective_max_running_requests_per_dp=32`, в startup log —
   local chunk 4096. Сравнить running 128/256/512 → 16/32/64 на rank, graph
   cap 16/32/64, chunk 16384/32768/65536 → 2048/4096/8192 на rank.
   Значения увеличивать по p95 TTFT/ITL и доступному KV.

## Границы поддержки

Основной Compose и validator требуют **TP8/DP8/DPA/EP1/A2A none**.
`flashinfer_mxfp4` на SM90 зарегистрирован для dispatch `none`; добавление
DeepEP к этому runner не обосновано его наличием в других V4 профилях.
Pinned model source делает gather перед TP-MoE, combine/scatter после,
обрабатывает global image routing IDs, text-only и idle batches.

Полное сочетание H200 + Vision + DPA + HiCache ещё требует GPU-прогона.
TP-only можно держать локальным профилем сравнения. Основной публикуемый
recipe соответствует многопользовательской throughput задаче.

## Evidence

- [Official model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/tree/6821d6ad3681a4b137b066b76094fa82ebd0a380)
- [Current cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
- [H200 Vision matrix](https://github.com/sgl-project/sglang/blob/755f97c6223d4e70b11cc3aac934bde6c5fa2438/docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx)
- [DPA without EP](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide)
- [Pinned DPA + TP-MoE and Vision routing](https://github.com/sgl-project/sglang/blob/40b3e15ddbd9a1067e181283d9900dd3f4d76ed7/python/sglang/srt/models/deepseek_v4.py#L2234)
- [Pinned SM90 MXFP4 dispatch](https://github.com/sgl-project/sglang/blob/40b3e15ddbd9a1067e181283d9900dd3f4d76ed7/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py#L301)
- [Pinned Hopper quantization code](https://github.com/sgl-project/sglang/blob/40b3e15ddbd9a1067e181283d9900dd3f4d76ed7/python/sglang/srt/layers/quantization/mxfp4.py)
- [Pinned V4 host-pool assembly](https://github.com/sgl-project/sglang/blob/40b3e15ddbd9a1067e181283d9900dd3f4d76ed7/python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py)
- [Pinned V4 L2/L3 tests](https://github.com/sgl-project/sglang/blob/40b3e15ddbd9a1067e181283d9900dd3f4d76ed7/test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4.py)
- [Image metadata](https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/dev-dsv4-flash-vision)

- [Pinned DPA prefill normalization](https://github.com/sgl-project/sglang/blob/40b3e15ddbd9a1067e181283d9900dd3f4d76ed7/python/sglang/srt/arg_groups/parallel_hook.py#L190)
- [Pinned per-worker running limit](https://github.com/sgl-project/sglang/blob/40b3e15ddbd9a1067e181283d9900dd3f4d76ed7/python/sglang/srt/mem_cache/kv_cache_configurator.py#L1949)
