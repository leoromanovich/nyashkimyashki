# DeepSeek V4 Flash Vision — HGX 8×H200

Стартовый профиль для **200 пользователей × 2 сессии**, контекста **200000**,
**2 ТБ RAM / 17 ТБ SSD**. Статически проверен 2026-09-07. GPU smoke,
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
на дату проверки latest release — v0.5.19, PR открыт. Cookbook требует preview.
Его H200 Vision balanced cell использует **TP4** и помечен `verified: false`.
Здесь TP увеличен до **8** под имеющийся HGX; остальные serving/cache лимиты —
обоснованный старт для измерений. Throughput и latency этого сочетания неизвестны.

## Профиль

| Параметр | Значение | Назначение |
| --- | --- | --- |
| Parallelism | TP8, DP1, EP1 | единый endpoint на всех GPU |
| Context | 200000 | общий бюджет prompt + reasoning + answer |
| Running / queued | 128 / 512 | активная генерация и ограниченная очередь |
| Scheduler | FCFS | порядок поступления при общей нагрузке |
| Chunked prefill | 8192 | ограничение длительности prefill шага |
| Static memory | 0.85 | запас для vision activations и runtime |
| CUDA graphs decode | до batch 128 | соответствует admission cap |
| KV | FP8, page 256 | DeepSeek V4 compressed attention |
| RAM cache | ratio 1.25, direct/page_first_direct | повторное использование холодных префиксов |
| SSD cache | file, 10T cap, 3T min free, LRU 0.9 | холодные истории между запросами |
| Reasoning | high | default при контексте 200k |

Контекст 200k включает выход. Практический клиентский бюджет:
prompt ≤184000, `max_tokens=16000`; image/template tokens входят в prompt.
Клиент должен сохранять историю и tool messages. KV cache может вытесняться.

400 сессий могут порождать запросы с паузами. `max-running-requests=128` —
верхняя граница scheduler; фактический batch ограничен доступными KV blocks.
400 одновременных запросов заполнят running batch и очередь. При переполнении
очереди SGLang возвращает HTTP 503; клиенту нужен bounded retry с jitter.
400 уникальных историй по 200k — **80 млн токенов**. Их постоянное нахождение
в HBM/RAM этим профилем не гарантируется. Общие префиксы уменьшают объём.

В TP-only V4 MLA/KV пулы реплицируются по ranks. Нельзя складывать восемь
одинаковых `max_total_num_tokens` как независимую ёмкость. Стартовый ratio 1.25
выбран с запасом для 2 ТБ RAM: измерить суммарные `Allocating ... host memory`
всех восьми ranks; целевой суммарный host KV ≤1.2–1.3 ТБ и MemAvailable ≥400 ГБ.
При нехватке RAM уменьшать ratio, затем mem fraction. `--hicache-size`
в этом image для V4 вызывает ValueError. `direct` здесь означает GPU↔RAM I/O;
file backend использует обычный filesystem I/O.

Под `/hicache` нужен отдельный каталог этой модели и версии; filesystem quota
около 12 ТБ задаёт жёсткий предел. 10T — встроенный LRU cap; 3T — резерв
свободного места filesystem. Контролировать bytes и inodes. Остаток SSD нужен
для весов, Docker layers и служебных данных. При смене checkpoint, engine или
KV layout использовать новый cache namespace. `unhealthy` требует реакции
мониторинга: Docker restart policy сама по себе не перезапускает unhealthy.

## Отличия от GLM-5.2 NVFP4

- Собственные parsers: `deepseek-v4` для reasoning, `deepseekv4` для tools.
- Стартовый Hopper FP4 профиль использует TP8/DP1. DPA поддерживается отдельным
  путём с TP для экспертов; ниже описан кандидат для A/B. Text-only
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
./cc feature deepseek-v4-flash-vision-h200 run nyashkimyashki-sglang-dsv4-vision-h200-control config
```

Скопировать `.env.example` во внешний env-файл, задать два разных ключа,
создать каталоги `MODEL_CACHE_DIR` и `HICACHE_DIR`, проверить quota и bind IP.
По умолчанию endpoint доступен с localhost; `BIND_IP` должен быть адресом,
доступным существующему gateway. Auth и health используют порт 30000.

```bash
export DSV4_ENV_FILE=/absolute/path/dsv4-vision.env
./cc feature deepseek-v4-flash-vision-h200 run nyashkimyashki-sglang-dsv4-vision-h200-control config

# Явное разрешение runtime side effects на целевом HGX.
export DSV4_CONFIRM=mutate-dsv4-vision-h200
./cc feature deepseek-v4-flash-vision-h200 run nyashkimyashki-sglang-dsv4-vision-h200-control pull
./cc feature deepseek-v4-flash-vision-h200 run nyashkimyashki-sglang-dsv4-vision-h200-control preflight
./cc feature deepseek-v4-flash-vision-h200 run nyashkimyashki-sglang-dsv4-vision-h200-control up -d
./cc feature deepseek-v4-flash-vision-h200 run nyashkimyashki-sglang-dsv4-vision-h200-control logs --tail 100 -f
./cc feature deepseek-v4-flash-vision-h200 run nyashkimyashki-sglang-dsv4-vision-h200-control smoke
```

Первый запуск скачивает pinned checkpoint в `/models` при доступе к Hugging Face.
`HF_TOKEN` нужен только при требованиях доступа/rate limit. После загрузки
snapshot сохраняется в `MODEL_CACHE_DIR`. `config` и Nix check работают без GPU,
Docker daemon и весов. Production launch в рамках подготовки recipe не выполнялся.
`preflight` читает GPU/RAM/filesystem и проверяет уже скачанный image.
`up` и `restart` повторяют preflight.

## Приёмка на HGX

1. Smoke: text, SSE, tool call → tool result → answer, red/blue/red images.
   Скрипт печатает только статусы; ответы и внутренние рассуждения не сохраняет.
2. Отдельно проверить reasoning content с `reasoning_effort=high`, JSON/tool
   streaming, длинный многошаговый coding transcript, изображения после
   длинного текста, повтор одного изображения и смену изображения при том же тексте.
3. Проверить 32k / 128k / ~184k prompt с запасом на 16k output; границу
   200k и поведение превышения лимита. Затем нагрузку 32/64/128 активных
   запросов, 400 сессий, cold и warm prefixes, реальную долю vision.
4. Подтвердить RAM loadback, затем SSD loadback после вытеснения и restart:
   сравнить результаты с cold path и метрики `sglang:prefetched_tokens_total`,
   `sglang:backuped_tokens_total`, `usage.prompt_tokens_details.cached_tokens`.
   Простое наличие файлов не доказывает корректность восстановления.
5. Измерить p50/p95/p99 TTFT и ITL, output tok/s на активный запрос, queue wait,
   503, GPU/host memory и retractions. Только после этого повышать running и
   graph cap совместно до 192/256, mem fraction до 0.88–0.90, chunk до 16384.

## DPA для отдельного A/B

Первый кандидат для нагрузки 400 сессий — **TP8/DP8/DPA, EP1, A2A none**
с текущим `flashinfer_mxfp4`. В pinned image существует отдельный путь
DPA + TP-MoE: gather перед экспертами, combine/scatter после, обработка image
routing IDs соседних ranks. Каждый attention rank хранит KV своих запросов;
эксперты остаются tensor-sharded. DPA не требует DeepEP. У SM90 MXFP4 runner
в этом image зарегистрирован dispatch `none`; DeepEP для него не подключён.

Комбинация **H200 + Vision + DPA + HiCache** требует GPU-прогона. Compose и
`validate.py` фиксируют исходный TP8/DP1 профиль; переход на DPA требует
согласованного изменения конфигурации и её контракта. Перед A/B пересчитать
per-rank prefill/concurrency, host cache budgets и лимиты SSD для всех writers;
проверить text/image/idle mix, L2/L3 loadback и sticky routing по сессии.
Пометка Hopper FP4 «TP-only» в cookbook не доказывает запрет DPA.

Дополнительный A/B — **2 независимые TP4 реплики** со sticky routing и
раздельными cache namespaces. Эффект на ёмкость KV, p95 и throughput измерить.
Для hard SLA на 400 одновременно генерирующих длинных запросов нужен benchmark.

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
