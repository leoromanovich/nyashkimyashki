# DeepSeek V4 Flash Vision · vLLM + LMCache · HGX 8×H200

Throughput-профиль для кодинговых ассистентов и чата: **8×H200, 400000 total
tokens, 2 ТБ RAM, 17 ТБ SSD**. Состав: vLLM + общий LMCache + SMG + Collector.
Лимиты рассчитаны на несколько сотен сессий с очередью и паузами инструментов.
Фактическая активная ёмкость и SLA требуют нагрузочного прогона на HGX.

## API и трассировка

Внешний клиент обращается к SMG на `http://<host>:30000/v1`; укажите одинаковый
inference API key в `.env` и LiteLLM. Публичный bind по умолчанию — localhost;
для доступа с другого сервера задайте `SMG_BIND=<LAN IP>`.
Collector отправляет только очищенные traces в обязательный внешний OTLP/gRPC endpoint; TLS включён по умолчанию. OWUI, LiteLLM и Jaeger этим Compose не создаются.

[Интеграция OWUI/LiteLLM, просмотр запроса и проверочный клиент](../../misc/telemetry/README.md).
[Запуск через Nix](../../misc/telemetry/README.md#nix).


**Кандидат для проверки на HGX**: Compose, argv и probes проверяются локально;
сборка CUDA-образа, Vision + LMCache restore и производительность требуют GPU.

## Параметры

| Параметр | Значение |
| --- | --- |
| Модель | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, revision в `versions.json` |
| Parallelism | TP1 / DP8 / EP8, восемь attention/KV ranks |
| Hopper MoE | `marlin`, исходные MXFP4 experts |
| All-to-all | `allgather_reducescatter`; DeepEP high throughput — отдельный A/B |
| KV | `fp8_ds_mla`, block 256, native hybrid KV manager |
| Контекст | 400000: template + input + image tokens + reasoning + answer |
| Клиентский budget | до 384000 входных токенов + 16000 на reasoning/answer |
| Scheduler | 32 seqs/rank, 8192 batched tokens/rank, chunked prefill |
| Admission | 512 незавершённых запросов всего при одном API process |
| HBM | `gpu-memory-utilization=0.88` |
| LMCache | один общий MP server, 1024 GiB RAM + 8192 GiB SSD |
| Prefix locality | GPU prefix cache + общий RAM/SSD cache между DP ranks |
| API | SMG `127.0.0.1:30000/v1`; engine `127.0.0.1:30001/v1` |
| Images | до 8 на запрос; video отключено |
| Speculation | выключена до проверки cache restore и throughput |

32 × 8 = потолок 256 активных последовательностей. Допуск зависит от
доступных KV blocks; 400 сессий включают паузы, tool execution и очередь.
`--max-num-queued-reqs=512` считает **running + waiting**, превышение даёт
HTTP 503. Дополнительные API processes умножили бы этот предел.

TP1/DP8/EP8 следует стратегии `single_node_dep` из vLLM recipes: attention
и dense layers реплицируются, эксперты распределяются по восьми GPU.
Это throughput-профиль, сопоставимый с DPA-профилями SGLang.
H200 использует Marlin для MXFP4; Blackwell-specific MegaMoE/NVFP4 overrides
из других рецептов сюда переносить нельзя.

## Версии и сборка

База — официальный `vllm/vllm-openai:deepseekv4-flash-vision`, закреплённый
по **amd64 digest**. Источник Docker image не указан в upstream labels:
`base_image_source_revision=null` сохранён намеренно. Vision PR #54566
уже merged; описание открытого PR в recipes отстаёт от GitHub.
Официальная Vision-квалификация относится к GB200, TP4 и короткому контексту.
H200 DEP8 с LMCache остаётся отдельной комбинацией.

LMCache text-рецепт предупреждает о FP4 dispatch и CUDA graph startup
ошибках в проверенном vLLM dev snapshot. Здесь закреплён отдельный Vision
image и явно выбран Marlin. Это ограничивает дрейф версий; успешность
model load и CUDA graphs подтверждается на HGX до допуска нагрузки.

`Dockerfile` собирает LMCache из commit `21a5db10…`, проверяет SHA-256
архива и компилирует CUDA extensions под torch из serving image.
Constraint-файл запрещает замену torch/vLLM/transformers/FlashInfer/CUDA
packages. Совпадение основных версий проверяется после установки.
Дополнительные Python build dependencies разрешаются при сборке;
полный результат следует закрепить собственным image digest после canary.
`/opt/recipe/build-packages.json` сохраняет фактические основные версии.

Нужен Linux x86_64 builder с CUDA 13-compatible NVIDIA driver R580+ на
serving host. В base image есть nvcc; wheel LMCache с чужой torch ABI
не используется. Сборка на macOS и загрузка модели локально не проверялись.

## Почему внешний LMCache MP

V4 хранит несколько compressed/SWA/state KV groups. Используется внешний
`lmcache.integration.vllm.lmcache_mp_connector.LMCacheMPConnector` с
`SupportsHMA`, сохранением native NHD layout и image-aware cache keys.
`--no-disable-hybrid-kv-cache-manager` явно включает native hybrid manager;
совместимость флага и connector проверяется по фактическому образу.
`--separate-object-groups` разделяет группы при хранении и восстановлении.
Обычный offload shorthand не выражает этот контракт.

Один MP server обслуживает восемь DP ranks через CUDA IPC. RAM budget
1024 GiB общий для узла. Оба контейнера видят все GPU и используют
host IPC/network; LMCache RPC 5555 и HTTP 8081 слушают только loopback.
LMCache HTTP health: `/healthcheck`, metrics: `/metrics`.

Путь данных: GPU KV → LMCache L1 RAM → LMCache L2 SSD; lookup/prefetch
выполняет обратное восстановление. L2 adapter `nixl_store_dynamic` с
backend `POSIX` принадлежит LMCache. GDS и отдельный vLLM NIXLConnector
этому профилю не требуются.

### Как ограничен диск

- L1: LRU, trigger 0.85, eviction ratio 0.1.
- L2: 8192 GiB учтённых объектов, LRU с trigger 0.85 и eviction ratio 0.1;
  удаление объектов освобождает файлы. Запись сверх учтённого cap отклоняется.
- После restart индекс строится лениво при lookup. Старые неиндексированные
  объекты и временные файлы после аварии могут выйти за внутренний учёт.
- **Отдельный filesystem/project quota 10 TiB для `LMCACHE_DISK_DIR`
  обязателен как жёсткая граница**, плюс контроль свободных inode.
  Preflight проверяет минимум 3 ТБ свободного места; quota настраивается
  администратором для выбранной файловой системы.

Namespace каталога включает model/image/LMCache revisions. При смене
одного из них создавайте новый каталог. Удалять старый namespace допустимо
после остановки соответствующей пары сервисов и проверки пути. Кэш содержит
данные запросов; доступ к его каталогу ограничьте сервисным пользователем.
Секреты храните во внешнем env-файле с правами 0600.

## Запуск через Control Center

Из корня `llm_engines`; feature input автоматически указывает на worktree:

```bash
export VLLM_DSV4_ENV_FILE=/etc/llm/deepseek-v4-vision-vllm.env
./cc feature vllm-dsv4-vision-h200-400k run nyashkimyashki-vllm-dsv4-lmcache-h200-control config
./cc feature vllm-dsv4-vision-h200-400k run nyashkimyashki-vllm-dsv4-lmcache-h200-control build-image
./cc feature vllm-dsv4-vision-h200-400k run nyashkimyashki-vllm-dsv4-lmcache-h200-control preflight
```

Скопируйте `.env.example` во внешний env-файл, задайте API key, cache paths
и создайте каталоги. Включите Nix features `nix-command flakes`, если
они отключены. Для экспериментов добавьте `RESTART_POLICY=no` во внешний
env-файл; штатная политика Compose — `unless-stopped`.
Сборка не запускает serving. Preflight проверяет восемь
H200, driver, RAM/SSD, установленные extensions, Vision/HMA/MM imports
и **каждый CLI-флаг по фактическому образу**.

После одобрения runtime-действия на выделенном HGX:

```bash
export VLLM_DSV4_CONFIRM=mutate-vllm-dsv4-h200
./cc feature vllm-dsv4-vision-h200-400k run nyashkimyashki-vllm-dsv4-lmcache-h200-control up
./cc feature vllm-dsv4-vision-h200-400k run nyashkimyashki-vllm-dsv4-lmcache-h200-control smoke
```

`up` использует уже собранный image. `config` выводит результат проверки
без секретов. Доступны `logs`, `ps`, `stop`, `down`, `restart`.
При restart LMCache перезапускайте также vLLM, чтобы обновить CUDA IPC
handles. `down` сохраняет model/cache volumes. Порт 30001 отделён от
SGLang engine 30000. SMG использует 30000; при смене стека освободите этот
порт. Оба восьми-GPU сервиса одновременно на HGX запускать нельзя.

## Проверка перед нагрузкой

1. Проверьте `nvidia-smi topo -m`, model load, Marlin/DEP8, hybrid KV groups
   и регистрацию всех восьми workers в LMCache. Запишите фактический
   GPU KV budget, число блоков и model/image versions.
2. `smoke` проверяет text, SSE, два последовательных tool round trips
   с `tool_choice=auto` в JSON и SSE, затем red/blue/red images на ranks
   0/1/7. Повторный вызов содержит историю предыдущих tools/results.
3. Проверьте эквивалентность результатов при первом запросе, повторе и переносе
   на другой rank. Подтвердите store/load/hit по LMCache metrics/logs.
   Корректный повторный ответ сам по себе не доказывает offload.
4. Для L1 restore прогрейте synthetic prefix, дождитесь завершения store,
   перезапустите только vLLM и повторите запрос. Для SSD restore дождитесь
   завершения L2 store, остановите оба сервиса, запустите их с тем же
   namespace и повторите запрос. Подтвердите чтение L2 после очистки RAM.
5. Повторите text/tools/images на 32k, 128k и около 400k total tokens.
   Input budget для клиента: до 384000 с output reserve 16000, с учётом
   template/image tokens. Long-context correctness проверяется отдельно.
6. Сравните cold, GPU-hot, RAM-hit и SSD-hit под 32/64/128/256 активными
   запросами, затем 400 сессиями с реальными паузами инструментов.
   Измеряйте TTFT p50/p95/p99, TPOT p95, output tok/s, queue, 503,
   preemptions, cache hit/load time и disk occupancy/inodes.

Для первого canary: `MAX_NUM_SEQS=8`, `CUDA_GRAPH_MAX_BS=8`, batch 4096.
Далее seqs 16 → 32 → 64 и batch 4096 → 8192 → 16384 по TTFT/TPOT
и preemptions. Рост seqs без доступного KV может ухудшить latency.
400k задаёт максимальную длину одного запроса; 400 полностью заполненных
контекстов одновременно нельзя обещать без измерений.

`ALL2ALL_BACKEND=deepep_high_throughput` — A/B после базовой correctness,
при наличии совместимого DeepEP в образе. DSpark/MTP добавляется после
проверки LMCache со speculation отдельным экспериментом.

Для GPU prefix locality trusted gateway может вычислять
`SHA256(tenant + conversation/repository identity) % 8` и передавать
`X-data-parallel-rank`. Удаляйте одноимённый внешний header. Без стабильной
identity internal queue-aware balancer выбирает rank сам; общий LMCache
сохраняет возможность reuse между ranks. Следите за hot-rank imbalance.

### Синтетическая проверка длинного контекста и offload

На инстансе без другой нагрузки:

```bash
./cc feature vllm-dsv4-vision-h200-400k run nyashkimyashki-vllm-dsv4-lmcache-h200-control acceptance --run-id canary-20260914-01 --prompt-tokens 384000 --stage warm
```

`/tokenize` считает полный chat template. Скрипт подбирает вход в пределах
128 токенов от заданного budget, резервирует 16000 output tokens и проверяет
маркеры в начале, середине и конце. Затем повторяет тот же запрос на ranks
0/0/1/7. Новый `run-id` создаёт новый prefix; сохраняйте его при restore.
Для ступеней 32k/128k total задайте `--prompt-tokens 16000` / `112000`.

Stdout содержит только JSON с correctness, prompt/completion counts, client
TTFT, elapsed и дельтами агрегированных LMCache counters. Его можно сохранить
в файл метрик. Prompt, ответы, tool payloads и metric labels не сохраняются.
Скрипт обращается прямо к engine; для отдельного smoke через SMG задайте
`VLLM_DSV4_BASE_URL=http://127.0.0.1:30000/v1`.

После `warm` дождитесь завершения L1/L2 stores по `/metrics` и нулевого
`lmcache_mp_num_inflight_l2_stores`. Для проверки RAM перезапустите только
vLLM, дождитесь его health и выполните:

```bash
./cc feature vllm-dsv4-vision-h200-400k run nyashkimyashki-vllm-dsv4-lmcache-h200-control acceptance --run-id canary-20260914-01 --prompt-tokens 384000 --stage ram-restore
```

Для SSD остановите пару vLLM/LMCache, запустите её с тем же SSD namespace
и выполните `acceptance` с теми же `run-id`/budget и `--stage ssd-restore`.
Runtime-действия выполняются оператором через `stop` / `up` / `restart`;
probe сервисы не перезапускает. Например, `restart vllm` очищает GPU cache,
`stop vllm lmcache` останавливает пару; после `up` дождитесь `/health`.

RAM-прогон требует прироста `lmcache_mp_num_chunks_loaded_total`;
SSD-прогон дополнительно требует `lmcache_mp_l2_load_completed_requests_total`.
Счётчики completed GPU transfers сами по себе не доказывают успешную загрузку.
Иная нагрузка мешает атрибуции: эти проверки требуют выделенного инстанса.
Качество tools/images на длинном контексте и конкурентный throughput
проверяются отдельно; три маркера дают ограниченную проверку text correctness.

## Источники

- [Официальный Vision recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp).
- [DEP strategy](https://github.com/vllm-project/recipes/blob/abacf57e353807216b2a7e0a740db5292af041a7/strategies/single_node_dep.yaml).
- [Vision PR #54566](https://github.com/vllm-project/vllm/pull/54566).
- [LMCache DeepSeek V4 Flash recipe](https://docs.lmcache.ai/recipes/deepseek_v4_flash.html): upstream text-квалификация.
- [Hybrid KV groups](https://github.com/LMCache/LMCache/blob/21a5db10f1fc280fc7a5d59ce2660697669a0e1d/docs/design/integration/vllm/hybrid-kv-cache-groups.md).
- [MP connector](https://github.com/LMCache/LMCache/blob/21a5db10f1fc280fc7a5d59ce2660697669a0e1d/lmcache/integration/vllm/lmcache_mp_connector.py).
- [Dynamic NIXL L2](https://github.com/LMCache/LMCache/blob/21a5db10f1fc280fc7a5d59ce2660697669a0e1d/lmcache/v1/distributed/l2_adapters/nixl_store_dynamic_l2_adapter.py).

400k profile/source review: 2026-09-14; исходные pins от 2026-09-08 сохранены
в `versions.json`. Текущая GPU-квалификация не заявляется.
