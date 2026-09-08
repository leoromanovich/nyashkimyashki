# DeepSeek V4 Flash Vision · vLLM + LMCache · HGX 8×H200

Профиль для агентного кодинга: 200 пользователей × 2 сессии, контекст
200000 токенов, 2 ТБ RAM, 17 ТБ SSD. **Кандидат для проверки на HGX**:
Compose и source contracts проверяются локально; сборка CUDA-образа,
Vision + LMCache restore и производительность требуют GPU-прогона.

## Параметры

| Параметр | Значение |
| --- | --- |
| Модель | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, revision в `versions.json` |
| Parallelism | TP1 / DP8 / EP8, восемь attention/KV ranks |
| Hopper MoE | `marlin`, исходные MXFP4 experts |
| All-to-all | `allgather_reducescatter`; DeepEP high throughput — отдельный A/B |
| KV | `fp8_ds_mla`, block 256, native hybrid KV manager |
| Контекст | 200000: input + image tokens + output |
| Scheduler | 32 seqs/rank, 8192 batched tokens/rank, chunked prefill |
| Admission | 512 незавершённых запросов всего при одном API process |
| HBM | `gpu-memory-utilization=0.88` |
| LMCache | один общий MP server, 1024 GiB RAM + 8192 GiB SSD |
| Prefix locality | GPU prefix cache + общий RAM/SSD cache между DP ranks |
| API | `127.0.0.1:30001/v1`, ключ через внешний env-файл |
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
./cc feature vllm-dsv4-vision-lmcache-h200 run nyashkimyashki-vllm-dsv4-lmcache-h200-control config
./cc feature vllm-dsv4-vision-lmcache-h200 run nyashkimyashki-vllm-dsv4-lmcache-h200-control build-image
./cc feature vllm-dsv4-vision-lmcache-h200 run nyashkimyashki-vllm-dsv4-lmcache-h200-control preflight
```

Скопируйте `.env.example` во внешний env-файл, задайте API key, cache paths
и создайте каталоги. Включите Nix features `nix-command flakes`, если
они отключены. Сборка не запускает serving. Preflight проверяет восемь
H200, driver, RAM/SSD, установленные extensions, Vision/HMA/MM imports
и **каждый CLI-флаг по фактическому образу**.

После одобрения runtime-действия на выделенном HGX:

```bash
export VLLM_DSV4_CONFIRM=mutate-vllm-dsv4-h200
./cc feature vllm-dsv4-vision-lmcache-h200 run nyashkimyashki-vllm-dsv4-lmcache-h200-control up
./cc feature vllm-dsv4-vision-lmcache-h200 run nyashkimyashki-vllm-dsv4-lmcache-h200-control smoke
```

`up` использует уже собранный image. `config` выводит результат проверки
без секретов. Доступны `logs`, `ps`, `stop`, `down`, `restart`.
При restart LMCache перезапускайте также vLLM, чтобы обновить CUDA IPC
handles. `down` сохраняет model/cache volumes. Порт 30001 отделён от
SGLang 30000; оба восьми-GPU сервиса одновременно на HGX запускать нельзя.

## Проверка перед нагрузкой

1. Проверьте `nvidia-smi topo -m`, model load, Marlin/DEP8, hybrid KV groups
   и регистрацию всех восьми workers в LMCache. Запишите фактический
   GPU KV budget, число блоков и model/image versions.
2. `smoke` проверяет text, SSE, tool round trip и red/blue/red images
   на ranks 0/1/7. Длинный синтетический prefix пересекает LMCache chunks.
3. Проверьте эквивалентность результатов при первом запросе, повторе и переносе
   на другой rank. Подтвердите store/load/hit по LMCache metrics/logs.
   Корректный повторный ответ сам по себе не доказывает offload.
4. Для L1 restore прогрейте synthetic prefix, дождитесь завершения store,
   перезапустите только vLLM и повторите запрос. Для SSD restore дождитесь
   завершения L2 store, остановите оба сервиса, запустите их с тем же
   namespace и повторите запрос. Подтвердите чтение L2 после очистки RAM.
5. Повторите text/tools/images на 32k, 128k и около 200k total tokens.
   Input budget для клиента: до 184000 с output reserve 16000, с учётом
   template/image tokens. Long-context correctness проверяется отдельно.
6. Сравните cold, GPU-hot, RAM-hit и SSD-hit под 32/64/128/256 активными
   запросами, затем 400 сессиями с реальными паузами инструментов.
   Измеряйте TTFT p50/p95/p99, TPOT p95, output tok/s, queue, 503,
   preemptions, cache hit/load time и disk occupancy/inodes.

Для первого canary: `MAX_NUM_SEQS=8`, `CUDA_GRAPH_MAX_BS=8`, batch 4096.
Далее seqs 16 → 32 → 64 и batch 4096 → 8192 → 16384 по TTFT/TPOT
и preemptions. Рост seqs без доступного KV может ухудшить latency.
200k задаёт максимальную длину одного запроса; 400 полностью заполненных
контекстов одновременно нельзя обещать без измерений.

`ALL2ALL_BACKEND=deepep_high_throughput` — A/B после базовой correctness,
при наличии совместимого DeepEP в образе. DSpark/MTP добавляется после
проверки LMCache со speculation отдельным экспериментом.

Для GPU prefix locality trusted gateway может вычислять
`SHA256(tenant + conversation/repository identity) % 8` и передавать
`X-data-parallel-rank`. Удаляйте одноимённый внешний header. Без стабильной
identity internal queue-aware balancer выбирает rank сам; общий LMCache
сохраняет возможность reuse между ranks. Следите за hot-rank imbalance.

## Источники

- [Официальный Vision recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp).
- [DEP strategy](https://github.com/vllm-project/recipes/blob/abacf57e353807216b2a7e0a740db5292af041a7/strategies/single_node_dep.yaml).
- [Vision PR #54566](https://github.com/vllm-project/vllm/pull/54566).
- [LMCache DeepSeek V4 Flash recipe](https://docs.lmcache.ai/recipes/deepseek_v4_flash.html): upstream text-квалификация.
- [Hybrid KV groups](https://github.com/LMCache/LMCache/blob/21a5db10f1fc280fc7a5d59ce2660697669a0e1d/docs/design/integration/vllm/hybrid-kv-cache-groups.md).
- [MP connector](https://github.com/LMCache/LMCache/blob/21a5db10f1fc280fc7a5d59ce2660697669a0e1d/lmcache/integration/vllm/lmcache_mp_connector.py).
- [Dynamic NIXL L2](https://github.com/LMCache/LMCache/blob/21a5db10f1fc280fc7a5d59ce2660697669a0e1d/lmcache/v1/distributed/l2_adapters/nixl_store_dynamic_l2_adapter.py).

Source review: 2026-09-08; полные revisions и base digest в `versions.json`.
