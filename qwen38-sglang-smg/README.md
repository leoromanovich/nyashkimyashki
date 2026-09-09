# Qwen3.8-27B-FP8 · SGLang + Shepherd Model Gateway

Готовый профиль для одной GPU с 48 GiB VRAM: FP8 E4M3 KV, BF16 GDN,
MTP EAGLE 3/1/4 + ReplaySSM, HiCache RAM → SSD и cache-aware routing через SMG.

| Компонент | Версия / настройка |
| --- | --- |
| Модель | Qwen/Qwen3.8-27B-FP8, revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| SGLang | 0.5.19, базовый Docker image закреплён digest |
| Shepherd Model Gateway | самостоятельный SMG 1.10.1 |
| gRPC bridge / proto | 0.9.1 / 0.4.16, compatibility patch при сборке |
| Контекст / running limit | 32768 / 8 |
| GPU KV / GDN | измерено 247360 tokens / 35 slots |
| Host HiCache | 64 decimal GB; измерено 1336256 KV tokens |
| SSD cache | cap 48 GiB, watermark 0.9, оставить 20 GiB свободными |

## Трассировка запросов

[TRACING.md](TRACING.md) — включение OTLP, локальный Jaeger и подключение
OpenCode → OpenWebUI → LiteLLM → SMG → SGLang. Включён host script
`opencode_trace.py` для проверки CLI, инструментов и всей цепочки spans.
Tracing включается отдельными Compose
overlays; базовый запуск сохраняет только существующие `/metrics`.

## Требования

Linux x86_64, Docker Engine с Compose v2 и NVIDIA Container Toolkit.
Проверено на RTX 4090 с 48 GiB, driver 580.173.02, 128 GiB RAM и NVMe.
Обычная RTX 4090 24 GiB этот профиль не вмещает. Образ использует CUDA 13;
драйвер должен поддерживать его запуск.

Модель занимает около 29 GiB. Для model/cache каталогов предусмотрите минимум
120 GiB свободного SSD; Docker images и build cache требуют отдельного места.
64 GB host HiCache дополнительно к памяти загрузки модели и runtime требуют
достаточного запаса RAM. Веса монтируются read-only в обоих runtime сервисах.

## Запуск

Из корня этого репозитория:

```bash
cd qwen38-sglang-smg
cp .env.example .env
```

Проверьте пути и GPU_ID в `.env`. По умолчанию API доступен только на localhost.
Для bx и удалённых клиентов задайте `API_BIND=192.168.0.59`. Встроенной
аутентификации здесь нет; открывайте API только в доверенной сети либо за proxy.
Во время экспериментов задайте `RESTART_POLICY=no` в локальном `.env`.

```bash
docker compose config --quiet
docker compose build
docker compose run --rm --no-deps download
docker compose up -d --wait --wait-timeout 600
docker compose ps
```

Сервис `download` работает только по явному вызову, использует тот же образ,
загружает закреплённый checkpoint через `hf` и завершает работу. Host Python/hf
для загрузки не нужны. Повторный запуск переиспользует локальные файлы.
Для просмотра плана загрузки: `docker compose run --rm --no-deps download --dry-run`.
Runtime работает с `HF_HUB_OFFLINE=1`. Model/cache bind-каталоги создаёт Docker.

API: `http://127.0.0.1:30000/v1`, model ID `Qwen3.8-27B`.
Worker gRPC 19051, metrics 19052 и ZMQ 5557 доступны в Compose network;
SMG metrics: `http://127.0.0.1:29000/metrics` на Docker host.

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"Ответь: привет"}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

Замените адрес в curl, если изменили API_BIND/API_PORT.

## Проверки и управление

Следующие проверки входят в image. OpenCode probe запускается отдельно на host
по [инструкции tracing](TRACING.md#opencode-воспроизводимая-проверка):

```bash
docker compose exec -T sglang python3 /opt/qwen38/benchmark.py --mode smoke
docker compose exec -T sglang python3 /opt/qwen38/inspect_cache.py info
docker compose exec -T sglang python3 /opt/qwen38/inspect_cache.py loads
docker compose exec -T sglang python3 /opt/qwen38/benchmark.py --mode quality
docker compose exec -T sglang python3 /opt/qwen38/benchmark.py --mode precision
docker compose logs --tail 100 sglang smg
docker compose stop
```

Для изменения параметров и повторного запуска пересоздавайте оба сервиса:

```bash
docker compose up -d --force-recreate --wait --wait-timeout 600
```

Политика `unless-stopped` предназначена для штатного запуска. Экспериментальное
`restart=no` храните в ignored `.env`; перед push проверьте published defaults.
После изменения `.env` уже созданные контейнеры получают новую policy при
пересоздании. `down` сохраняет bind-каталоги с весами и SSD cache.

Control Center предоставляет те же операции через Nix app
`nyashkimyashki-qwen38-sglang-smg-control`: build, download, up, stop, info, loads,
metrics, events, watch-loads, smoke, quality, precision, benchmark, ssd-restore,
disconnect-check и publication-check. Для local development используется
feature input override; `QWEN38_CONFIG_DIR` выбирает каталог Compose.
Самостоятельный запуск папки обходится стандартным Docker Compose.

## SSD и routing

SMG опрашивает `GetLoads` раз в секунду и подписывается на `SubscribeKvEvents`.
Routing policy `cache_aware`, block/page size=64. KV/GDN tensors хранятся
в SGLang; gateway получает metadata. HiCache использует write_through,
page_first/kernel, file backend и wait_complete prefetch.

Изолированная проверка SSD на свободном от других запросов worker:

```bash
mkdir -p results
docker compose exec -T sglang python3 /opt/qwen38/benchmark.py \
  --mode ssd-restore --ssd-retrieval --output-tokens 32 --drop-file-cache \
  > results/ssd-restore.json
```

Проверяются успешный reset GPU/RAM cache, SSD prefetch, cached tokens,
физические чтения и контрольный ответ. После тяжёлой нагрузки FlushCache может
не дождаться idle: сохранённый JSON позволяет повторить restore после restart
через `--restore-record`; подробности доступны в `benchmark.py --help`.

При смене KV/GDN dtype либо ENABLE_MTP используйте отдельный SSD_CACHE_DIR:
cache hashes не кодируют формат tensor files. Значения SSD cap принимают
`48Gi` или число байт; суффикс `48GB` не поддерживается.

`max-mamba-cache-size=35` удерживает GDN pool: экономия BF16 отдаётся в KV.
При изменении размеров пула, host memory или concurrency перепроверьте runtime
allocations и admission. Параметры в `.env.example` относятся к проверенному
48 GiB профилю; масштабирование на другие карты требует новых замеров.

## Ограничения

- KV scales отсутствуют в checkpoint; SGLang использует `1.0`. Полноценная
  калибровка и quality benchmark не выполнялись. BF16 ReplaySSM может накапливать
  округления. Synthetic проверки прошли; полная эквивалентность точности неизвестна.
- На 8K input подтверждены 8 running; на 24–28K + 1024 output одновременно
  наблюдались 7 running. Все запросы завершились. Raw KV capacity не задаёт
  допустимую параллельность для любого контекста.
- Bridge передаёт события только DP rank 0, теряет storage medium и не имеет
  initial snapshot/cursor replay. TP1/DP1 подтверждает работу связки. Native
  multi-GPU DPA, выбор между несколькими workers и tier-aware routing не проверены.
  При DP1 флаг enable-dp-attention не создаёт несколько attention replicas.
- `max-tree-size` ограничивает fallback metadata tree; event index этим флагом
  не ограничивается. Совпадение KV events не гарантирует наличие GDN checkpoint.
- Vision остаётся на GPU; перенос encoder на CPU в этот recipe не входит.

Проверки, результаты и точные границы приведены в [VALIDATION.md](VALIDATION.md).
`patch_servicer.py` исправляет переименованное поле prefill queue для GetLoads(all)
и подключает request tracing frontend к scheduler. `patch_smg.py` включает
передачу W3C context в gRPC client. Оба patch проверяют версии/source anchors;
при обновлении engine/gateway/bridge их нужно пересмотреть.
`--tokenizer-path /model` также обязателен для этого bridge.
