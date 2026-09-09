# Request tracing: OpenCode → OpenWebUI → LiteLLM → SMG → SGLang

W3C `traceparent` / `tracestate` связывают spans одного запроса. Каждый сервис
отправляет свои spans через OTLP в общий backend. Один общий адрес Collector
без передачи HTTP/gRPC context сам по себе не объединяет запросы.

```text
OpenWebUI HTTP span
└── HTTP client → LiteLLM
    └── LiteLLM server span
        ├── LiteLLM LLM-call span
        └── SMG http_request
            └── SMG grpc_execute (dispatch)
                └── sglang.generate (полный lifecycle inference)
                    ├── gRPC bridge
                    └── Scheduler [TP 0] [PP 0]
                        ├── prefill waiting
                        ├── prefill forward
                        └── decode forward
```

Родительские связи проверены через все четыре сервиса. Точные названия native spans зависят
от версии; наличие стадий зависит от режима и trace level. `grpc_execute`
измеряет отправку gRPC и получение stream handle. Для длительности генерации
смотрите `sglang.generate`, scheduler stages и полный HTTP span. Поле SMG
`latency` относится к получению HTTP response headers.

## Что входит

- SMG 1.10.1: native W3C HTTP extraction и OTLP/gRPC exporter. `patch_smg.py`
  подключает существующий `OtelTraceInjector` к SGLang client factory: релиз
  оставляет эту factory на `NoopTraceInjector`. Поэтому image собирается из
  pinned commit `2f945936b5be75d1a8bbb8a7dbb2f6585ac76e8e`, через Rust 1.98.0,
  Cargo.lock и профиль CI (`release`, opt-level 2, thin LTO). Первая сборка требует скачивания Rust dependencies;
  Docker BuildKit сохраняет Cargo cache для повторных сборок.
- SGLang 0.5.19 + bridge 0.9.1: build-time patch и `tracing_bridge.py` включают
  tracing frontend, связывают metadata с `APIServerReqTimeStats`, передают его
  в scheduler через существующий pickle IPC, закрывают spans при отмене.
- Collector Contrib 0.160.0: OTLP/gRPC 4317, OTLP/HTTP 4318, память ≤512 MiB,
  очередь/retry и metadata allowlist. Метрики продолжают работать на `/metrics`;
  этот overlay экспортирует только traces.
- Опциональный Jaeger 2.20.0: локальный UI, временное хранение до 2000 traces
  в RAM, память ≤512 MiB. Перезапуск очищает историю.

Образы Collector и Jaeger закреплены digest. Patch проверяет версии и точные
исходные anchors; обновление зависимостей требует повторной проверки.

## Локальный просмотр

Из этой папки после настройки `.env` согласно README:

```bash
docker compose -f docker-compose.yaml -f compose.tracing.yaml -f compose.trace-ui.yaml config --quiet
docker compose -f docker-compose.yaml -f compose.tracing.yaml -f compose.trace-ui.yaml build sglang smg
docker compose -f docker-compose.yaml -f compose.tracing.yaml -f compose.trace-ui.yaml up -d --wait --wait-timeout 600
```

Для последующих коротких команд можно записать в **локальный** `.env`:

```dotenv
COMPOSE_FILE=docker-compose.yaml:compose.tracing.yaml:compose.trace-ui.yaml
RESTART_POLICY=no
```

`restart=no` применяется во время экспериментов. Published default —
`unless-stopped`. При первом включении tracing SGLang и SMG пересоздаются,
веса загружаются заново; model/cache bind-каталоги сохраняются.

На bx UI слушает localhost. Откройте SSH tunnel на своём компьютере:

```bash
ssh -N -L 16686:127.0.0.1:16686 192.168.0.59
```

Перейдите на <http://localhost:16686>. В Search выберите `smg` или `sglang`,
время `Last Hour`, нажмите Find Traces. Trace ID можно вставить в поиск либо
открыть `/trace/<trace_id>`. Внутри смотрите waterfall, родительские spans,
длительности queue/prefill/decode, `error.type`, `request.id`, token counts.

Проверочный запрос с собственным корневым span и автоматической проверкой
связей в Jaeger:

```bash
docker compose exec -T sglang python3 /opt/qwen38/trace_probe.py --mode stream
docker compose exec -T sglang python3 /opt/qwen38/trace_probe.py --mode nonstream
docker compose exec -T sglang python3 /opt/qwen38/trace_probe.py --mode cancel
docker compose exec -T sglang python3 /opt/qwen38/trace_probe.py --mode unsampled
```

Вывод содержит trace ID и URL path. `unsampled` проверяет отсутствие spans при sampling flag 00. В `cancel` клиент закрывает поток после
трёх содержательных chunks; SGLang span должен завершиться с `Cancelled`.
При использовании внешнего backend добавьте `--no-verify`: запрос и OTLP
root span сохраняются, проверка Jaeger API отключается.

## Общий OTLP backend

Включите `compose.tracing.yaml`. Уберите `compose.trace-ui.yaml` из локального
`COMPOSE_FILE`: локальный UI overlay всегда направляет данные в Jaeger.
Настройте `.env`:

```dotenv
COMPOSE_FILE=docker-compose.yaml:compose.tracing.yaml
OTLP_UPSTREAM_ENDPOINT=collector.example.net:4317
OTLP_UPSTREAM_INSECURE=false
# При необходимости; хранить только локально / в secret manager:
# OTLP_UPSTREAM_AUTHORIZATION=Bearer ...
TELEMETRY_ENVIRONMENT=production
```

`OTLP_UPSTREAM_ENDPOINT` — endpoint **gRPC**, с поддержкой traces. Внешнее
соединение использует TLS. Для private CA смонтируйте CA и добавьте `tls.ca_file`
в локальный конфиг Collector. Для backend, принимающего только OTLP/HTTP,
замените exporter на `otlp_http` и укажите его `traces_endpoint`.
Движки продолжают отправлять в `otel-collector:4317` внутри Docker network.

Валидация и применение:

```bash
docker compose run --rm --no-deps otel-collector validate --config=/etc/otelcol/config.yaml
docker compose up -d --no-deps otel-collector
```

При изменении содержимого bind-mounted YAML дополнительно выполните
`docker compose restart otel-collector`. Изменение destination Collector не
требует перезапуска GPU worker. Недоступность backend вызывает bounded retry;
очередь ограничена, длительный сбой может потерять telemetry.

Для upstream сервисов на другом Docker host откройте Collector в доверенной
сети: `OTLP_BIND=<LAN IP>` в `.env` и пересоздайте только Collector. Его входные
4317/4318 в данном примере без TLS/auth; внешние сети подключайте через свой
защищённый Collector endpoint. По умолчанию host bind — `127.0.0.1`.

## OpenWebUI

Проверено с OpenWebUI **0.9.6**. Укажите OpenAI-compatible connection на LiteLLM и включите backend tracing:

```dotenv
ENABLE_OTEL=True
ENABLE_OTEL_TRACES=True
ENABLE_OTEL_METRICS=False
ENABLE_OTEL_LOGS=False
OTEL_SERVICE_NAME=open-webui
OTEL_OTLP_SPAN_EXPORTER=grpc
OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector-host>:4317
OTEL_EXPORTER_OTLP_INSECURE=True
OTEL_TRACES_SAMPLER=parentbased_always_on
```

Нужны **оба** флага `ENABLE_OTEL` и `ENABLE_OTEL_TRACES`. Native instrumentation
FastAPI/aiohttp/httpx создаёт server/client spans и переносит W3C context в
исходящие HTTP requests. Endpoint должен быть доступен из контейнера WebUI;
`localhost` внутри него относится к самому WebUI. Для HTTPS/TLS используйте
настройки своего Collector и `OTEL_EXPORTER_OTLP_INSECURE=False`.
Перезапустите OpenWebUI после изменения env. Фронтенд браузера отдельно здесь
не инструментируется; trace начинается на backend OpenWebUI либо продолжается
из пришедшего `traceparent`.

## LiteLLM

Пример рассчитан на OTel v2, проверен с LiteLLM **1.89.1**. Добавьте env:

```dotenv
LITELLM_OTEL_V2=true
OTEL_EXPORTER=otlp_grpc
OTEL_ENDPOINT=http://<collector-host>:4317
OTEL_SERVICE_NAME=litellm
OTEL_TRACES_SAMPLER=parentbased_always_on
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=NO_CONTENT
```

Требуются `grpcio` и `opentelemetry-instrumentation-fastapi`; проверьте их наличие
в своём image. OTel v1 callback `otel` при выборе v2 отключите, остальные нужные
callbacks сохраните. [Официальная настройка v2](https://docs.litellm.ai/docs/observability/opentelemetry_v2).

Скопируйте `litellm_trace_headers.py` **рядом с фактическим config.yaml внутри
контейнера**. Например, `/trace/config.yaml` и `/trace/litellm_trace_headers.py`.
В конфиг добавьте entries из `litellm-tracing.example.yaml`, сохранив остальные
модели/callbacks. SMG API: `http://192.168.0.59:30000/v1`, либо
`http://smg:30000/v1` при общей Docker network. Для доступа с другого host
задайте `API_BIND=192.168.0.59` в локальном `.env` recipe и пересоздайте SMG;
текущий bind по умолчанию — `127.0.0.1`.

```yaml
litellm_settings:
  callbacks:
    - litellm_trace_headers.trace_headers
```

Callback инжектирует context активного span LiteLLM в `extra_headers`. Он
сохраняет остальные заголовки и sampling bit, заменяет устаревшие W3C headers.
При отсутствии активного tracing context ничего не меняет. Простое копирование
входящего `traceparent` способно сделать SMG соседним span по отношению к LiteLLM;
callback обеспечивает родительство от активного proxy span. LLM-call span LiteLLM
может быть соседним дочерним span, поскольку pre-call hook выполняется раньше.
Для OTel v1 и других версий отдельно проверьте active context и parentage.

Проверка через LiteLLM из контейнера SGLang:

```bash
TRACE_API_KEY=... docker compose exec -e TRACE_API_KEY -T sglang python3 /opt/qwen38/trace_probe.py --base-url http://<litellm-host>:4000 --expect-services litellm
```

Probe ожидает ту же модель `Qwen3.8-27B`. Он проверяет наличие `litellm` среди
**предков** SGLang span, включая SMG. Это выявляет разрыв дерева даже при общем
trace ID. Пользовательский ключ передаётся через окружение и не печатается.

Полная проверка через OpenWebUI использует его API token и backend endpoint:

```bash
TRACE_API_KEY=... docker compose exec -e TRACE_API_KEY -T sglang python3 /opt/qwen38/trace_probe.py --base-url http://<openwebui-host>:8080 --api-path /api/chat/completions --expect-services open-webui,litellm
```

Ключ должен принадлежать пользователю с доступом к модели `Qwen3.8-27B`.
`--mode cancel` дополнительно проверяет прохождение отмены до SGLang.
Batch exporters отправляют spans с задержкой; probe ждёт полное дерево до 30 s.
Фактическая проверка всей цепочки прошла в изолированных upstream контейнерах;
подключение ваших существующих OpenWebUI/LiteLLM выполняется настройками выше.

## OpenCode: воспроизводимая проверка

`opencode_trace.py` упаковывает проверенный smoke test OpenCode **1.17.7**.
Он создаёт root span `opencode.run` с `service.name=opencode-smoke` и передаёт
новый W3C context в HTTP headers провайдера. Один запуск CLI получает один
trace ID. Цикл с инструментами содержит несколько отдельных обращений к модели:

```text
opencode.run — запуск CLI целиком, включая старт процесса
├── OpenWebUI → LiteLLM → SMG → SGLang: запрос на запись файла
├── OpenWebUI → LiteLLM → SMG → SGLang: запрос на чтение файла
└── OpenWebUI → LiteLLM → SMG → SGLang: финальный ответ
```

Root span добавляет тестовая обёртка. Внутренние tool operations и подготовка
контекста OpenCode отдельных spans здесь не получают. `client_seconds`
включает startup CLI и работу инструментов; это полное время тестового запуска.
Для длительности inference выбирайте дочерние `sglang.generate`.

Подготовка:

1. Соберите/запустите SMG, SGLang, Collector и при необходимости Jaeger по
   разделам выше. Docker images для добавления OpenCode probe пересобирать
   не требуется: скрипт запускается на host.
2. Настройте свои OpenWebUI и LiteLLM по соответствующим разделам. OpenWebUI
   должен видеть модель `Qwen3.8-27B` через LiteLLM. Временные upstream containers
   из наших экспериментов удалены; порт 18090 сам по себе сервис не создаёт.
3. На Linux/POSIX host запуска нужны Python 3.9+, git и OpenCode 1.17.7. Python dependencies
   входят в stdlib. OpenCode в serving images не устанавливается.
4. Подготовьте OpenWebUI access token/API key с доступом к этой модели.
   `--base-url` указывает API root OpenWebUI, например
   `http://openwebui.example:8080/api`; SDK добавляет `/chat/completions`.

В bash, из папки recipe на host с OpenCode:

```bash
opencode --version
export OWUI_API_BASE='http://openwebui.example:8080/api'
read -r -s -p 'OpenWebUI token: ' TRACE_API_KEY
export TRACE_API_KEY
python3 opencode_trace.py --base-url "$OWUI_API_BASE" --mode all
unset TRACE_API_KEY
```

Замените адрес на свой. По умолчанию скрипт отправляет root span в
`http://127.0.0.1:4318/v1/traces` и проверяет Jaeger на
`http://127.0.0.1:16686`. Эти адреса относятся к host запуска OpenCode.
Для удалённых Collector/UI задайте `--otlp-http-endpoint` и `--jaeger-url`.
Первый параметр принимает полный OTLP/HTTP URL с `/v1/traces`. С внешним
trace backend используйте `--no-verify`: ответы и экспорт root span проверяются,
чтение Jaeger и проверка серверных spans пропускаются.

`--mode all` запускает три проверки:

| Mode | Проверка |
| --- | --- |
| `arithmetic` | Точный числовой ответ |
| `json` | Валидный JSON с ожидаемыми полями |
| `tools` | `write` → `read` одного временного файла → финальный ответ |

Каждый mode можно запустить отдельно. `--opencode /path/to/opencode` выбирает
binary; `--work-root /data/scratch` выбирает существующий родительский каталог
временного окружения. `--timeout 120` ограничивает один запуск CLI.

JSON-вывод содержит `passed`, `trace_id`, `jaeger_path`, число model requests,
длительности и результат проверки parent chain. Exit code 0 означает, что все
выбранные проверки прошли. Скрипт ждёт batch exporters до 30 s и проверяет
наличие всех пяти сервисов среди предков каждого SGLang request span.
В Jaeger выберите service `opencode-smoke` или откройте `/trace/<trace_id>`.
Для тестовых запусков sampling включён через флаг `01` в `traceparent`.

Скрипт использует `--pure`, явный `--dir`, отдельные XDG config/cache/data/state
и абсолютный путь тестового файла. Все инструменты запрещены по умолчанию;
для `tools` разрешены операции над `proof.txt`. Временное окружение, история
OpenCode и файлы удаляются после прогона; постоянные настройки OpenCode сохраняются.
Prompt/response и reasoning не пишутся в отчёт или root span. Проверка Jaeger
дополнительно ищет запрещённые payload attributes и синтетические маркеры.
Native OpenCode V2 и постоянная instrumentation произвольных CLI сессий в этот
probe не входят.

## Sampling, стоимость и границы

Для проверки используется 100% sampling. Для постоянной нагрузки задайте на
самом верхнем сервисе `OTEL_TRACES_SAMPLER=parentbased_traceidratio` и
`OTEL_TRACES_SAMPLER_ARG=0.01`; downstream оставьте `parentbased_always_on`.
При наличии входящего context применяется решение родителя. Новые прямые
запросы к SMG без родителя будут sampled согласно его собственному sampler.

`SGLANG_TRACE_LEVEL=1` оставляет основные стадии; уровень 3 добавляет подробные
итерации и значительно больше spans. Это CPU tracing scheduler/runtime;
CUDA kernel profiling включается отдельно. GPU KV/cache sizing сохранены.
Фактический overhead зависит от нагрузки и detail level.

Collector сохраняет только перечисленные в `transform/metadata_only` атрибуты
идентификаторов, статусов, модели, token counts и ranks. Prompt/completion,
HTTP headers, SQL/Redis statements и текст exception events исключаются из
экспорта. При расширении allowlist проверьте новые поля. Сырые application logs
этот recipe не отправляет. Поиск ошибок сохраняет `error.type` и span status.

При TP/DP>1 общий trace ID связывает процессы; rank attributes помогают выбрать
worker. Native DPA на нескольких GPU в этом профиле не проверен.

Отключение: уберите tracing overlays из `COMPOSE_FILE`, выставьте
`ENABLE_TRACING=0` и пересоздайте SMG/SGLang. Остановите Collector/Jaeger по имени
сервиса до удаления overlays. Проверка unit contract:

```bash
docker compose run --rm --no-deps --entrypoint python3 sglang /opt/qwen38/test_tracing.py
```

## Control Center / Nix

Nix app принимает `QWEN38_TRACING=1`, `QWEN38_TRACE_UI=1` и путь
`QWEN38_CONFIG_DIR` к локальной копии recipe. Feature override:

```bash
QWEN38_TRACING=1 QWEN38_TRACE_UI=1 ./cc feature qwen38-opencode-tracing-docs run nyashkimyashki-qwen38-sglang-smg-control config --quiet
QWEN38_TRACING=1 QWEN38_TRACE_UI=1 ./cc feature qwen38-opencode-tracing-docs run nyashkimyashki-qwen38-sglang-smg-control trace-check --mode stream
```

OpenCode probe запускается на host Nix app. Задайте `TRACE_API_KEY` способом
из раздела OpenCode перед вызовом:

```bash
./cc feature qwen38-opencode-tracing-docs run nyashkimyashki-qwen38-sglang-smg-control opencode-trace-check --base-url "$OWUI_API_BASE" --mode all
```

Операции: `build`, `up`, `logs`, `trace-config-check`, `trace-unit`, `trace-check`,
`opencode-trace-check`. Последняя требует установленный OpenCode на host.
Для remote Docker bind paths должны существовать на Docker host. В режиме Nix
явные flags выбора overlays имеют приоритет над `COMPOSE_FILE` из `.env`.

Sources: [SMG client factory](https://github.com/smg-project/smg/blob/v1.10.1/model_gateway/src/routers/grpc/client.rs),
[SMG tracing](https://github.com/smg-project/smg/blob/v1.10.1/model_gateway/src/observability/otel_trace.rs),
[bridge](https://github.com/smg-project/smg/blob/v1.10.1/grpc_servicer/smg_grpc_servicer/sglang/request_manager.py),
[SGLang tracing](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/observability/trace.py),
[OpenWebUI setup](https://github.com/open-webui/open-webui/blob/main/backend/open_webui/utils/telemetry/setup.py),
[Jaeger](https://www.jaegertracing.io/docs/2.20/getting-started/).
