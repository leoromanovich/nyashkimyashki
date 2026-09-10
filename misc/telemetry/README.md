# Сквозная трассировка inference recipes

Запрос проходит `OpenCode → OpenWebUI → LiteLLM → SMG → SGLang/vLLM`.
Каждая граница передаёт W3C `traceparent`/`tracestate`; spans попадают в один
trace по общему ID, даже если экспортируются через разные Collectors.

## Два режима

| Рецепт | Контейнеры телеметрии | Назначение |
| --- | --- | --- |
| Qwen и небольшие Gemma | SMG, Collector, Jaeger | готовый локальный просмотр |
| GLM, DeepSeek, Kimi | SMG, Collector | экспорт в существующее OTLP-хранилище |

Collector ограничен 512 MiB, очередь — 256 batches, retries — 60 секунд.
Он получает **только traces**. `/metrics` у SMG и движков доступны отдельно.
OWUI/LiteLLM для больших моделей не поднимаются этим репозиторием.
Mooncake/LMCache остаются частью cache-топологии соответствующих recipes;
собственных request spans этих cache-сервисов рецепт не добавляет.

Модельные параметры сохранены при реорганизации. Новый общий путь SMG → worker
использует HTTP и нативный tracing движка. Qwen3.8 сохраняет ранее проверенный
gRPC bridge с KV events и двумя compatibility patches. HTTP cache-aware policy
не даёт тот же набор сигналов о KV-состоянии, что Qwen gRPC/KV-event integration.
У vLLM внутренние DP ranks продолжают обслуживаться его собственным balancer.

## Запуск

Клонируйте весь репозиторий: recipes используют общие файлы через Compose
`extends`. Выполняйте команды в выбранной папке модели:

```bash
cp .env.example .env
# Отредактируйте пути к моделям/cache, ключи и внешний OTLP endpoint.
# Для экспериментов: RESTART_POLICY=no в локальном .env.
docker compose config --quiet
docker compose up -d --build
```

Для внешних recipes обязательны настройки `.env`:

```dotenv
OTLP_UPSTREAM_ENDPOINT=collector.example.internal:4317
OTLP_UPSTREAM_INSECURE=false
# Если upstream требует authentication:
OTLP_UPSTREAM_AUTHORIZATION='Bearer <token-from-secret-manager>'
TELEMETRY_ENVIRONMENT=production
SMG_BIND=192.168.0.59
SMG_PORT=30000
```

`OTLP_UPSTREAM_ENDPOINT` — **gRPC host:port**, без `/v1/traces`. Порт 4318 и
путь `/v1/traces` относятся к OTLP/HTTP при приёме Collector. Для upstream,
предоставляющего только HTTP, измените exporter в копии Collector config.
Для доверенной сети без TLS задайте `OTLP_UPSTREAM_INSECURE=true`.

При bridge networking Collector доступен движку как `otel-collector:4317`.
При host networking он слушает loopback на 14317/14318; это оставляет 4317/4318
для уже существующей инфраструктуры. Если запускаете несколько recipes на одном
host, назначьте уникальные `COLLECTOR_GRPC_PORT`, `COLLECTOR_HTTP_PORT`,
`ENGINE_PORT`, `SMG_PORT` и `SMG_METRICS_PORT`.

Native SGLang tracing — optional dependency. `Dockerfile.sglang` добавляет
OTel SDK/exporter к прежнему базовому image и проверяет import. vLLM 0.22.1,
0.26.0 и текущий Vision recipe используют bundled OTel dependencies.
Модельные tags/digests сохранены; старые mutable tags остаются legacy recipes.

Для Gemma MTP запускайте только выбранный worker:

```bash
SMG_WORKER_SERVICE=gemma4-mtp docker compose --profile mtp up -d --build gemma4-mtp smg otel-collector jaeger
```

Сначала остановите `gemma4`, если base уже работает: варианты используют одну
GPU и один порт. Qwen работает через свой gRPC gateway; его LAN bind задаётся
`API_BIND`, а его подробные проверки описаны в
[TRACING.md](../../sglang/qwen3.8-27b-fp8-hicache/TRACING.md).

## OpenWebUI и LiteLLM

В LiteLLM укажите OpenAI-compatible upstream `http://<inference-host>:30000/v1`,
model ID из `/v1/models` и inference API key из `.env`. Для обычных HTTP recipes
SMG и worker используют один ключ. Qwen gRPC recipe рассчитан на доверенный
локальный доступ и существующую auth-границу перед ним.

Проверенные upstream версии: OpenWebUI 0.9.6, LiteLLM 1.89.1.
Конфиг существующего OpenWebUI:

```dotenv
ENABLE_OTEL=True
ENABLE_OTEL_TRACES=True
ENABLE_OTEL_METRICS=False
ENABLE_OTEL_LOGS=False
OTEL_SERVICE_NAME=open-webui
OTEL_OTLP_SPAN_EXPORTER=grpc
OTEL_EXPORTER_OTLP_ENDPOINT=http://<shared-collector>:4317
OTEL_EXPORTER_OTLP_INSECURE=True
OTEL_TRACES_SAMPLER=parentbased_always_on
```

Конфиг существующего LiteLLM:

```dotenv
LITELLM_OTEL_V2=true
OTEL_EXPORTER=otlp_grpc
OTEL_ENDPOINT=http://<shared-collector>:4317
OTEL_SERVICE_NAME=litellm
OTEL_TRACES_SAMPLER=parentbased_always_on
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=NO_CONTENT
```

Примеры выше используют trusted plaintext receiver. Для TLS настройте exporter
по требованиям инфраструктуры. `localhost` внутри контейнера относится к нему
самому. У LiteLLM нужны `grpcio` и `opentelemetry-instrumentation-fastapi`.

Смонтируйте [litellm_trace_headers.py](litellm_trace_headers.py) **рядом с
фактическим config.yaml внутри LiteLLM** и добавьте callback к существующим:

```yaml
litellm_settings:
  callbacks:
    - litellm_trace_headers.trace_headers
```

При OTel v2 отключите старый callback `otel`; остальные callbacks сохраните.
Наш hook инжектирует активный proxy span в `extra_headers`. Простое копирование
входящего `traceparent` может потерять правильного родителя LiteLLM.
Оба upstream приложения могут отправлять traces через свой общий Collector;
он должен вести в тот же backend, что Collectors моделей.

## Конфиденциальность

В нативных request spans сохраняются IDs, длительности, ranks, token counts,
model/status metadata. Захват prompts/responses не включён. Collector применяет
allowlist к span/resource/event attributes, очищает status message, убирает
query string из span names и нормализует event names. Токены авторизации,
заголовки и текст исключений в экспорт не попадают. Debug exporter отсутствует.

Примените ту же политику к Collector существующих OWUI/LiteLLM, если их spans
поступают в backend отдельным путём. Новые custom spans не должны использовать
содержимое запроса в имени span или разрешённом атрибуте. Рецепт не меняет
обычные application logs и историю чатов существующих приложений.

Фильтр сохраняет `gen_ai.latency.*` для queue, TTFT, prefill, decode и inference,
а также token counts и error status. Детальное `--collect-detailed-traces all`
у vLLM не включено: оно добавляет накладные расходы. SGLang использует trace
level 1; для короткой диагностики можно повысить `SGLANG_TRACE_LEVEL`.

## Посмотреть конкретный запрос

Локальный Jaeger: `http://localhost:16686`. Для просмотра с другого компьютера:

```bash
ssh -N -L 16686:127.0.0.1:16686 <inference-host>
```

Выберите service `smg`, `sglang` или `vllm`, время и Find Traces. Если известен
trace ID, откройте `/trace/<id>`. В общем backend используйте поиск по тому же
ID. Смотрите parent/child waterfall и tags:

- SGLang: корневой `Req …` и scheduler/prefill/decode stages; Qwen gRPC —
  `sglang.generate`. Набор стадий зависит от версии движка.
- vLLM: `llm_request` и `gen_ai.latency.time_in_queue`, `time_to_first_token`,
  `time_in_model_prefill`, `time_in_model_decode` в **секундах**. Для этих версий
  queue/prefill/decode представлены числовыми длительностями внутри span.
- SMG: полный `http_request`; поле `latency` отражает получение response headers.
- `inference.recipe` различает модели при одинаковых service names `smg`/`sglang`.

## Проверочный запрос

Python-клиент [trace_probe.py](trace_probe.py) использует только stdlib, создаёт
собственный root и печатает trace ID без содержимого запроса или ответа:

```bash
# Выполняется из каталога модели; ключ передайте через окружение.
read -r -s -p 'Inference API key: ' TRACE_API_KEY
export TRACE_API_KEY
docker compose exec -T -e TRACE_API_KEY smg python3 /opt/telemetry/trace_probe.py --model <served-model-name> --engine vllm
unset TRACE_API_KEY
```

Для SGLang замените `--engine vllm` на `--engine sglang`. Для host networking
добавьте `--base-url http://127.0.0.1:30000` и
`--otlp-http-endpoint http://127.0.0.1:14318/v1/traces`.
Для внешнего backend добавьте `--no-verify` либо `--jaeger-url <read-api-url>`.
`--no-verify` подтверждает запрос и приём root Collector; доставку и полное
дерево проверьте в своём backend. Есть режимы `--mode nonstream` и `unsampled`.

Для пути через OWUI передайте его API token, `--base-url http://<owui>:8080`,
`--api-path /api/chat/completions`, модель alias и
`--expect-services open-webui,litellm`. Экспортеры могут доставлять части дерева
до 30 секунд; проверка ожидает полный parent chain.

[OpenCode CLI probe](../../sglang/qwen3.8-27b-fp8-hicache/TRACING.md#opencode-воспроизводимая-проверка)
остаётся специализированным для Qwen3.8. Он создаёт root вокруг CLI run;
внутренние tool spans OpenCode отдельно не инструментированы.

## Nix

В LLM Engines Control Center задайте путь к checkout для относительных runtime mounts:

```bash
export RECIPE_ROOT=/path/to/nyashkimyashki
./cc feature recipes-layout-tracing run nyashkimyashki-inference-recipe-control vllm/kimi-k2.6-dep8-eagle3 config
./cc feature recipes-layout-tracing run nyashkimyashki-inference-recipe-control vllm/kimi-k2.6-dep8-eagle3 build --env-file /secure/path/model.env
RECIPE_CONFIRM=mutate-inference ./cc feature recipes-layout-tracing run nyashkimyashki-inference-recipe-control vllm/kimi-k2.6-dep8-eagle3 up --env-file /secure/path/model.env
```

App поддерживает config/build/up/stop/down/logs/ps/trace-check; `--variant mtp`
выбирает Gemma MTP. Для trace-check задайте `--model` и `TRACE_API_KEY` в env.
`config` проверяет `.env.example` без вывода секретов. Runtime использует `.env`
или явный `--env-file`. Для up/stop/down требуется `RECIPE_CONFIRM=mutate-inference`.
GPU preflight и model smoke из README конкретного рецепта остаются обязательными.

## Проверки и границы

[CPU fixture](tests/README.md) прошёл настоящий SMG HTTP routing → synthetic
worker → Collector → Jaeger для SGLang и vLLM: stream, nonstream и unsampled.
Проверены общий ID, родительские связи, удаление синтетических payload attributes,
event names/status messages и сохранение timing/token metadata.

Это проверка transport/export/filter; multi-GPU модели, новые SGLang wrapper
images и GPU cold start после реорганизации ещё требуют запуска на целевом host.
Исторические Qwen GPU и OpenCode результаты описаны в его VALIDATION.md.
Модельные параметры, native caches и production-сервисы этим CPU-тестом не менялись.

Источники: [SMG v1.10.1 HTTP injection](https://github.com/smg-project/smg/blob/v1.10.1/model_gateway/src/routers/http/router.rs),
[vLLM v0.22.1 request timing](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/engine/output_processor.py),
[SGLang tracing](https://github.com/sgl-project/sglang/blob/v0.5.12.post1/python/sglang/srt/observability/trace.py),
[LiteLLM OTel v2](https://docs.litellm.ai/docs/observability/opentelemetry_v2).
