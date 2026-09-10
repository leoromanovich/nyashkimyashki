# CPU tracing contract

`docker compose` здесь запускает синтетические HTTP workers, SMG, Collector и
Jaeger. GPU и веса отсутствуют. Проверяются настоящий SMG HTTP transport,
OTLP exporter, фильтрация Collector и parentage; native inference runtime этим
тестом не проверяется. При тесте все контейнеры используют локальный `RESTART_POLICY=no`; published default — `unless-stopped`.

```bash
RESTART_POLICY=no JAEGER_UI_PORT=16687 docker compose up -d --build --wait
TRACE_API_KEY=synthetic-fixture-key python3 ../trace_probe.py --engine vllm --model synthetic --base-url http://127.0.0.1:33000 --otlp-http-endpoint http://127.0.0.1:14328/v1/traces --jaeger-url http://127.0.0.1:16687
TRACE_API_KEY=synthetic-fixture-key python3 ../trace_probe.py --engine sglang --model synthetic --base-url http://127.0.0.1:33002 --otlp-http-endpoint http://127.0.0.1:14328/v1/traces --jaeger-url http://127.0.0.1:16687
RESTART_POLICY=no JAEGER_UI_PORT=16687 docker compose down -v
```

Повторить с `--mode nonstream` и `--mode unsampled`. Mock намеренно добавляет
синтетический prompt в attributes, event name и status message; в Jaeger они
должны отсутствовать, а timing/token attributes должны сохраниться.
