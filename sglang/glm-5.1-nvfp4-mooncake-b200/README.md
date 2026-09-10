# glm-5.1-nvfp4-mooncake-b200

## Запуск и трассировка

Состав: **sglang + SMG + Collector**.
TP8/DP8, Mooncake SSD, B200; legacy flags. Параметры модели и cache budgets сохранены при переносе.

```bash
cp .env.example .env
# Настройте пути, API keys и OTLP_UPSTREAM_ENDPOINT в .env.
# Для экспериментов добавьте RESTART_POLICY=no.
docker compose config --quiet
docker compose up -d --build
```

Внешний клиент обращается к SMG на `http://<host>:30000/v1`; укажите одинаковый
inference API key в `.env` и LiteLLM. Публичный bind по умолчанию — localhost;
для доступа с другого сервера задайте `SMG_BIND=<LAN IP>`.
Collector отправляет только очищенные traces в обязательный внешний OTLP/gRPC endpoint; TLS включён по умолчанию. OWUI, LiteLLM и Jaeger этим Compose не создаются.

[Интеграция OWUI/LiteLLM, просмотр запроса и проверочный клиент](../../misc/telemetry/README.md).
[Запуск через Nix](../../misc/telemetry/README.md#nix).

## Состояние рецепта

Модельные параметры перенесены из старого Compose без GPU-переизмерений.
SMG HTTP propagation и Collector проверены на синтетическом CPU-стенде.
Перед эксплуатацией проверьте сборку image, cold start, короткий/длинный запрос,
streaming, tools и один полный trace на целевом GPU-сервере.
Старые версии движков сохранены; runtime совместимость их model-specific flags
после этой реорганизации на GPU не подтверждена.

Для Mooncake требуется выделенный каталог SSD с filesystem quota. Перенос
папки меняет базу относительных bind mounts: задайте прежние абсолютные пути,
если хотите использовать уже скачанные веса, cache и данные сервисов.
