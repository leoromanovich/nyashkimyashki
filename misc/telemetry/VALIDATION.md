# Validation · 2026-09-10

- 15 самостоятельных recipes: Compose quiet + JSON argv; local/external topology,
  README/migration paths, restart default и экспериментальный override.
- Три sticky Compose и CPU fixture: Compose/argv проверены отдельно.
- Model contracts: обе Gemma, SGLang DeepSeek H200 и vLLM DeepSeek LMCache прошли.
- Shell/Python/YAML syntax прошли, duplicate YAML mappings устранены.
- SMG 1.10.1 CPU image собран и запущен. Реальные HTTP routing и OTLP export
  проверены с синтетическими SGLang/vLLM workers: stream/nonstream/unsampled.
- Проверены CHILD_OF, общий trace ID, сохранение queue/token/error metadata;
  synthetic prompt удалён из attributes, event names и status message.
- SGLang HTTP `--dp-aware` + API key прошёл на synthetic DP1 worker.
  Auth на inference POST: missing/invalid key → 401, valid key → 200.
  Публичный `/v1/models` не является проверкой inference auth.
- Nix: все 12 checks прошли offline с feature input override; обычный запуск
  `./cc` блокирует выключенный nix-command, онлайн cache недоступен по DNS.
  Новый Nix app выполнил config Kimi без вывода resolved credentials.

GPU модели после миграции не запускались. Legacy model-specific flags и старые
mutable image tags не получили новой GPU-квалификации. Проверенный ранее Qwen
standalone gRPC bridge сохранён; его исторические результаты — в recipe.
CPU synthetic workers проверяют transport/filter, а native queue/prefill/decode
взяты из source движков и требуют canary на целевом hardware.
