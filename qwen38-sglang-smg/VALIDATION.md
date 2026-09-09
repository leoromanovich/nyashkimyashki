# Проверенный профиль — 2026-09-09

RTX 4090 48 GiB, 128 GiB RAM, NVMe, SGLang 0.5.19 + SMG 1.10.1,
bridge 0.9.1 / proto 0.4.16. MTP EAGLE 3/1/4 + ReplaySSM, context 32768,
mem_fraction 0.94, GDN pool 35, host 64 decimal GB.

| KV / GDN | GPU KV tokens | Host KV tokens | GDN SSM GiB |
| --- | ---: | ---: | ---: |
| BF16 / FP32 | 82176 | 444160 | 5.06 |
| FP8 E4M3 / FP32 | 164416 | 888320 | 5.06 |
| FP8 E4M3 / BF16 | 247360 | 1336256 | 2.53 |

GPU KV вырос в 3.0101 раза. BF16 GDN освободил 2.53125 GiB с учётом padding
slot. FP8 packed MTP KV: 34816 bytes/token, 2228224 bytes/page. BF16 GDN
checkpoint с conv: 78446592 bytes. В checkpoint нет KV scales, fallback=1.0.

## Функциональная проверка

- Оба новых профиля: 11/11 arithmetic/retrieval + 9/9 retrieval с needle
  около 10/50/90% текста и input до 28723 tokens.
- Генерация 2048 tokens завершилась; смысл длинного ответа не оценивался.
- Final профиль: 38/38 ответов input8192/output128, плюс 16/16 длинных
  запросов input24576/28672, output1024, C8.
- Восемь running подтверждены для 8K; в длинных burst максимум семь,
  пик used KV 207872. Причина дополнительной admission границы не установлена.
- 14 отмен HTTP/gRPC: очереди освободились, subsequent completion 16 tokens
  завершился. В логах worker/gateway нет ERROR/Traceback.
- SMG event stream подключён, block 64 изучен, GetLoads(all) работает.

## Скорость

Один сопоставимый прогон после очистки старых SSD caches, cold/repeat;
cache pressure между профилями различается. Универсальное ускорение не доказано.

| Input8192, repeat | BF16 KV / FP32 GDN | FP8 KV / BF16 GDN |
| --- | ---: | ---: |
| C1 TPOT median ms | 13.184 | 15.485 |
| C8 output tok/s | 130.562 | 150.145 |
| C8 TTFT median s | 2.122 | 3.042 |

## SSD restore

Exact 8192-token retrieval, успешный FlushCache и POSIX_FADV_DONTNEED только
файлов этого cache. Cached/prefetched 8128 tokens; физические NVMe reads
361443328 bytes. Контрольный ответ верен cold/SSD/warm.

| Проход | TTFT s |
| --- | ---: |
| Cold | 2.006 |
| SSD | 3.074 |
| GPU repeat | 0.143 |

Полный hash принудительных 32 output tokens cold отличается от SSD/warm;
SSD и warm совпадают. Общая эквивалентность точности не установлена.
В wait_complete SSD restore в этом probe медленнее полного prefill.

Полные локальные экспериментальные журналы сохранены отдельно и исключены
из Git/Docker context. Результаты воспроизводятся включёнными benchmark.py,
inspect_cache.py и disconnect_probe.py; модельные тексты не сохраняются.

## Проверка самостоятельной папки

- Оба Docker images собраны через Compose на Linux x86_64; build context
  ограничен Dockerfile и runtime scripts.
- Одноразовый Compose download отработал с закреплённой revision и уже
  материализованными весами; host hf/Python не использовались.
- Smoke и GetLoads(all) из новых images через существующий API прошли:
  GPU KV 247360, max_running 8, очереди пусты. SMG image сообщает 1.10.1.
- Worker argv побайтно совпал с GPU-проверенным профилем: 62 аргумента;
  SMG 28. Compose config --quiet, argv и restart policy прошли для defaults,
  .env.example и экспериментального RESTART_POLICY=no.
- Nix path-snapshot с feature override: 11/11 checks на aarch64-darwin;
  knowledge validation: 50 notes. Остальные Nix platforms не проверялись.

GPU worker для проверки упаковки не пересоздавался: параметры serving сохранены,
проверены новые image build и упакованные команды. Сырые журналы и полный
исследовательский отчёт остаются в локальном архиве.
