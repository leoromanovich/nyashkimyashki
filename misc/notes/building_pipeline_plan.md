# Целевая структура в repo B

```text
repo-b/
  build.yaml
  build.lock
  justfile
  Dockerfile
  .env.example
  .env

  tools/
    pipeline.py
    common.py              # опционально позже
    artifact_utils.py      # опционально позже
    git_utils.py           # опционально позже

  artifacts/
    wheels/
    lib/
    bin/
    include/
    data/

  .work/
    repo_a/
    repo_c/
    repo_d/

  docs/
    build.md
```

На первом этапе можно оставить только:

```text
repo-b/
  build.yaml
  justfile
  tools/pipeline.py
```

Остальное добавить по мере стабилизации.

---

# Этап 0. Инвентаризация существующего процесса

Перед кодом надо описать текущую реальность. Не “как должно быть”, а именно “как сейчас работает”.

Для каждого внешнего репозитория нужно заполнить такую таблицу:

```text
repo        откуда берём       branch/tag/sha       submodules       команда сборки        что является результатом       куда сейчас кладётся
repo_a      git@.../a.git       main                 recursive        ./build.sh            build/libx.so                  repo-b/vendor/lib/
repo_c      git@.../c.git       release/v1           no               python -m build       dist/*.whl                     repo-b/wheels/
repo_d      не собираем         version 1.2.3         no               нет                   package.whl                    repo-b/wheels/
```

Также отдельно:

```text
какие env нужны
какие системные зависимости нужны
какие команды надо запускать из repo B
какие команды надо запускать из repo A/C
какие пути сейчас захардкожены
что должно попасть в Docker build context
какой smoke test показывает, что контейнер живой
```

На этом этапе задача — не улучшать, а **сделать карту текущего процесса**.

---

# Этап 1. Добавить минимальный `justfile`

Сначала `just` должен быть просто удобным входом.

```make
set dotenv-load
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

PY := ".venv/bin/python"

default:
    just --list

bootstrap:
    python3 -m venv .venv
    {{PY}} -m pip install -U pip
    {{PY}} -m pip install pyyaml

all:
    {{PY}} tools/pipeline.py all

fetch:
    {{PY}} tools/pipeline.py fetch

build:
    {{PY}} tools/pipeline.py build

docker:
    {{PY}} tools/pipeline.py docker-build

run:
    {{PY}} tools/pipeline.py run

check:
    {{PY}} tools/pipeline.py check

clean:
    rm -rf .work artifacts

clean-artifacts:
    rm -rf artifacts

clean-work:
    rm -rf .work
```

Первый ожидаемый интерфейс:

```bash
just bootstrap
just all
```

Для разработчика это должно выглядеть просто.

---

# Этап 2. Описать текущий процесс в `build.yaml`

На первом проходе `build.yaml` должен быть максимально близок к текущим ручным командам.

Например:

```yaml
workspace: .work
artifacts_dir: artifacts

env:
  CMAKE_BUILD_PARALLEL_LEVEL: "16"
  UV_LINK_MODE: copy

repos:
  repo_a:
    type: git
    url: git@github.com:company/repo-a.git
    ref: main
    submodules: recursive

    build:
      - ./existing_build_script.sh

    artifacts:
      - from: build/librepo_a.so
        to: artifacts/lib/

  repo_c:
    type: git
    url: git@github.com:company/repo-c.git
    ref: release/v1
    submodules: false

    build:
      - python -m build --wheel

    artifacts:
      - from: dist/*.whl
        to: artifacts/wheels/

prebuilt_artifacts:
  repo_d_wheel:
    from: ../repo-d-dist/repo_d-1.2.3-py3-none-any.whl
    to: artifacts/wheels/

docker:
  context: .
  dockerfile: Dockerfile
  tag: company/repo-b-image:dev
  build_args:
    ENABLE_FEATURE_X: "1"

run:
  name: repo-b-dev
  image: company/repo-b-image:dev
  env_file: .env
  ports:
    - "8000:8000"
  volumes:
    - "/models:/models"
  args:
    - "--host"
    - "0.0.0.0"
    - "--port"
    - "8000"

checks:
  - name: health
    command: curl -fsS http://localhost:8000/health

  - name: import-check
    command: docker run --rm company/repo-b-image:dev python -c "import my_package"
```

На этом этапе нормально, если внутри `build` пока остаются старые команды:

```yaml
build:
  - ./build_old_way.sh
```

Это даже желательно. Сначала надо обернуть существующее, а не переписывать.

---

# Этап 3. Сделать минимальный `pipeline.py`

На первом этапе runner должен уметь:

```text
читать build.yaml
создавать .work/
клонировать репозитории
checkout ref
обновлять submodules
запускать build commands
копировать artifacts
собирать docker
запускать checks
```

Минимальный скелет:

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "build.yaml"


class PipelineError(RuntimeError):
    pass


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_cmd(
    cmd: str | list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    print(f"\n$ {cmd}")
    print(f"  cwd={cwd}")

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    subprocess.run(
        cmd,
        cwd=str(cwd),
        env=full_env,
        shell=isinstance(cmd, str),
        check=True,
    )


def workspace(config: dict[str, Any]) -> Path:
    return ROOT / config.get("workspace", ".work")


def global_env(config: dict[str, Any]) -> dict[str, str]:
    return {str(k): str(v) for k, v in (config.get("env") or {}).items()}


def fetch_repo(config: dict[str, Any], name: str, repo: dict[str, Any]) -> None:
    path = workspace(config) / name
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        run_cmd(["git", "clone", repo["url"], str(path)], cwd=ROOT)
    else:
        run_cmd(["git", "fetch", "--all", "--tags", "--prune"], cwd=path)

    run_cmd(["git", "checkout", str(repo["ref"])], cwd=path)

    # На первом этапе pull допустим для branch-based процесса.
    # Позже лучше перейти на lock по commit SHA.
    run_cmd(["git", "pull", "--ff-only"], cwd=path)

    submodules = repo.get("submodules", False)
    if submodules == "recursive":
        run_cmd(["git", "submodule", "update", "--init", "--recursive"], cwd=path)
    elif submodules is True:
        run_cmd(["git", "submodule", "update", "--init"], cwd=path)


def build_repo(config: dict[str, Any], name: str, repo: dict[str, Any]) -> None:
    path = workspace(config) / name

    env = global_env(config)
    env.update({str(k): str(v) for k, v in (repo.get("env") or {}).items()})

    for cmd in repo.get("build", []):
        run_cmd(cmd, cwd=path, env=env)


def copy_artifacts(config: dict[str, Any], name: str, repo: dict[str, Any]) -> None:
    repo_path = workspace(config) / name

    for artifact in repo.get("artifacts", []):
        src_pattern = repo_path / artifact["from"]
        dst = ROOT / artifact["to"]

        matches = [Path(p) for p in glob.glob(str(src_pattern))]
        if not matches:
            raise PipelineError(f"No artifacts matched: {src_pattern}")

        for src in matches:
            if dst.as_posix().endswith("/") or dst.suffix == "":
                dst.mkdir(parents=True, exist_ok=True)
                final_dst = dst / src.name
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                final_dst = dst

            print(f"copy {src} -> {final_dst}")
            shutil.copy2(src, final_dst)


def copy_prebuilt(config: dict[str, Any]) -> None:
    for name, item in (config.get("prebuilt_artifacts") or {}).items():
        src_pattern = ROOT / item["from"]
        dst = ROOT / item["to"]

        matches = [Path(p) for p in glob.glob(str(src_pattern))]
        if not matches:
            raise PipelineError(f"No prebuilt artifact matched: {src_pattern}")

        for src in matches:
            if dst.as_posix().endswith("/") or dst.suffix == "":
                dst.mkdir(parents=True, exist_ok=True)
                final_dst = dst / src.name
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                final_dst = dst

            print(f"copy {src} -> {final_dst}")
            shutil.copy2(src, final_dst)


def build_all(config: dict[str, Any]) -> None:
    artifacts_dir = ROOT / config.get("artifacts_dir", "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for name, repo in (config.get("repos") or {}).items():
        print(f"\n=== {name}: fetch ===")
        fetch_repo(config, name, repo)

        print(f"\n=== {name}: build ===")
        build_repo(config, name, repo)

        print(f"\n=== {name}: artifacts ===")
        copy_artifacts(config, name, repo)

    copy_prebuilt(config)


def docker_build(config: dict[str, Any]) -> None:
    docker = config["docker"]

    cmd = [
        "docker",
        "build",
        "-t",
        docker["tag"],
        "-f",
        str(ROOT / docker.get("dockerfile", "Dockerfile")),
    ]

    for k, v in (docker.get("build_args") or {}).items():
        cmd += ["--build-arg", f"{k}={v}"]

    cmd.append(str(ROOT / docker.get("context", ".")))

    run_cmd(cmd, cwd=ROOT)


def docker_run(config: dict[str, Any]) -> None:
    run = config["run"]

    cmd = ["docker", "run", "--rm"]

    if run.get("name"):
        cmd += ["--name", run["name"]]

    if run.get("env_file"):
        cmd += ["--env-file", str(ROOT / run["env_file"])]

    for port in run.get("ports", []):
        cmd += ["-p", str(port)]

    for volume in run.get("volumes", []):
        cmd += ["-v", str(volume)]

    cmd.append(run["image"])
    cmd += [str(x) for x in run.get("args", [])]

    run_cmd(cmd, cwd=ROOT)


def checks(config: dict[str, Any]) -> None:
    for check in config.get("checks", []):
        print(f"\n=== check: {check['name']} ===")
        run_cmd(check["command"], cwd=ROOT, env=global_env(config))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["all", "fetch", "build", "docker-build", "run", "check"],
    )
    parser.add_argument("repo", nargs="?")
    args = parser.parse_args()

    config = load_config()

    try:
        if args.command == "all":
            build_all(config)
            docker_build(config)
            checks(config)

        elif args.command == "fetch":
            for name, repo in (config.get("repos") or {}).items():
                if args.repo and args.repo != name:
                    continue
                fetch_repo(config, name, repo)

        elif args.command == "build":
            for name, repo in (config.get("repos") or {}).items():
                if args.repo and args.repo != name:
                    continue
                fetch_repo(config, name, repo)
                build_repo(config, name, repo)
                copy_artifacts(config, name, repo)

        elif args.command == "docker-build":
            docker_build(config)

        elif args.command == "run":
            docker_run(config)

        elif args.command == "check":
            checks(config)

    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed: exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    except PipelineError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Это ещё не идеальный runner, но он уже позволяет адаптировать существующий процесс.

---

# Этап 4. Перенести текущие команды без изменения логики

Допустим, сейчас разработчик делает руками:

```bash
cd ../repo-a
git checkout main
git submodule update --init --recursive
./scripts/build_release.sh
cp build/libfoo.so ../repo-b/vendor/lib/

cd ../repo-c
git checkout release/v1
python -m build --wheel
cp dist/*.whl ../repo-b/wheels/

cd ../repo-b
docker build -t company/app:dev .
docker run ...
```

На первом шаге это превращается в:

```yaml
repos:
  repo_a:
    url: git@github.com:company/repo-a.git
    ref: main
    submodules: recursive
    build:
      - ./scripts/build_release.sh
    artifacts:
      - from: build/libfoo.so
        to: artifacts/lib/

  repo_c:
    url: git@github.com:company/repo-c.git
    ref: release/v1
    submodules: false
    build:
      - python -m build --wheel
    artifacts:
      - from: dist/*.whl
        to: artifacts/wheels/
```

То есть сначала мы меняем только одно:

```text
ручные действия → build.yaml + pipeline.py
```

Команды сборки пока остаются старыми.

---

# Этап 5. Привести Dockerfile к новой модели

Dockerfile должен перестать зависеть от внешних репозиториев напрямую.

Плохо:

```dockerfile
RUN git clone git@github.com:company/repo-a.git
RUN cd repo-a && ./build.sh
```

Нормально:

```dockerfile
COPY artifacts/wheels/ /tmp/wheels/
COPY artifacts/lib/ /opt/app/lib/

RUN pip install --no-cache-dir /tmp/wheels/*.whl

ENV LD_LIBRARY_PATH=/opt/app/lib:$LD_LIBRARY_PATH
```

То есть:

```text
pipeline.py отвечает за сборку
Dockerfile отвечает за упаковку
```

Это важное разделение. Иначе процесс опять расползётся.

---

# Этап 6. Добавить `build.lock`

Когда базовая сборка заработала, надо фиксировать фактическое состояние.

`build.lock` должен отвечать на вопрос:

```text
из каких именно коммитов и каких именно файлов собран этот Docker image?
```

Пример:

```yaml
repos:
  repo_a:
    url: git@github.com:company/repo-a.git
    requested_ref: main
    resolved_sha: 2f8c0d9a4a2e...
    submodules:
      third_party/foo: 91ab23...

  repo_c:
    url: git@github.com:company/repo-c.git
    requested_ref: release/v1
    resolved_sha: a0d71e2f91...

artifacts:
  - name: librepo_a.so
    source_repo: repo_a
    path: artifacts/lib/librepo_a.so
    sha256: d7a8...

  - name: repo_c-0.1.0-py3-none-any.whl
    source_repo: repo_c
    path: artifacts/wheels/repo_c-0.1.0-py3-none-any.whl
    sha256: 4f91...

docker:
  tag: company/repo-b-image:dev
  image_id: sha256:...
```

После этого стоит добавить режимы:

```bash
just all
just all --use-lock
just lock
```

Смысл:

```text
обычная разработка: ref может быть branch
воспроизводимая сборка: checkout по resolved_sha из build.lock
```

---

# Этап 7. Добавить поддержку адаптации под существующие особенности

В реальности в старом процессе почти всегда есть нюансы. Их лучше явно предусмотреть в `build.yaml`.

## 7.1. Команды до сборки

Например, где-то нужно применить патч, включить venv, скачать зависимости:

```yaml
repos:
  repo_a:
    url: git@github.com:company/repo-a.git
    ref: main
    submodules: recursive

    before_build:
      - git apply ../../patches/repo_a_fix.patch
      - ./scripts/bootstrap.sh

    build:
      - ./scripts/build_release.sh
```

В runner:

```python
for cmd in repo.get("before_build", []):
    run_cmd(cmd, cwd=path, env=env)

for cmd in repo.get("build", []):
    run_cmd(cmd, cwd=path, env=env)
```

## 7.2. Команды после сборки

```yaml
after_build:
  - strip build/librepo_a.so
```

## 7.3. Разные директории сборки

```yaml
repos:
  repo_a:
    build_dir: cpp
    build:
      - cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
      - cmake --build build --parallel
    artifacts:
      - from: cpp/build/libfoo.so
        to: artifacts/lib/
```

Или лучше:

```yaml
repos:
  repo_a:
    workdir: cpp
    build:
      - cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
      - cmake --build build --parallel
    artifacts:
      - from: cpp/build/libfoo.so
        to: artifacts/lib/
```

В runner:

```python
repo_cwd = path / repo.get("workdir", ".")
```

## 7.4. Локальные репозитории вместо clone

Иногда разработчик уже имеет локальный checkout.

Можно добавить:

```yaml
repos:
  repo_a:
    type: local
    path: ../repo-a
    ref: main
    submodules: recursive
    build:
      - ./scripts/build_release.sh
    artifacts:
      - from: build/libfoo.so
        to: artifacts/lib/
```

Но я бы добавлял это не сразу. Сначала лучше сделать один стандартный путь через `.work`.

## 7.5. Переменные окружения на уровне репозитория

```yaml
repos:
  repo_a:
    env:
      CC: clang
      CXX: clang++
      CUDA_HOME: /usr/local/cuda
    build:
      - ./build.sh
```

## 7.6. Проверка наличия обязательных файлов

```yaml
repos:
  repo_a:
    required_outputs:
      - build/librepo_a.so
      - build/repo_a_config.json
```

И runner должен падать, если чего-то нет.

---

# Этап 8. Сделать smoke checks реальными

Проверки должны быть максимально тупыми, но полезными.

Например:

```yaml
checks:
  - name: image-imports
    command: docker run --rm company/repo-b-image:dev python -c "import package_a; import package_c"

  - name: binary-exists
    command: docker run --rm company/repo-b-image:dev test -f /opt/app/lib/librepo_a.so

  - name: ldd-check
    command: docker run --rm company/repo-b-image:dev ldd /opt/app/lib/librepo_a.so

  - name: health
    command: curl -fsS http://localhost:8000/health
```

Если контейнер должен быть запущен перед healthcheck, лучше разделить:

```bash
just docker
just run
just check
```

Или добавить отдельную команду:

```bash
just smoke
```

где runner сам делает:

```text
docker run -d
ждёт health endpoint
делает checks
останавливает контейнер
```

Но это лучше добавить после MVP.

---

# Этап 9. Не ломать существующий процесс сразу

Переход должен быть таким:

## Шаг 1

Существующий процесс остаётся рабочим.

Добавляется:

```bash
just all
```

Но он пока считается экспериментальным.

## Шаг 2

`just all` должен собрать то же самое, что ручной процесс.

Сравнить:

```text
одинаковые whl?
одинаковые .so?
одинаковый Docker image behavior?
одинаковые параметры запуска?
```

## Шаг 3

Новые разработчики начинают использовать `just`.

Старые ручные скрипты остаются как fallback.

## Шаг 4

Когда `just all` стабилен, старые команды помечаются deprecated.

Например:

```bash
scripts/old_build.sh
```

внутри:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "Deprecated: use 'just all' from repo-b instead"
just all
```

## Шаг 5

Через некоторое время старые скрипты удаляются или становятся thin wrappers.

---

# Этап 10. Сделать `docs/build.md`

Документация должна быть короткой и конкретной.

Пример:

````markdown
# Build

## Первый запуск

```bash
just bootstrap
cp .env.example .env
just all
````

## Собрать один внешний репозиторий

```bash
just build repo_a
```

## Собрать Docker image

```bash
just docker
```

## Запустить контейнер

```bash
just run
```

## Проверить

```bash
just check
```

## Где править

* `build.yaml` — репозитории, ветки, команды сборки, артефакты.
* `Dockerfile` — упаковка готовых артефактов.
* `.env` — локальные параметры запуска.
* `tools/pipeline.py` — логика исполнения.

````

---

# Этап 11. Рекомендуемая последовательность внедрения

Я бы делал именно так.

## День 1: снять карту процесса

Результат:

```text
docs/current-build.md
````

Там должно быть:

```text
какие репозитории нужны
какие ветки
какие команды
какие env
какие артефакты
куда они копируются
как собирается Docker
как запускается Docker
как понять, что всё работает
```

## День 2: добавить каркас

Результат:

```text
justfile
tools/pipeline.py
build.yaml
```

Команда:

```bash
just all
```

может пока падать, но уже должна проходить первые шаги.

## День 3: подключить первый внешний repo

Лучше начать с самого простого:

```text
repo, где нужно просто взять wheel
```

Потом:

```text
python repo с wheel build
```

Потом:

```text
C++ repo
```

Не наоборот. C++ почти всегда сложнее.

## День 4: подключить Docker

Когда `artifacts/` заполняется правильно, адаптировать Dockerfile:

```text
COPY artifacts/...
```

После этого:

```bash
just docker
```

должен работать.

## День 5: добавить checks

Минимум:

```text
docker image builds
python imports work
required .so files exist
basic health endpoint responds
```

## День 6+: добавить lock и hardening

После того как процесс реально работает:

```text
build.lock
sha256 артефактов
resolved commit SHA
dirty repo checks
--skip-fetch
--dry-run
--only repo
```

---

# Этап 12. Что должно получиться в итоге

Команда для полного процесса:

```bash
just all
```

Делает:

```text
1. читает build.yaml
2. создаёт .work/
3. клонирует repo_a, repo_c, ...
4. checkout нужных ref
5. обновляет submodules
6. запускает существующие build-команды
7. копирует whl/.so/bin/include/data в artifacts/
8. собирает Docker image
9. запускает smoke checks
10. пишет build.lock
```

Точечная работа:

```bash
just build repo_a
just docker
just check
```

Для нового разработчика:

```bash
just bootstrap
cp .env.example .env
just all
```

---

# Главный принцип адаптации

Не надо делать так:

```text
старый процесс плохой → переписываем всё сразу
```

Надо делать так:

```text
старый процесс работает → описываем его в build.yaml → запускаем через just → стабилизируем → постепенно чистим
```

Иначе вы получите второй параллельный процесс сборки, который “почти такой же”, но ломается в других местах.

Правильный критерий готовности первой версии:

```text
just all собирает тот же контейнер, что текущий ручной процесс
```

Критерий готовности второй версии:

```text
новый человек может собрать проект по README без знания внутренней истории сборки
```

Критерий готовности нормальной версии:

```text
по build.lock можно понять, из каких repo SHA и каких artifact sha256 собран Docker image
```
