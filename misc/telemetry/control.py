#!/usr/bin/env python3
"""Run one catalogued recipe without printing resolved secrets."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(os.getenv("RECIPE_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
CATALOG = json.loads((Path(__file__).parent / 'recipes.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', choices=[x['path'] for x in CATALOG])
    parser.add_argument('action', choices=['config', 'build', 'up', 'stop', 'down', 'logs', 'ps', 'trace-check'])
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--variant', choices=['base', 'mtp'], default='base')
    parser.add_argument('--model', help='Model ID from /v1/models; required for trace-check')
    args, extra = parser.parse_known_args()
    recipe = next(x for x in CATALOG if x['path'] == args.recipe)
    directory = ROOT / args.recipe
    envfile = args.env_file or directory / ('.env.example' if args.action == 'config' else '.env')
    if not envfile.is_file():
        parser.error('copy .env.example to .env and configure model paths, keys and OTLP endpoint')
    env = os.environ.copy()
    prefix = [env['COMPOSE_BIN']] if env.get('COMPOSE_BIN') else ['docker', 'compose']
    compose = prefix + ['--project-directory', str(directory), '--env-file', str(envfile), '-f', str(directory/'docker-compose.yaml')]
    if args.variant == 'mtp':
        if 'gemma4-mtp' not in recipe['services']:
            parser.error('this recipe has no MTP variant')
        env['SMG_WORKER_SERVICE'] = 'gemma4-mtp'
        compose += ['--profile', 'mtp']
    if args.action == 'config':
        subprocess.run(compose + ['config', '--quiet'], env=env, check=True)
        print('Compose configuration valid')
        return
    if args.action in {'up','stop','down'} and env.get('RECIPE_CONFIRM') != 'mutate-inference':
        parser.error('set RECIPE_CONFIRM=mutate-inference for runtime changes')
    if args.action == 'up':
        subprocess.run(compose + ['config', '--quiet'], env=env, check=True)
        if args.variant == 'mtp':
            selected = ['gemma4-mtp', 'smg', 'otel-collector', 'jaeger']
        elif 'gemma4-mtp' in recipe['services']:
            selected = ['gemma4', 'smg', 'otel-collector', 'jaeger']
        else:
            selected = []
        command = ['up', '-d', '--build'] + extra + selected
    elif args.action == 'trace-check':
        if not args.model:
            parser.error('--model is required for trace-check')
        # Resolve only in memory to obtain actual internal addresses/ports.
        config = json.loads(subprocess.check_output(compose + ['config', '--format', 'json'], env=env))
        smg = config['services']['smg']
        arguments = smg.get('command', [])
        port = arguments[arguments.index('--port')+1]
        host = arguments[arguments.index('--host')+1]
        base = 'http://' + ('127.0.0.1' if host=='0.0.0.0' else host) + ':' + port
        collector = config['services']['otel-collector']['environment']
        http = ('http://127.0.0.1:' + collector['OTLP_RECEIVER_HTTP_PORT'] if recipe['host_network'] else 'http://otel-collector:4318') + '/v1/traces'
        command = ['exec', '-T', '-e', 'TRACE_API_KEY', 'smg', 'python3', '/opt/telemetry/trace_probe.py', '--engine', recipe['engine'], '--model', args.model, '--base-url', base, '--otlp-http-endpoint', http]
        if recipe['mode'] == 'external' and '--jaeger-url' not in extra:
            command += ['--no-verify']
        command += extra
    else:
        command = [args.action] + extra
    subprocess.run(compose + command, env=env, check=True)


if __name__ == '__main__':
    main()
