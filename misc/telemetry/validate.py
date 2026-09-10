#!/usr/bin/env python3
"""Validate the recipe catalog, Compose argv and telemetry deployment contracts."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import yaml


def duplicate_keys(node):
    if isinstance(node, yaml.MappingNode):
        keys = [k.value for k, _ in node.value]
        assert len(keys) == len(set(keys)), 'Duplicate explicit YAML mapping key'
        for _, value in node.value:
            duplicate_keys(value)
    elif isinstance(node, yaml.SequenceNode):
        for value in node.value:
            duplicate_keys(value)


def main(root):
    shared = root / 'misc/telemetry'
    recipes = json.loads((shared/'recipes.json').read_text())
    assert len({x['path'] for x in recipes}) == len(recipes)
    assert not list(root.glob('*.yaml')), 'Compose files must live in recipe folders'
    for path in root.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    for path in root.rglob('*.yaml'):
        duplicate_keys(yaml.compose(path.read_text()))
    compose_bin = os.getenv('COMPOSE_BIN')
    prefix = [compose_bin] if compose_bin else ['docker', 'compose']
    env = os.environ.copy()
    env.pop('RESTART_POLICY', None)
    for recipe in recipes:
        directory = root / recipe['path']
        assert (directory/'README.md').is_file(), 'Missing recipe README'
        command = prefix + ['--project-directory', str(directory), '--env-file', str(directory/'.env.example'), '-f', str(directory/'docker-compose.yaml')]
        if 'gemma4-mtp' in recipe['services']:
            command += ['--profile','mtp']
        subprocess.run(command + ['config', '--quiet'], env=env, check=True, capture_output=True)
        config = json.loads(subprocess.check_output(command+['config', '--format', 'json'], env=env, stderr=subprocess.PIPE))
        services = config['services']
        for name, service in services.items():
            if name == 'download':
                continue
            assert service.get('restart') == 'unless-stopped', 'Invalid published restart policy'
            argv = service.get('command') or []
            assert isinstance(argv, list) and all(a == a.strip() and '\n' not in a for a in argv), 'Malformed argv'
            assert not any(a.startswith('--') and ' --' in a for a in argv), 'Joined flags'
        collector = services['otel-collector']
        assert int(collector['mem_limit']) <= 512*1024**2
        assert not collector.get('deploy',{}).get('resources',{}).get('reservations',{}).get('devices')
        assert any(v['target']=='/etc/otelcol/config.yaml' and Path(v['source']).is_file() for v in collector['volumes'])
        for service in (services['smg'], collector):
            assert not service.get('deploy',{}).get('resources',{}).get('reservations',{}).get('devices'), 'Telemetry acquired GPUs'
        if recipe['mode'] == 'external':
            assert not {'jaeger','open-webui','litellm'} & services.keys(), 'External recipe duplicates shared infrastructure'
            assert collector['environment']['OTLP_UPSTREAM_ENDPOINT'] != 'jaeger:4317'
        else:
            assert 'jaeger' in services
        for name in recipe['services']:
            # Compose omits disabled profiles; alternate variants are checked separately.
            if name not in services:
                continue
            service = services[name]
            argv = service.get('command') or []
            if recipe.get('transport') == 'grpc':
                assert service['environment']['ENABLE_TRACING']=='1'
            elif recipe['engine']=='sglang':
                if argv:
                    assert '--enable-trace' in argv and '--otlp-traces-endpoint' in argv
                else:
                    assert any('--enable-trace' in p.read_text() for p in directory.glob('*-entrypoint.sh'))
                assert 'SGLANG_BASE_IMAGE' in service['build']['args']
            else:
                assert '--otlp-traces-endpoint' in argv
        # A no-restart local experiment must override all long-running containers.
        experiment = json.loads(subprocess.check_output(command+['config','--format','json'], env=env|{'RESTART_POLICY':'no'}, stderr=subprocess.PIPE))
        assert all(s.get('restart')=='no' for n,s in experiment['services'].items() if n!='download')
    extra_count = 0
    for path in sorted((root/'misc').rglob('docker-compose*.yaml')):
        prefix_args = prefix + ['--project-directory', str(path.parent), '-f', str(path)]
        example = path.parent/'env.example'
        if example.exists():
            prefix_args += ['--env-file', str(example)]
        subprocess.run(prefix_args+['config','--quiet'], env=env, check=True, capture_output=True)
        config = json.loads(subprocess.check_output(prefix_args+['config','--format','json'], env=env, stderr=subprocess.PIPE))
        assert all(s.get('restart')=='unless-stopped' for s in config['services'].values())
        assert all(all(a==a.strip() and '\n' not in a for a in (s.get('command') or [])) for s in config['services'].values())
        extra_count += 1
    print('OK:',extra_count,'misc/test Compose files')
    print('OK:',len(recipes),'recipes; layout, README, YAML keys, argv, restart, local/external tracing contracts')


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[2])
