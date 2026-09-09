#!/usr/bin/env python3
"""Bounded disconnect probes; output counts/timings only, never model text."""
import argparse
import concurrent.futures
import datetime
import hashlib
import http.client
import importlib.metadata
import json
import pathlib
import socket
import time
import urllib.parse
import uuid

import grpc
from smg_grpc_proto import sglang_scheduler_pb2 as pb, sglang_scheduler_pb2_grpc as rpc
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://smg:30000')
    parser.add_argument('--worker', default='localhost:19051')
    parser.add_argument('--tokenizer', default='/model')
    args = parser.parse_args()
    stub = rpc.SglangSchedulerStub(grpc.insecure_channel(args.worker))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    url = urllib.parse.urlsplit(args.base_url)
    assert url.scheme == 'http', 'Probe is intended for the internal Docker HTTP endpoint'

    def emit(result):
        print(json.dumps(result), flush=True)

    def loads():
        reply = stub.GetLoads(pb.GetLoadsRequest(include=['all']), timeout=5)
        return {'running': sum(r.num_running_reqs for r in reply.loads),
                'waiting': sum(r.num_waiting_reqs for r in reply.loads)}

    def idle(label):
        start = time.monotonic()
        samples = []
        # Consecutive idle snapshots over 1.5 seconds avoid trusting one stale sample.
        idle_since = None
        while time.monotonic() - start < 10:
            current = loads()
            now = time.monotonic()
            samples.append({'seconds': now - start, **current})
            if not any(current.values()):
                idle_since = idle_since or now
                if now - idle_since >= 1.5:
                    emit({'test': label, 'pass': True, 'idle_observed_after_s': idle_since - start,
                          'samples': samples})
                    return
            else:
                idle_since = None
            time.sleep(.25)
        emit({'test': label, 'pass': False, 'samples': samples})
        raise RuntimeError('Scheduler did not become idle; stop further probes')

    def disconnect(chat=False, early=False):
        prompt = 'Disconnect audit ' + uuid.uuid4().hex + '. Count from 1 to 5000, one number per line. '
        if early:
            ids = tokenizer.encode(prompt + 'Each numbered record has a key, a value, and a checksum. ' * 3000)
            prompt = tokenizer.decode(ids[:24576])
        body = {'model': 'Qwen3.8-27B', 'max_tokens': 2048, 'ignore_eos': True,
                'temperature': 0, 'stream': True}
        if chat:
            body.update(messages=[{'role': 'user', 'content': prompt}],
                        chat_template_kwargs={'enable_thinking': False})
        else:
            body['prompt'] = prompt
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=60)
        response = None
        start = time.monotonic()
        chunks = 0
        try:
            conn.request('POST', '/v1/chat/completions' if chat else '/v1/completions',
                         json.dumps(body), {'Content-Type': 'application/json'})
            if early:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    active = loads()
                    if any(active.values()):
                        break
                    time.sleep(.02)
                assert any(active.values()), 'Early cancel request was never observed by scheduler'
            else:
                response = conn.getresponse()
                assert response.status == 200, response.status
                while chunks < 3:
                    line = response.readline()
                    assert line, 'Unexpected stream EOF'
                    if not line.startswith(b'data:'):
                        continue
                    raw = line[5:].strip()
                    assert raw != b'[DONE]', 'Request finished before cancellation'
                    obj = json.loads(raw)
                    assert not obj.get('error'), 'Stream returned error'
                    for choice in obj.get('choices', []):
                        delta = choice.get('delta', {})
                        if choice.get('text') or delta.get('content') or delta.get('reasoning_content'):
                            chunks += 1
                active = loads()
                assert any(active.values()), 'Generation must still be active before cancellation'
            elapsed = time.monotonic() - start
            # Force an actual TCP disconnect; no response body is drained.
            if conn.sock:
                conn.sock.shutdown(socket.SHUT_RDWR)
            return {'kind': 'prefill' if early else ('chat' if chat else 'completion'),
                    'before_disconnect': active, 'seconds_until_disconnect': elapsed,
                    'text_chunks_read': chunks, 'max_tokens': 2048}
        finally:
            if response:
                response.close()
            conn.close()

    emit({'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'versions': {n: importlib.metadata.version(n) for n in ['sglang', 'smg-grpc-servicer']},
          'request_manager_sha256': hashlib.sha256(pathlib.Path(importlib.metadata.distribution(
              'smg-grpc-servicer').locate_file('smg_grpc_servicer/sglang/request_manager.py')).read_bytes()).hexdigest()})
    idle('initial_idle')
    for name, chat, early in [('completion', False, False), ('chat', True, False), ('prefill_24k', False, True)]:
        row = disconnect(chat, early)
        emit({'test': name, 'disconnect': row})
        idle(name + '_cleanup')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _: disconnect(), range(8)))
    emit({'test': 'eight_streams', 'disconnects': rows})
    idle('eight_streams_cleanup')
    for i in range(3):
        rid = 'disconnect-probe-' + uuid.uuid4().hex
        request = pb.GenerateRequest(request_id=rid,
            tokenized=pb.TokenizedInput(input_ids=tokenizer.encode('Count from 1 to 5000, one number per line.')),
            sampling_params=pb.SamplingParams(temperature=0, top_p=1, top_k=-1, repetition_penalty=1,
                max_new_tokens=2048, n=1, ignore_eos=True, skip_special_tokens=True), stream=True)
        call = stub.Generate(request, timeout=60)
        try:
            next(call)
            active = loads()
            cancelled = call.cancel()
            emit({'test': f'direct_grpc_{i}', 'cancelled': cancelled, 'before_disconnect': active})
            idle(f'direct_grpc_{i}_cleanup')
        except BaseException:
            call.cancel()
            stub.Abort(pb.AbortRequest(request_id=rid, reason='Bounded audit cleanup'), timeout=5)
            raise
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=60)
    conn.request('POST', '/v1/completions', json.dumps({'model': 'Qwen3.8-27B',
        'prompt': 'Reply with a short greeting.', 'max_tokens': 16, 'temperature': 0, 'ignore_eos': True}),
        {'Content-Type': 'application/json'})
    response = conn.getresponse()
    assert response.status == 200
    result = json.load(response)
    conn.close()
    assert result['usage']['completion_tokens'] == 16
    emit({'test': 'subsequent_completion', 'pass': True, 'usage': result['usage']})
    idle('final_idle')


if __name__ == '__main__':
    main()
