#!/usr/bin/env python3
"""Check the diagnostic ID on the HTTP response, independently of JSON IDs and SSE frames.

Run against an isolated server with FLM_TEST_SERVER_LOG set to the file capturing its stdout. Set
FLM_TEST_LOG_CONTENT to the server's FLM_LOG_CONTENT value (0 or 1) for the concurrent log
case. The wire checks still run without log access; FLM_TEST_NO_ENGINE skips only real chat.
"""

from __future__ import annotations

import http.client
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from urllib.parse import urlsplit

import flm_api
from flm_api import Api, Skip, case, expect
from test_http_safety import FOREIGN_ORIGIN, LOCAL_ORIGIN, cors_mode
from test_log_privacy import declared, log_since, log_size

CHAT = '/v1/chat/completions'
INBOUND = {'X-Request-Id': 'caller-header-marker', 'X-Client-Request-Id': 'client-header-marker',
           'Authorization': 'Bearer auth-marker', 'Content-Type': 'application/json'}


def connect(api):
    u = urlsplit(api.url)
    cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
    return cls(u.hostname, u.port, timeout=api.timeout)


def request(api, method, path, body=None, headers=None):
    with closing(connect(api)) as conn:
        conn.request(method, path, body, {**INBOUND, **(headers or {})})
        resp = conn.getresponse()
        return resp.status, resp.headers, resp.read()


def response_id(headers):
    values = headers.get_all('x-request-id') or []
    expect(len(values) == 1, f'expected exactly one x-request-id, got {values!r}')
    value = values[0]
    expect(bool(re.fullmatch(r'[!-~]{1,128}', value)) and ',' not in value, f'not one bounded opaque ID: {value!r}')
    expect(value not in INBOUND.values() and value != 'cancel-marker', 'server echoed a caller ID')
    return value


@case
def normalized_scope_and_errors(api: Api) -> None:
    """Early and routed HTTP errors get IDs; a near-prefix or an Ollama route does not."""
    log = os.getenv('FLM_TEST_SERVER_LOG')
    offset = log_size(log) if log else 0
    seen = []
    for method, path, route, code, body, headers in [
        ('GET', '/v1/models/?q=1', '/v1/models', 200, None, {}), ('GET', '/v1', 'unmatched', 404, None, {}),
        ('GET', '/v1/systemone/missing', 'unmatched', 404, None, {}), ('POST', CHAT, CHAT, 400, b'{', {}),
        # Valid JSON reaches the embeddings handler; its callback stages a new error response.
        ('POST', '/v1/embeddings', '/v1/embeddings', 400, b'{}', {}),
        ('POST', CHAT, CHAT, 413, None, {'Content-Length': str(300 * 1024 * 1024)}),
    ]:
        status, hdr, payload = request(api, method, path, body, headers)
        expect(status == code, f'{method} {path}: expected {code}, got {status}')
        if path == '/v1/embeddings':
            expect(json.loads(payload)['error']['param'] == 'model', 'expected the embeddings validation error')
        seen.append((response_id(hdr), method, route, code))
    expect(len({r[0] for r in seen}) == len(seen), 'HTTP attempts reused an ID')
    for path in ('/api/version', '/v10/models', '/v1-other'):
        _, hdr, _ = request(api, 'GET', path, headers={'Origin': LOCAL_ORIGIN})
        expect(not hdr.get_all('x-request-id'), f'{path}: ID outside /v1')
        expect('x-request-id' not in exposed(hdr), f'{path}: ID exposed')
    if log:
        check_records(log, offset, seen)


@case
def cors_and_preflight_reuse(api: Api) -> None:
    """The same socket can carry two attempts, not one connection-scoped ID."""
    mode = cors_mode(api)
    log = os.getenv('FLM_TEST_SERVER_LOG')
    offset = log_size(log) if log else 0
    origin = {'Origin': LOCAL_ORIGIN}
    with closing(connect(api)) as conn:
        conn.request('OPTIONS', '/v1/models', headers={**origin, 'Access-Control-Request-Method': 'GET',
                                                      'Access-Control-Request-Headers': 'x-example'})
        first = conn.getresponse()
        first_id = response_id(first.headers)
        first.read()
        if mode == 'off':
            expect(not any(k.lower().startswith('access-control-') for k in first.headers), 'CORS off changed')
            conn.close()  # without preflight handling OPTIONS is an ordinary, socket-closing response
        else:
            expect(first.status == 200 and first.getheader('Access-Control-Allow-Headers') == 'x-example',
                   'preflight permissions changed')
            expect('x-request-id' not in exposed(first.headers), 'expose the ID only on actual responses')
        sock = conn.sock
        conn.request('GET', '/v1/models', headers=origin)
        if mode != 'off':
            expect(sock is not None and conn.sock is sock, 'preflight did not reuse the TCP connection')
        second = conn.getresponse()
        second_id = response_id(second.headers)
        expect(second.status == 200 and second_id != first_id, 'GET did not get its own ID')
        expect(('x-request-id' in exposed(second.headers)) == (mode != 'off'), 'wrong header exposure on GET')
        if mode == 'off':
            expect(not any(k.lower().startswith('access-control-') for k in second.headers), 'CORS off changed')
        second.read()
    seen = [(first_id, 'OPTIONS', 'unmatched', first.status), (second_id, 'GET', '/v1/models', 200)]
    if mode == 'local':
        status, hdr, _ = request(api, 'GET', '/v1/models', headers={'Origin': FOREIGN_ORIGIN})
        expect(status == 403 and 'x-request-id' not in exposed(hdr), 'denied origin gained permission')
        seen.append((response_id(hdr), 'GET', '/v1/models', 403))
    if log:
        check_records(log, offset, seen)


def exposed(headers):
    return {v.strip().lower() for line in headers.get_all('Access-Control-Expose-Headers', []) for v in line.split(',')}


@case
def retry_and_stream(api: Api) -> None:
    if os.getenv('FLM_TEST_NO_ENGINE') == '1':
        raise Skip('needs a model for real chat and SSE')
    mode = cors_mode(api)
    body = api.body([{'role': 'user', 'content': 'Say hello.'}], max_tokens=8, request_id='cancel-marker')
    log = os.getenv('FLM_TEST_SERVER_LOG')
    offset = log_size(log) if log else 0
    seen = []
    for stream in (False, False, True):
        with closing(connect(api)) as conn:
            headers = {**INBOUND, 'Origin': LOCAL_ORIGIN} if stream else INBOUND
            conn.request('POST', CHAT, json.dumps(dict(body, stream=stream)).encode(), headers)
            reply = conn.getresponse()
            expect(reply.status == 200, f'chat status {reply.status}')
            rid = response_id(reply.headers)  # before reading any body or SSE event
            seen.append((rid, 'POST', CHAT, reply.status))
            if stream:
                expect(reply.getheader('Content-Type', '').startswith('text/event-stream'), 'expected SSE')
                if mode == 'off':
                    expect(not any(k.lower().startswith('access-control-') for k in reply.headers), 'CORS off on SSE')
                else:
                    expect(reply.getheader('Access-Control-Allow-Origin') in (LOCAL_ORIGIN, '*')
                           and 'x-request-id' in exposed(reply.headers), 'browser cannot read the SSE response ID')
                frames = [line[5:].strip() for line in reply if line.startswith(b'data:')]
                expect(len(frames) > 1 and frames[-1] == b'[DONE]', 'stream did not finish normally')
                objects = [json.loads(frame) for frame in frames[:-1]]
            else:
                objects = [json.load(reply)]
            expect(all(obj['id'] != rid and obj.get('choices') is not None for obj in objects),
                   'HTTP ID confused with a completion/chunk ID')
    if mode == 'local':
        status, hdr, _ = request(api, 'POST', CHAT, json.dumps(dict(body, stream=True)).encode(),
                                 {'Origin': FOREIGN_ORIGIN})
        expect(status == 403 and not hdr.get('Access-Control-Allow-Origin') and 'x-request-id' not in exposed(hdr),
               'denied stream request must have no CORS permission or ID exposure')
        seen.append((response_id(hdr), 'POST', CHAT, status))
    expect(len({r[0] for r in seen}) == len(seen), 'repeated chat/stream attempts shared an ID')
    if log:
        check_records(log, offset, seen)


@case
def log_join_under_concurrency(api: Api) -> None:
    """FLM_TEST_SERVER_LOG names the isolated server log; run in both content logging modes."""
    log, _ = declared()
    offset = log_size(log)
    cases = [('GET', '/v1/models?query-marker', '/v1/models', 200),
             ('PROBEMARKER', '/v1/private-path-marker', 'unmatched', 404)] * 6

    def submit(item):
        method, path, route, code = item
        status, hdr, _ = request(api, method, path, b'{"request_id":"cancel-marker","text":"prompt-marker"}',
                                 {'Origin': LOCAL_ORIGIN})
        expect(status == code, f'{method} status {status}')
        return (response_id(hdr), method if method == 'GET' else 'UNKNOWN', route, str(code))

    with ThreadPoolExecutor(max_workers=4) as pool:
        attempts = list(pool.map(submit, cases))
    expect(len({a[0] for a in attempts}) == len(attempts), 'concurrent attempts shared an ID')
    for path in ('/api/version', '/v10/marker'):
        request(api, 'GET', path)
    check_records(log, offset, attempts)


def check_records(log, offset, attempts):
    # Match *entire* lines, not a substring among unrelated or interleaved output. Equality of
    # multisets checks uniqueness, metadata, no extra lines from controls and no leaked fields.
    marker = 'flm.http_response'
    actual = Counter(line for line in log_since(log, offset).splitlines() if marker in line)
    expected = Counter(f'{marker} id={rid} method={method} route={route} status={code}'
                       for rid, method, route, code in attempts)
    expect(actual == expected, f'correlation log mismatch: extra {actual - expected}; missing {expected - actual}')


CASES = flm_api.collect(globals())

if __name__ == '__main__':
    flm_api.main(CASES, __doc__.splitlines()[0])
