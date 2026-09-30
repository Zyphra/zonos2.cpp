#!/usr/bin/env python3
"""Exercise the Windows launcher's downloads with real curl and a flaky local server.

Run after preparing dist/start-zonos2.ps1:
    python tests/windows-download.py --powershell powershell.exe
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


CLIENT = r"""
param([string]$Launcher, [string]$ModelDir, [string]$BaseUrl,
      [string]$Connections, [string]$Name, [switch]$ExpectFailure)
$ErrorActionPreference = 'Stop'
$env:ZONOS2_DL_CONNECTIONS = $Connections
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Launcher, [ref]$null, [ref]$errors)
if ($errors) { throw 'Launcher parse failed' }
# Load the actual download functions without launching the server or fetching models.
$functions = $ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    @('Download-Segmented', 'Download-One') -contains $node.Name
}, $false)
if ($functions.Count -ne 2) { throw 'Download functions not found' }
foreach ($function in $functions) { Invoke-Expression $function.Extent.Text }
function Get-SizeGB([string]$name) { return '?' }
$ok = Download-One $Name
if ($ExpectFailure) {
    if ($ok) { throw 'Expected the download to fail' }
} elseif (-not $ok) {
    throw 'Expected the download to succeed'
}
"""

PAYLOAD = bytes(range(256)) * 512
requests = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def record(self):
        entries = requests.setdefault(self.path, [])
        entries.append((self.command, self.headers.get('Range')))
        return sum(method == 'GET' for method, _ in entries)

    def do_HEAD(self):
        self.record()
        self.send_response(200)
        self.send_header('Content-Length', str(len(PAYLOAD)))
        self.end_headers()

    def do_GET(self):
        count = self.record()
        name = self.path.rsplit('/', 1)[-1]
        if name == 'missing.bin' or (name == 'transient.bin' and count == 1):
            self.send_response(404 if name == 'missing.bin' else 503)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return

        requested_range = self.headers.get('Range')
        start = int(requested_range.split('=')[1].split('-')[0]) if requested_range else 0
        if name == 'no-range.bin':
            start = 0  # Deliberately ignore Range, forcing a fresh download.
        self.send_response(206 if start else 200)
        if start:
            self.send_header('Content-Range', f'bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}')
        body = PAYLOAD[start:]
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        interrupted = name in ('interrupted.bin', 'no-range.bin') and count == 1
        interrupted |= name == 'recover.bin' and count <= 4
        self.wfile.write(body[:2048] if interrupted else body)
        self.wfile.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--powershell', default='powershell.exe')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix='zonos2 download tests ') as temp:
            work = Path(temp)
            client = work / 'client.ps1'
            client.write_bytes(CLIENT.encode('utf-8-sig'))
            env = os.environ.copy()
            # On Unix, expose the real curl under the executable name used on Windows.
            if os.name != 'nt':
                curl = shutil.which('curl')
                if not curl:
                    raise RuntimeError('curl is required')
                (work / 'curl.exe').symlink_to(curl)
                env['PATH'] = str(work) + os.pathsep + env['PATH']

            for variant in ('source', 'packaged'):
                launcher = repo / ('scripts' if variant == 'source' else 'dist') / 'start-zonos2.ps1'
                models = work / variant
                models.mkdir()
                base = f'http://127.0.0.1:{server.server_port}/{variant}'

                def download(name, connections='', failure=False):
                    command = [args.powershell, '-NoProfile', '-NonInteractive',
                               '-ExecutionPolicy', 'Bypass', '-File', str(client),
                               '-Launcher', str(launcher), '-ModelDir', str(models),
                               '-BaseUrl', base, '-Connections', connections, '-Name', name]
                    if failure:
                        command.append('-ExpectFailure')
                    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
                    if result.returncode:
                        raise AssertionError(result.stdout + result.stderr)
                    path = models / name
                    if failure:
                        assert not path.exists(), 'Failed download was published as complete'
                    else:
                        assert path.read_bytes() == PAYLOAD, 'Download was corrupted'
                        assert not path.with_name(name + '.part').exists(), 'Successful download left a partial file'
                    return requests.get(f'/{variant}/{name}', [])

                for connections in ('', '1', 'invalid'):
                    entries = download('good.bin', connections)
                    assert entries[-1] == ('GET', None)
                    assert all(method != 'HEAD' for method, _ in entries), 'Default must disable segmentation'
                print(f'PASS: {variant} defaults to one connection; explicit 1 and invalid override are safe', flush=True)

                entries = download('opt-in.bin', '2')
                assert entries[0][0] == 'HEAD', 'Parallel download override was ignored'

                entries = download('interrupted.bin')
                assert entries == [('GET', None), ('GET', 'bytes=2048-')], entries
                print(f'PASS: {variant} interrupted transfer automatically resumes byte-identically', flush=True)

                entries = download('no-range.bin')
                assert entries == [('GET', None), ('GET', 'bytes=2048-'), ('GET', None)], entries
                print(f'PASS: {variant} refused range restarts cleanly', flush=True)

                entries = download('recover.bin', failure=True)
                assert len(entries) == 4, 'Interrupted transfers must stop after three retries'
                assert (models / 'recover.bin.part').stat().st_size == 8192
                entries = download('recover.bin')
                assert entries[-1] == ('GET', 'bytes=8192-')
                print(f'PASS: {variant} retries are bounded and a later run resumes the partial file', flush=True)

                entries = download('transient.bin')
                assert len(entries) == 2, 'Temporary HTTP error was not retried'
                entries = download('missing.bin', failure=True)
                assert len(entries) == 1, 'Permanent HTTP error should fail without additional retries'
                print(f'PASS: {variant} temporary HTTP errors retry; missing files fail promptly', flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == '__main__':
    main()
