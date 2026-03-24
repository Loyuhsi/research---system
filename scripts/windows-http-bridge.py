#!/usr/bin/env python3
import argparse
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "accept-encoding",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge WSL localhost HTTP requests to Windows localhost via curl.exe.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, required=True)
    parser.add_argument("--name", default="bridge")
    return parser.parse_args()


def parse_http_response(raw_response: bytes):
    if b"\r\n\r\n" in raw_response:
        raw_headers, _, body = raw_response.partition(b"\r\n\r\n")
        lines = [line for line in raw_headers.decode("iso-8859-1", errors="replace").split("\r\n") if line]
    elif b"\n\n" in raw_response:
        raw_headers, _, body = raw_response.partition(b"\n\n")
        lines = [line for line in raw_headers.decode("iso-8859-1", errors="replace").split("\n") if line]
    else:
        return 502, "Bad Gateway", [], raw_response

    if not lines or not lines[0].startswith("HTTP/"):
        return 502, "Bad Gateway", [], body

    parts = lines[0].split(" ", 2)
    status_code = int(parts[1]) if len(parts) > 1 else 502
    reason = parts[2] if len(parts) > 2 else ""
    headers = []

    for line in lines[1:]:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers.append((name.strip(), value.lstrip()))

    return status_code, reason, headers, body


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "WSLWindowsBridge/0.1"

    def do_GET(self):
        self._forward()

    def do_POST(self):
        self._forward()

    def do_PUT(self):
        self._forward()

    def do_PATCH(self):
        self._forward()

    def do_DELETE(self):
        self._forward()

    def do_OPTIONS(self):
        self._forward()

    def do_HEAD(self):
        self._forward()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def _forward(self):
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length) if content_length else b""
        target_url = f"http://{self.server.target_host}:{self.server.target_port}{self.path}"

        cmd = ["curl.exe", "-sS", "-i", "-X", self.command, target_url]

        for name, value in self.headers.items():
            if name.lower() in HOP_BY_HOP_HEADERS:
                continue
            cmd.extend(["-H", f"{name}: {value}"])

        if body:
            cmd.extend(["--data-binary", "@-"])

        result = subprocess.run(cmd, input=body if body else None, capture_output=True, check=False)

        if result.returncode != 0 and not result.stdout:
            message = result.stderr or result.stdout or b"curl.exe bridge failed"
            self.send_response(502, "Bad Gateway")
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(message)))
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(message)
                except BrokenPipeError:
                    pass
            return

        status_code, reason, headers, response_body = parse_http_response(result.stdout)
        self.send_response(status_code, reason)

        for name, value in headers:
            if name.lower() in HOP_BY_HOP_HEADERS:
                continue
            self.send_header(name, value)

        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()

        if self.command != "HEAD":
            try:
                self.wfile.write(response_body)
            except BrokenPipeError:
                pass


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class, target_host: str, target_port: int):
        super().__init__(server_address, handler_class)
        self.target_host = target_host
        self.target_port = target_port


def main():
    import signal

    args = parse_args()
    server = BridgeServer((args.listen_host, args.listen_port), BridgeHandler, args.target_host, args.target_port)
    print(
        f"[bridge] {args.name} listening on http://{args.listen_host}:{args.listen_port} "
        f"-> http://{args.target_host}:{args.target_port}",
        flush=True,
    )

    def _shutdown(signum, frame):
        print(f"\n[bridge] received signal {signum}, shutting down...", file=sys.stderr, flush=True)
        server.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
