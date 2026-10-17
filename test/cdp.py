"""Minimal Chrome DevTools Protocol driver (stdlib only) for testing in a throwaway Vivaldi profile.

Usage (run with `py -3.10 -E`):
  cdp.py launch [--headed] [--ext]     start Vivaldi with a temp profile + remote debugging
  cdp.py open <url>                    open a new tab
  cdp.py eval <url-substr> <js>        evaluate JS (awaits promises) in the first tab whose URL contains url-substr
  cdp.py evalfile <url-substr> <file>  same, JS read from a file
  cdp.py evalext <url-substr> <js>     evaluate in the extension's content-script world
  cdp.py reload <url-substr>           reload ignoring cache
  cdp.py shot <url-substr> <png>       screenshot of that tab
  cdp.py tabs                          list tabs
  cdp.py close <url-substr>            close tab(s)
  cdp.py quit                          close the browser
"""
import base64, json, os, socket, struct, subprocess, sys, time, urllib.request

PORT = int(os.environ.get("CDP_PORT", "9333"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIVALDI = os.path.expandvars(r"%LOCALAPPDATA%\Vivaldi\Application\vivaldi.exe")
PROFILE = os.environ.get("CDP_PROFILE", os.path.join(os.environ.get("TEMP", "."), "nnf-vivaldi-profile"))


class WS:
    def __init__(self, url):
        assert url.startswith("ws://")
        hostport, path = url[5:].split("/", 1)
        host, port = hostport.split(":")
        self.sock = socket.create_connection((host, int(port)), timeout=60)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("handshake failed")
            buf += chunk
        head, self.rest = buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise ConnectionError(head.decode(errors="replace"))

    def _read(self, n):
        out = self.rest[:n]
        self.rest = self.rest[n:]
        while len(out) < n:
            chunk = self.sock.recv(max(65536, n - len(out)))
            if not chunk:
                raise ConnectionError("socket closed")
            out += chunk
        if len(out) > n:
            self.rest = out[n:] + self.rest
            out = out[:n]
        return out

    def send(self, text):
        data = text.encode()
        hdr = bytearray([0x81])
        n = len(data)
        if n < 126:
            hdr.append(0x80 | n)
        elif n < 65536:
            hdr.append(0x80 | 126); hdr += struct.pack(">H", n)
        else:
            hdr.append(0x80 | 127); hdr += struct.pack(">Q", n)
        mask = os.urandom(4)
        hdr += mask
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(data))
        self.sock.sendall(bytes(hdr) + masked)

    def recv(self):
        msg = b""
        while True:
            b1, b2 = self._read(2)
            op, n = b1 & 0x0F, b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            if b2 & 0x80:
                mask = self._read(4)
                payload = bytes(b ^ mask[i & 3] for i, b in enumerate(self._read(n)))
            else:
                payload = self._read(n)
            if op == 0x9:  # ping
                continue
            if op == 0x8:
                raise ConnectionError("closed by peer")
            msg += payload
            if b1 & 0x80:
                return msg.decode()


class Browser:
    def __init__(self):
        info = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=5))
        self.ws = WS(info["webSocketDebuggerUrl"])
        self.id = 0
        self.events = []

    def call(self, method, params=None, session=None, timeout=120):
        self.id += 1
        msg = {"id": self.id, "method": method, "params": params or {}}
        if session:
            msg["sessionId"] = session
        self.ws.send(json.dumps(msg))
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = json.loads(self.ws.recv())
            if r.get("id") == self.id:
                if "error" in r:
                    raise RuntimeError(f"{method}: {r['error']}")
                return r.get("result", {})
            self.events.append(r)
        raise TimeoutError(method)

    def tabs(self):
        return [t for t in self.call("Target.getTargets")["targetInfos"] if t["type"] == "page"]

    def attach(self, url_substr):
        for t in self.tabs():
            if url_substr in t["url"]:
                return self.call("Target.attachToTarget", {"targetId": t["targetId"], "flatten": True})["sessionId"]
        raise LookupError(f"no tab with url containing {url_substr!r}")

    def ext_context(self, session, name="NuruNuru Frames"):
        """Execution context id of the extension's content script (isolated world) in the main frame."""
        self.events.clear()
        main_frame = self.call("Page.getFrameTree", {}, session)["frameTree"]["frame"]["id"]
        self.call("Runtime.enable", {}, session)
        for ev in self.events:
            if ev.get("method") == "Runtime.executionContextCreated" and ev.get("sessionId") == session:
                c = ev["params"]["context"]
                aux = c.get("auxData", {})
                if c.get("name") == name and aux.get("frameId") == main_frame:
                    return c["id"]
        raise LookupError("content script context not found")

    def eval(self, session, js, timeout=120, context=None):
        params = {"expression": js, "awaitPromise": True, "returnByValue": True,
                  "userGesture": True, "timeout": timeout * 1000}
        if context is not None:
            params["contextId"] = context
        r = self.call("Runtime.evaluate", params, session, timeout + 5)
        if "exceptionDetails" in r:
            d = r["exceptionDetails"]
            raise RuntimeError(d.get("exception", {}).get("description") or d.get("text"))
        return r["result"].get("value")


def launch(headed, ext):
    args = [VIVALDI, f"--user-data-dir={PROFILE}", f"--remote-debugging-port={PORT}", "--mute-audio",
            "--no-first-run", "--no-default-browser-check", "--autoplay-policy=no-user-gesture-required",
            "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows", "--window-size=1600,1000"]
    if not headed:
        args += ["--headless=new", "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                 "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"]
    if ext:
        e = os.environ.get("CDP_EXT") or os.path.join(ROOT, "extension")
        args += [f"--load-extension={e}", f"--disable-extensions-except={e}"]
    args.append("about:blank")
    subprocess.Popen(args, creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1)
            print("ready on port", PORT)
            return
        except Exception:
            time.sleep(0.5)
    sys.exit("browser did not start")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    cmd = a[0]
    if cmd == "launch":
        return launch("--headed" in a, "--ext" in a)
    b = Browser()
    if cmd == "tabs":
        for t in b.call("Target.getTargets")["targetInfos"]:
            print(t["type"], t["targetId"][:8], t["url"][:120], "|", t["title"][:60])
    elif cmd == "open":
        print(b.call("Target.createTarget", {"url": a[1]})["targetId"])
    elif cmd in ("eval", "evalfile", "evalext"):
        js = open(a[2], encoding="utf-8").read() if cmd == "evalfile" or os.path.isfile(a[2]) else a[2]
        s = b.attach(a[1])
        ctx = b.ext_context(s) if cmd == "evalext" else None
        v = b.eval(s, js, timeout=float(os.environ.get("CDP_TIMEOUT", "120")), context=ctx)
        print(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=1))
    elif cmd == "shot":
        s = b.attach(a[1])
        data = b.call("Page.captureScreenshot", {"format": "png"}, s)["data"]
        open(a[2], "wb").write(base64.b64decode(data))
        print("saved", a[2])
    elif cmd == "activate":
        for t in b.tabs():
            if a[1] in t["url"]:
                b.call("Target.activateTarget", {"targetId": t["targetId"]})
                print("activated", t["url"][:80])
                break
    elif cmd == "reload":
        s = b.attach(a[1])
        b.call("Page.enable", {}, s)
        b.call("Page.reload", {"ignoreCache": True}, s)
        t0 = time.time()
        while time.time() - t0 < 30:
            try:
                if b.eval(s, "document.readyState") == "complete":
                    break
            except Exception:
                pass
            time.sleep(0.3)
        print("reloaded")
    elif cmd == "close":
        for t in b.tabs():
            if a[1] in t["url"]:
                b.call("Target.closeTarget", {"targetId": t["targetId"]})
    elif cmd == "quit":
        try:
            b.call("Browser.close", timeout=5)
        except Exception:
            pass


if __name__ == "__main__":
    main()
