"""
isaacsim_client.py

Sends Python source code to a running Isaac Sim instance via the VSCode
extension TCP socket (port 8226) and returns stdout + any errors.

Protocol: send UTF-8 source, receive JSON reply:
    {"status": "ok"|"error", "output": "...", "ename": "...", "evalue": "...", "traceback": [...]}

Usage (as a library):
    from isaacsim_client import run_in_isaac

    output = run_in_isaac("print(1 + 1)")   # returns "2"

Usage (as a script):
    python3 isaacsim_client.py "print('hello from isaac')"
    python3 isaacsim_client.py < some_script.py
"""

import json
import socket
import sys

ISAAC_HOST = "127.0.0.1"
ISAAC_PORT = 8226
TIMEOUT    = 30.0   # seconds; increase for heavy operations


def run_in_isaac(code: str, host: str = ISAAC_HOST, port: int = ISAAC_PORT,
                 timeout: float = TIMEOUT) -> str:
    """
    Send Python `code` to the running Isaac Sim VSCode extension.
    Returns stdout output on success.
    Raises RuntimeError on execution error (includes traceback in message).
    Raises ConnectionRefusedError / TimeoutError on connection problems.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect((host, port))
        s.sendall(code.encode("utf-8"))
        # Don't shutdown write — server responds asynchronously then closes.
        # Just read until the server closes the connection.
        chunks = []
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)

    raw = b"".join(chunks).decode("utf-8", errors="replace")
    reply = json.loads(raw)

    if reply.get("status") == "error":
        tb = "\n".join(reply.get("traceback", []))
        raise RuntimeError(
            f"{reply.get('ename')}: {reply.get('evalue')}\n{tb}"
        )

    return reply.get("output", "")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        code = sys.argv[1]
    else:
        code = sys.stdin.read()

    if not code.strip():
        print("Usage: python3 isaacsim_client.py \"<python code>\"")
        print("       python3 isaacsim_client.py < script.py")
        sys.exit(1)

    try:
        output = run_in_isaac(code)
        if output:
            print(output)
    except RuntimeError as e:
        print(f"Isaac Sim error:\n{e}", file=sys.stderr)
        sys.exit(1)
    except ConnectionRefusedError:
        print(f"ERROR: Cannot connect to Isaac Sim at {ISAAC_HOST}:{ISAAC_PORT}")
        print("Make sure Isaac Sim is running with the VSCode extension enabled.")
        sys.exit(1)
    except TimeoutError:
        print(f"ERROR: Isaac Sim did not respond within {TIMEOUT}s")
        sys.exit(1)
