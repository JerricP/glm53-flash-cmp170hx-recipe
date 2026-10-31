#!/usr/bin/env python3
"""Greedy correctness check: 20 fixed prompts, temperature 0, one at a time, cold prefix cache (unique cache_salt).

  greedy_check.py run OUT.json            # record a run
  greedy_check.py compare BASE.json NEW.json

compare reports, per prompt, whether the token streams match; if not, the first diverging position, the baseline's
top-1/top-2 logprob margin there (a near-tie is a small margin), and both tokens.
"""
import json, sys, time, uuid, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "GLM5.3-Flash"
MAX_TOKENS = 256
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_CODE = "".join(
    f"def handler_{i}(req):\n    user = req.get('user_{i}')\n    if not user:\n        return {{'status': 400}}\n"
    f"    total = sum(x * {i % 7 + 1} for x in req.get('items', []))\n    return {{'status': 200, 'total': total}}\n\n"
    for i in range(110))
_DOC = " ".join(f"Clause {i}: the tenant shall maintain item {i} in good repair and report defects within {i % 9 + 2} days."
                for i in range(220))

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain the difference between a process and a thread in three sentences.",
    "What is 17 * 23? Show the steps.",
    "Write a bash one-liner that counts lines in all .py files under the current directory.",
    "Translate to French: 'The server restarts every night at two o'clock.'",
    "Write a SQL query that returns the five customers with the highest total order value.",
    "Give a regular expression that matches an IPv4 address and explain each part.",
    "Write a Rust function that reverses a vector of i32 in place.",
    "List the planets of the solar system in order from the Sun.",
    "Write a JavaScript debounce function with a configurable delay.",
    "Explain what a mutex is to a beginner programmer.",
    "A train travels 180 km in 2.5 hours. What is its average speed in km/h?",
    "Write a C function that computes the length of a null-terminated string without strlen.",
    "Summarise the CAP theorem in one paragraph.",
    "Write a Python class for a stack with push, pop and peek, with type hints.",
    "What are the main differences between TCP and UDP?",
    "Write a haiku about compilers.",
    "Write a Go function that checks whether a string is a palindrome, ignoring case.",
    _CODE + "\nWhich handler multiplies by 5? List the first three by name and explain the pattern.",
    _DOC + "\n\nWithin how many days must defects for item 100 be reported? Answer briefly and explain.",
]


def ask(prompt, salt):
    body = {"model": MODEL, "temperature": 0, "max_tokens": MAX_TOKENS, "logprobs": True, "top_logprobs": 2,
            "cache_salt": salt, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(DIRECT.open(req, timeout=600))
    c = r["choices"][0]
    toks = [(t["token"], t["logprob"], [tl["logprob"] for tl in t["top_logprobs"]]) for t in c["logprobs"]["content"]]
    return {"tokens": toks, "finish": c["finish_reason"], "prompt_tokens": r["usage"]["prompt_tokens"]}


def _background(stop):
    # GC_BG=1: one other user generating the whole time, so every check prompt shares a batch (calibration only)
    while not stop.is_set():
        body = {"model": MODEL, "max_tokens": 2000, "temperature": 0.7,
                "messages": [{"role": "user", "content": f"[{uuid.uuid4()}] Write a long story about a lighthouse."}]}
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        DIRECT.open(req, timeout=600).read()


def run(out):
    import os, threading
    salt = f"gc-{uuid.uuid4()}"
    res = []
    stop = threading.Event()
    if os.environ.get("GC_BG") == "1":
        threading.Thread(target=_background, args=(stop,), daemon=True).start(); time.sleep(2)
    t0 = time.time()
    for i, p in enumerate(PROMPTS):
        res.append(ask(p, f"{salt}-{i}"))
    stop.set()
    json.dump({"time": time.strftime("%F %T"), "results": res}, open(out, "w"))
    print(f"recorded {len(res)} prompts in {time.time() - t0:.0f}s -> {out}")


def compare(a, b):
    A = json.load(open(a))["results"]; B = json.load(open(b))["results"]
    same = 0
    for i, (x, y) in enumerate(zip(A, B)):
        tx = [t[0] for t in x["tokens"]]; ty = [t[0] for t in y["tokens"]]
        if tx == ty:
            same += 1; print(f"P{i:02d} MATCH ({len(tx)} tokens)"); continue
        k = next((j for j in range(min(len(tx), len(ty))) if tx[j] != ty[j]), min(len(tx), len(ty)))
        if k < len(x["tokens"]):
            top = x["tokens"][k][2]
            margin = (top[0] - top[1]) if len(top) > 1 else float("nan")
        else:
            margin = float("nan")
        print(f"P{i:02d} DIVERGE at token {k}/{len(tx)}: base margin top1-top2 = {margin:.4f} nats; "
              f"base {tx[k] if k < len(tx) else '<end>'!r} vs new {ty[k] if k < len(ty) else '<end>'!r}")
    print(f"{same}/{len(A)} identical")


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "compare": lambda: compare(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
