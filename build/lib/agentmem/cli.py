"""Command-line inspector:  agentmem --db memory.db [--ns user-42] <command>"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .memory import AgentMemory
from .types import Layer


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agentmem", description="Inspect and manage agent memory")
    p.add_argument("--db", default="agentmem.db", help="SQLite file (default: agentmem.db)")
    p.add_argument("--ns", default="default", help="namespace")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=8)
    l = sub.add_parser("list"); l.add_argument("--layer", choices=[x.value for x in Layer]); l.add_argument("-n", type=int, default=20)
    sub.add_parser("sessions")
    a = sub.add_parser("add"); a.add_argument("text"); a.add_argument("--role", default="user")
    f = sub.add_parser("learn"); f.add_argument("fact"); f.add_argument("--key")
    sub.add_parser("consolidate")
    e = sub.add_parser("export"); e.add_argument("out")
    i = sub.add_parser("import"); i.add_argument("src")
    d = sub.add_parser("forget"); d.add_argument("id")
    args = p.parse_args(argv)

    mem = AgentMemory(args.db, namespace=args.ns, auto_consolidate=True)
    try:
        if args.cmd == "stats":
            print(json.dumps(mem.stats(), indent=2))
        elif args.cmd == "search":
            for h in mem.recall(args.query, k=args.k):
                print(f"{h.score:.3f}  [{h.item.layer.value:8}] imp={h.item.importance:.2f}  {h.item.content}")
        elif args.cmd == "list":
            layer = Layer(args.layer) if args.layer else None
            for it in mem.storage.list(args.ns, layer, limit=args.n):
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(it.created_at))
                print(f"{it.id[:8]}  {ts}  [{it.layer.value:8}] imp={it.importance:.2f}  {it.content}")
        elif args.cmd == "sessions":
            for s in mem.sessions():
                st = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.started_at))
                print(f"{s.id}  {st}  {'ended' if s.ended_at else 'open'}  {s.summary or ''}")
        elif args.cmd == "add":
            with mem.session():
                it = mem.add(args.text, role=args.role)
            print(f"added {it.id} (importance {it.importance:.2f})")
        elif args.cmd == "learn":
            print(mem.learn(args.fact, key=args.key).id)
        elif args.cmd == "consolidate":
            print(json.dumps(mem.consolidate().__dict__, indent=2))
        elif args.cmd == "export":
            mem.export_json(args.out); print(f"exported to {args.out}")
        elif args.cmd == "import":
            with open(args.src) as fh:
                print(f"imported {mem.import_data(json.load(fh))} memories")
        elif args.cmd == "forget":
            matches = [i for i in mem.storage.list(args.ns) if i.id.startswith(args.id)]
            if len(matches) != 1:
                print(f"{len(matches)} matches for {args.id!r}", file=sys.stderr); return 1
            mem.forget(matches[0].id); print(f"forgot {matches[0].id}")
    finally:
        mem.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
