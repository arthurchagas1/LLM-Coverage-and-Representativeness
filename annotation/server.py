#!/usr/bin/env python3
"""Annotation site for the human validation of the lineage taxonomy. Standard library only.

    .venv/bin/python annotation/server.py                 # http://127.0.0.1:18765
    .venv/bin/python annotation/server.py --port 9000
    .venv/bin/python annotation/server.py --host 0.0.0.0  # reachable from the network (use with care)

From another machine, tunnel instead of opening the port:  ssh -L 18765:localhost:18765 <server>
then open http://localhost:18765 in the browser.

Each annotator signs in with a name and sees the items in an order shuffled for that name. Every save is appended to
annotations/<name>.jsonl with a timestamp (the last record of an item wins), so nothing is ever overwritten. The site
never shows the taxonomy's own labels (data/system_labels.csv) or another annotator's answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ANN = Path(os.environ.get("ANNOTATION_DIR", HERE / "annotations"))
ANN.mkdir(parents=True, exist_ok=True)
LOCK = threading.Lock()

ITEMS = json.loads((DATA / "items.json").read_text(encoding="utf-8"))
ITEM_IDS = {it["item_id"] for it in ITEMS}
VOCAB = json.loads((DATA / "vocabulary.json").read_text(encoding="utf-8"))
GUIDE = (HERE / "GUIDELINES.md").read_text(encoding="utf-8") if (HERE / "GUIDELINES.md").exists() else ""

DECISIONS = ["in_scope", "out_of_scope", "insufficient_information"]
ARTIFACTS = ["official", "fine-tune", "merge", "mirror-reupload", "independent-pretraining", "other"]
CONFIDENCE = ["high", "medium", "low"]
NAME_RE = re.compile(r"^[a-z0-9_-]{1,32}$")


def order_for(name: str) -> list:
    """Batch 1 first, then 2, then 3, so that agreement can be computed after 60, 120, and 240 labels; the order
    inside each batch is shuffled per annotator."""
    rng = random.Random(int(hashlib.sha256(name.encode()).hexdigest(), 16) % 2 ** 32)
    out = []
    for b in sorted({it["batch"] for it in ITEMS}):
        ids = [it["item_id"] for it in ITEMS if it["batch"] == b]
        rng.shuffle(ids)
        out.extend(ids)
    return out


def load(name: str) -> dict:
    path = ANN / f"{name}.jsonl"
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["item_id"]] = rec
    return out


def clean(rec: dict) -> dict:
    """Validate one submitted annotation; raise ValueError on anything unexpected."""
    def pick(key, allowed, required=True):
        v = str(rec.get(key, "") or "").strip()
        if not v and not required:
            return ""
        if v not in allowed:
            raise ValueError(f"invalid {key}: {v!r}")
        return v

    def free(key, limit=500):
        return str(rec.get(key, "") or "").strip()[:limit]

    item_id = str(rec.get("item_id", ""))
    if item_id not in ITEM_IDS:
        raise ValueError("unknown item")
    decision = pick("decision", DECISIONS)
    inscope = decision == "in_scope"
    out = {"item_id": item_id, "decision": decision,
           "family": free("family", 80).lower(), "series": free("series", 80).lower(),
           "size_bucket": pick("size_bucket", VOCAB["size_buckets"], required=inscope),
           "size_exact": free("size_exact", 40).lower(),
           "role": pick("role", VOCAB["roles"], required=inscope),
           "artifact": pick("artifact", ARTIFACTS, required=False),
           "confidence": pick("confidence", CONFIDENCE),
           "evidence_url": free("evidence_url", 300), "notes": free("notes", 2000),
           "seconds": float(rec.get("seconds") or 0)}
    if inscope and (not out["family"] or not out["series"]):
        raise ValueError("family and series are required for in-scope items")
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console
        pass

    def send(self, code: int, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def name_from(self, q) -> str | None:
        name = (q.get("annotator") or [""])[0].strip().lower()
        return name if NAME_RE.match(name) else None

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            return self.send(200, PAGE, "text/html")
        if url.path == "/api/setup":
            name = self.name_from(q)
            if not name:
                return self.send(400, {"error": "name must be 1-32 characters: a-z, 0-9, _ or -"})
            return self.send(200, {"items": ITEMS, "order": order_for(name), "vocab": VOCAB, "guide": GUIDE,
                                   "artifacts": ARTIFACTS, "annotations": load(name)})
        return self.send(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/save":
            return self.send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            name = str(body.get("annotator", "")).strip().lower()
            if not NAME_RE.match(name):
                raise ValueError("bad annotator name")
            rec = clean(body)
        except (ValueError, json.JSONDecodeError) as exc:
            return self.send(400, {"error": str(exc)})
        rec.update(annotator=name, saved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with LOCK, (ANN / f"{name}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return self.send(200, {"ok": True, "saved_at": rec["saved_at"]})


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lineage annotation</title>
<style>
:root{--ink:#111;--muted:#555;--line:#ddd;--bg:#fafaf8;--card:#fff;--blue:#2a78d6;--green:#1a8a5a;--orange:#d9622b}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg)}
header{display:flex;align-items:center;gap:16px;padding:10px 18px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:5}
header h1{font-size:15px;margin:0;font-weight:600}.grow{flex:1}
.bar{height:6px;width:220px;background:#eee;border-radius:3px;overflow:hidden}.bar>div{height:100%;background:var(--green)}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);background:var(--card);border-radius:6px;cursor:pointer}
button.primary{background:var(--blue);color:#fff;border-color:var(--blue)}button:disabled{opacity:.5;cursor:default}
main{display:grid;grid-template-columns:200px 1fr 1fr;gap:16px;padding:16px;max-width:1500px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px}
#list{max-height:calc(100vh - 100px);overflow:auto;padding:6px}
#list div{padding:3px 8px;border-radius:5px;cursor:pointer;display:flex;justify-content:space-between;font-variant-numeric:tabular-nums}
#list div.cur{background:#e8f0fb}#list h4.batch{margin:10px 8px 4px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}#list .done{color:var(--green)}#list .todo{color:#bbb}
.repo{font-size:17px;font-weight:600;word-break:break-all}.repo a{color:var(--blue);text-decoration:none}
table{border-collapse:collapse;width:100%;margin-top:10px}td{border-top:1px solid #eee;padding:5px 6px;vertical-align:top;word-break:break-word}
td:first-child{color:var(--muted);width:150px;white-space:nowrap}
fieldset{border:0;padding:0;margin:0 0 14px}legend{font-weight:600;margin-bottom:5px}
.opts{display:flex;flex-wrap:wrap;gap:6px}.opts label{border:1px solid var(--line);border-radius:6px;padding:4px 9px;cursor:pointer;user-select:none}
.opts input{display:none}.opts label:has(input:checked){background:var(--blue);border-color:var(--blue);color:#fff}
input[type=text],textarea{width:100%;font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px}
textarea{height:60px}.hint{color:var(--muted);font-size:12px;margin-top:3px}.disabled{opacity:.35;pointer-events:none}
.row{display:flex;gap:10px;align-items:center;margin-top:8px}#msg{font-size:13px}.err{color:#c0392b}.ok{color:var(--green)}
#login{max-width:420px;margin:12vh auto}#guide{white-space:pre-wrap;font-size:13px;max-height:70vh;overflow:auto}
dialog{max-width:820px;border:1px solid var(--line);border-radius:10px}
</style></head><body>
<div id="login" class="card">
  <h2 style="margin-top:0">Lineage taxonomy annotation</h2>
  <p>Enter your annotator name (lowercase letters, digits, <code>_</code> or <code>-</code>). Use the same name every session to resume.</p>
  <input id="name" type="text" placeholder="e.g. ana" autofocus>
  <div class="row"><button class="primary" onclick="start()">Start</button><span id="loginmsg" class="err"></span></div>
</div>
<div id="app" hidden>
<header><h1>Lineage annotation</h1><span id="who"></span><div class="bar"><div id="prog"></div></div><span id="progtxt"></span>
<span class="grow"></span><button onclick="guide.showModal()">Guidelines</button><button onclick="jumpTodo()">Next unannotated</button></header>
<main>
  <div class="card" id="list"></div>
  <div class="card">
    <div class="hint" id="pos"></div>
    <div class="repo"><a id="repo" target="_blank" rel="noopener"></a> &#8599;</div>
    <div class="hint">Open the repository, its model card, <code>config.json</code> and declared base model. The table shows the raw Hub metadata only.</div>
    <table id="meta"></table>
  </div>
  <form class="card" id="form" onsubmit="return false">
    <fieldset><legend>1. Scope</legend><div class="opts" id="decision"></div>
      <div class="hint">In scope = a text-generative language model with identifiable lineage.</div></fieldset>
    <div id="hier">
    <fieldset><legend>2. Family</legend><input type="text" id="family" list="famlist" autocomplete="off" placeholder="e.g. llama, qwen, mistral">
      <datalist id="famlist"></datalist><div class="hint">Pick an existing slug when it fits; type a new one otherwise.</div></fieldset>
    <fieldset><legend>3. Series</legend><input type="text" id="series" list="serlist" autocomplete="off" placeholder="e.g. llama3.1, qwen2.5">
      <datalist id="serlist"></datalist></fieldset>
    <fieldset><legend>4. Parameter size (total)</legend><div class="opts" id="size_bucket"></div>
      <input type="text" id="size_exact" placeholder="exact size if known, e.g. 7b, 8x7b" style="margin-top:6px">
      <div class="hint">Lower bound included: 3B &rarr; 3B-7B, 7B &rarr; 7B-13B, 13B &rarr; 13B-34B, 70B &rarr; 70B-180B. Parameters, not training tokens.</div></fieldset>
    <fieldset><legend>5. Functional role</legend><div class="opts" id="role"></div></fieldset>
    <fieldset><legend>6. Artifact type</legend><div class="opts" id="artifact"></div></fieldset>
    </div>
    <fieldset><legend>7. Confidence</legend><div class="opts" id="confidence"></div></fieldset>
    <fieldset><legend>Evidence URL</legend><input type="text" id="evidence_url" placeholder="model card, config.json, official release page"></fieldset>
    <fieldset><legend>Notes</legend><textarea id="notes" placeholder="parents of merges, MoE active size, doubts..."></textarea></fieldset>
    <div class="row"><button onclick="go(-1)">&larr; Previous</button><button class="primary" onclick="save()">Save and next (Ctrl+Enter)</button>
      <span id="msg"></span></div>
  </form>
</main></div>
<dialog id="guide"><div id="guidetxt"></div><div class="row"><button onclick="guide.close()">Close</button></div></dialog>
<script>
let S={}, cur=0, t0=Date.now();
const $=id=>document.getElementById(id);
const OPT={decision:[["in_scope","In scope"],["out_of_scope","Out of scope"],["insufficient_information","Insufficient information"]],
  confidence:[["high","High"],["medium","Medium"],["low","Low"]]};
function radios(id, opts){$(id).innerHTML=opts.map(([v,l])=>`<label><input type="radio" name="${id}" value="${v}">${l}</label>`).join("");}
function val(id){const r=document.querySelector(`input[name=${id}]:checked`);return r?r.value:"";}
function setv(id,v){document.querySelectorAll(`input[name=${id}]`).forEach(r=>r.checked=(r.value===v));}
async function start(){
  const name=$("name").value.trim().toLowerCase();
  const r=await fetch("/api/setup?annotator="+encodeURIComponent(name)); const j=await r.json();
  if(!r.ok){$("loginmsg").textContent=j.error;return;}
  S={name,...j,byId:Object.fromEntries(j.items.map(i=>[i.item_id,i]))};
  localStorage.setItem("annotator",name);
  radios("decision",OPT.decision);radios("confidence",OPT.confidence);
  radios("size_bucket",S.vocab.size_buckets.map(v=>[v,v]));radios("role",S.vocab.roles.map(v=>[v,v]));
  radios("artifact",S.artifacts.map(v=>[v,v]));
  $("famlist").innerHTML=S.vocab.families.map(f=>`<option value="${f}">`).join("");
  $("guidetxt").innerHTML=`<div id="guide">${S.guide.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}</div>`;
  document.querySelectorAll("input[name=decision]").forEach(r=>r.onchange=scope);
  $("family").oninput=()=>{const s=S.vocab.series[$("family").value.trim().toLowerCase()]||[];
    $("serlist").innerHTML=s.map(x=>`<option value="${x}">`).join("");};
  $("login").hidden=true;$("app").hidden=false;$("who").textContent="annotator: "+name;
  cur=Math.max(0,S.order.findIndex(id=>!S.annotations[id])); if(cur<0)cur=0; show();
}
function scope(){$("hier").classList.toggle("disabled",val("decision")!=="in_scope");}
function list(){
  let html="",prev=0;
  S.order.forEach((id,k)=>{const b=S.byId[id].batch; if(b!==prev){html+=`<h4 class="batch">Batch ${b}</h4>`;prev=b;}
    html+=`<div class="${k===cur?"cur":""}" onclick="cur=${k};show()"><span>${k+1}. ${id}</span>
   <span class="${S.annotations[id]?"done":"todo"}">${S.annotations[id]?"&#10003;":"&bull;"}</span></div>`;});
  $("list").innerHTML=html;
  const n=Object.keys(S.annotations).length; $("prog").style.width=(100*n/S.order.length)+"%";
  const batches=[...new Set(S.items.map(i=>i.batch))].sort((a,b)=>a-b);
  const parts=batches.map(b=>{const ids=S.items.filter(i=>i.batch===b).map(i=>i.item_id);
    return `batch ${b}: ${ids.filter(i=>S.annotations[i]).length}/${ids.length}`;});
  $("progtxt").textContent=`${n}/${S.order.length} · ${parts.join(" · ")}`;
  const c=$("list").querySelector(".cur"); if(c)c.scrollIntoView({block:"nearest"});
}
function show(){
  const id=S.order[cur], it=S.byId[id], a=S.annotations[id]||{};
  $("pos").textContent=`Item ${cur+1} of ${S.order.length} · batch ${it.batch} · ${id}`;
  $("repo").textContent=it.repo_id;$("repo").href=it.url;
  $("meta").innerHTML=Object.entries(it.metadata).map(([k,v])=>`<tr><td>${k}</td><td>${(v||"–").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])).replace(/\|/g," | ")}</td></tr>`).join("");
  ["decision","size_bucket","role","artifact","confidence"].forEach(k=>setv(k,a[k]||""));
  ["family","series","size_exact","evidence_url","notes"].forEach(k=>$(k).value=a[k]||"");
  $("family").oninput();scope();$("msg").textContent=a.saved_at?"saved "+a.saved_at:"";$("msg").className="ok";
  t0=Date.now();list();window.scrollTo(0,0);
}
function go(d){cur=Math.min(S.order.length-1,Math.max(0,cur+d));show();}
function jumpTodo(){const k=S.order.findIndex(id=>!S.annotations[id]);if(k>=0){cur=k;show();}else{$("msg").textContent="All items annotated.";}}
async function save(){
  const rec={annotator:S.name,item_id:S.order[cur],decision:val("decision"),family:$("family").value,series:$("series").value,
    size_bucket:val("size_bucket"),size_exact:$("size_exact").value,role:val("role"),artifact:val("artifact"),
    confidence:val("confidence"),evidence_url:$("evidence_url").value,notes:$("notes").value,seconds:(Date.now()-t0)/1000};
  if(rec.decision!=="in_scope"){rec.family=rec.series=rec.size_bucket=rec.role=rec.artifact="";}
  const r=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(rec)});
  const j=await r.json(); if(!r.ok){$("msg").textContent=j.error;$("msg").className="err";return;}
  S.annotations[rec.item_id]={...rec,saved_at:j.saved_at}; if(cur<S.order.length-1)go(1);else{show();}
}
document.addEventListener("keydown",e=>{if(e.ctrlKey&&e.key==="Enter"&&!$("app").hidden)save();});
$("name").addEventListener("keydown",e=>{if(e.key==="Enter")start();});
const last=localStorage.getItem("annotator"); if(last)$("name").value=last;
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18765)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Annotation site: http://{args.host}:{args.port}  ({len(ITEMS)} items, saving to {ANN})")
    server.serve_forever()


if __name__ == "__main__":
    main()
