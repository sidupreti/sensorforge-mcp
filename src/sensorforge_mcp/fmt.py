"""Compact text for the agent: the verdict line, the before -> after table, the next step. No raw JSON."""
from __future__ import annotations

import time

KIND = {"fit": "Check fit", "lossless": "Lossless", "optimize": "Optimize"}
FINAL = ("done", "failed", "stopped", "unsupported", "refused")
MAX_LINE = 400


def kb(b):
    return "–" if b is None else f"{b / 1024:.1f} KB"


def pct(x):
    return "–" if x is None else f"{x * 100:.2f}%"


def change(a, b):
    if a in (None, 0) or b is None:
        return ""
    p = (b / a - 1) * 100
    return "0%" if abs(p) < 0.5 else f"{p:+.0f}%"


def ordinal(n):
    n = int(n)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def when(ts):
    if not ts:
        return "–"
    return time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(ts))


def short_when(ts):
    return time.strftime("%m-%d %H:%M", time.localtime(ts)) if ts else "–"


def duration(s):
    if s is None:
        return "–"
    s = int(round(s))
    if s < 60:
        return f"{s} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} min {s} s" if s else f"{m} min"
    h, m = divmod(m, 60)
    return f"{h} h {m} min"


def clip(t, n=MAX_LINE):
    t = " ".join(str(t or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def failed_verdict(j):
    return j.get("status") == "done" and str(j.get("verdict") or "").startswith("FAILED")


def outcome(j):
    """pass | no_model (refused / rejected within the limits) | failed | other"""
    v = str((j.get("summary") or {}).get("verdict") or j.get("verdict") or "")
    if j.get("status") != "done":
        return "other"
    if v.startswith("PASS") or v == "runs as is" or (j.get("kind") == "fit" and not v.startswith("FAILED")):
        return "pass"
    if v.startswith(("REFUSED", "REJECTED")):
        return "no_model"
    if v.startswith("FAILED"):
        return "failed"
    return "other"


def headline(j):
    p = j.get("params") or {}
    model = j.get("model") or j.get("model_name") or "model"
    bits = [f"{KIND.get(j.get('kind'), j.get('kind'))}{' (' + j['mode'] + ')' if j.get('mode') else ''} of {model}"]
    if p.get("target"):
        bits.append(f"on {p['target']}")
    return f"Run {j.get('id')}: " + " ".join(bits)


def status_line(j):
    st = j.get("status")
    if st == "queued":
        pos = j.get("position")
        return f"Status: queued{', ' + ordinal(pos) + ' in line' if pos else ''} (submitted {duration(time.time() - (j.get('created') or time.time()))} ago)."
    if st == "uploading":
        return "Status: the upload is being stored."
    if st == "running":
        # elapsed time against the usual length only: the service's own progress lines are its working notes
        el = time.time() - (j.get("started") or time.time())
        usual = {"fit": "seconds", "lossless": "under a minute"}.get(j.get("kind")) or (
            "up to an hour" if j.get("mode") == "thorough" else "about 15 minutes")
        return f"Status: running for {duration(el)} ({KIND.get(j.get('kind'), 'this run').lower()} runs usually take {usual})."
    if st == "done":
        return f"Status: {'failed' if failed_verdict(j) else 'done'} in {duration((j.get('usage') or {}).get('wall_seconds'))}."
    if st == "failed":
        return f"Status: failed. {clip(j.get('error') or 'The run failed.')}"
    if st == "refused":
        return f"Status: not accepted. {clip(j.get('error') or 'The upload was refused.')}"
    if st == "unsupported":
        return f"Status: not supported. {clip(j.get('error') or 'This model is not supported by this run yet.')}"
    if st == "stopped":
        return "Status: stopped."
    return f"Status: {st}."


def _budget(t, key):
    v = (t or {}).get(key)
    return f" / {v:g} KB" if isinstance(v, (int, float)) else ""


def fit_rows(j, targets=None):
    tmap = {t["name"]: t for t in (targets or [])}
    rows = []
    for r in (j.get("summary") or {}).get("rows") or []:
        t = tmap.get(r.get("target"))
        rows.append([r.get("target"), clip(r.get("verdict"), 120), kb(r.get("arena_bytes")) + _budget(t, "ram_kb"),
                     kb(r.get("file_bytes")) + _budget(t, "flash_kb"), kb(r.get("lossless"))])
    return rows


FIT_HEAD = ["Chip", "Verdict", "SRAM / budget", "Flash / budget", "SRAM after lossless"]
SRAM_NOTE_64 = "SRAM is the tensor arena, measured by running the model under TensorFlow Lite Micro (a 64-bit build: a 32-bit chip needs somewhat less)."
SRAM_NOTE_RV = "SRAM measured on TensorFlow Lite Micro built for a 32-bit RISC-V core."


def result(j, targets=None, max_notes=3):
    """The figures of a finished run."""
    s = j.get("summary") or {}
    out = []
    v = str(s.get("verdict") or j.get("verdict") or "")
    if j.get("kind") == "fit":
        out.append(f"Verdict: {v or 'no result'}")
        rows = fit_rows(j, targets)
        if rows:
            out.append(table(FIT_HEAD, rows))
            rv = any(r.get("ram_build") == "rv32" for r in s.get("rows") or [])
            out.append(SRAM_NOTE_RV if rv else SRAM_NOTE_64)
    else:
        out.append(f"Verdict: {v or 'no result'}")
        b, a = s.get("before") or {}, s.get("after") or {}
        vals = [b.get("ram"), a.get("ram"), b.get("flash"), a.get("flash"), b.get("accuracy"), a.get("accuracy")]
        if any(x is not None for x in vals):
            label = "Your file" if b.get("label") in (None, "your file") else "Before"
            rows = [["SRAM (tensor arena)", kb(b.get("ram")), kb(a.get("ram")), change(b.get("ram"), a.get("ram"))],
                    ["Flash (model file)", kb(b.get("flash")), kb(a.get("flash")), change(b.get("flash"), a.get("flash"))]]
            if j.get("kind") == "optimize":
                name = {"dense": "Output fidelity", "multilabel": "Multi-label macro F1"}.get(s.get("task"), "Accuracy")
                where = ", ".join(x for x in [s.get("accuracy_runtime"),
                                              f"{int(s['accuracy_samples']):,} test samples" if s.get("accuracy_samples") else ""] if x)
                d = ""
                if b.get("accuracy") is not None and a.get("accuracy") is not None:
                    d = f"{(a['accuracy'] - b['accuracy']) * 100:+.2f} pts"
                rows.append([name + (f" ({where})" if where else ""), pct(b.get("accuracy")), pct(a.get("accuracy")), d])
            out.append(table(["", label, "After", "Change"], rows))
            out.append(SRAM_NOTE_RV if s.get("ram_build") == "rv32" else SRAM_NOTE_64)
    changes = [clip(c) for c in s.get("changes") or []]
    if changes:
        out.append("What changed:\n" + "\n".join(f"- {c}" for c in changes[:6]))
    if s.get("refusal") and (s["refusal"] or {}).get("advice"):
        out.append(clip(s["refusal"]["advice"], 600))
    advice = [clip(c) for c in s.get("advice") or []]
    if advice:
        out.append("What to do next (from the service):\n" + "\n".join(f"- {c}" for c in advice[:6]))
    warns = [clip(w, 300) for w in s.get("warnings") or []]
    if warns:
        more = f"\n- ... and {len(warns) - max_notes} more in the report (get_report)" if len(warns) > max_notes else ""
        out.append("Notes:\n" + "\n".join(f"- {w}" for w in warns[:max_notes]) + more)
    return "\n".join(out)


def next_step(j, downloaded=False):
    """One line saying what to do next."""
    k, st, o = j.get("kind"), j.get("status"), outcome(j)
    rid = j.get("id")
    if st not in FINAL:
        return f"Next: wait_for_run(\"{rid}\") until it finishes."
    if st in ("failed", "unsupported", "refused") or o == "failed":
        return "Next: read the reason above, fix the input, and submit again; get_report(\"%s\") may say more." % rid
    if st == "stopped":
        return "Next: submit the run again when you are ready."
    if k == "fit":
        return ("Next: lossless(model, chip) gives less SRAM/flash with outputs identical to yours (no data); "
                "optimize(model, data, chip) goes further within an accuracy limit.")
    if o == "no_model":
        return ("Next: no model met every limit. Loosen one (accuracy_limit, ram_kb / flash_kb, or effort='thorough') and "
                f"run optimize again; get_report(\"{rid}\") has the details.")
    if o == "pass":
        if downloaded:
            return "Next: build the .tflite (or the c/ sources, see c/INTEGRATION.md) into the firmware."
        if j.get("has_download") and j.get("may_download") is False:
            return (f"Next: downloading the model is part of the paid plans; the full report is free: get_report(\"{rid}\").")
        if j.get("has_download"):
            return f"Next: download(\"{rid}\") unpacks the model and its C sources next to your model."
    return f"Next: get_report(\"{rid}\") for the details."


def job_text(j, targets=None, downloaded=False):
    out = [headline(j), status_line(j)]
    if j.get("status") == "done":
        out.append(result(j, targets))
    out.append(next_step(j, downloaded))
    return "\n".join(x for x in out if x)
