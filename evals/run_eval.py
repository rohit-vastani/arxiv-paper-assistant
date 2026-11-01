"""Evaluation for the paper assistant. Run from the project root.

    uv run python evals/run_eval.py validate --pdf data/pdfs/1706.03762v7.pdf
    uv run python evals/run_eval.py retrieval
    uv run python evals/run_eval.py e2e --api http://localhost:8000

validate   checks that every "evidence" phrase really appears in the PDF text (do this first)
retrieval  no LLM: does retrieve() return a passage containing the evidence? Unanswerable
           questions should return nothing (NO_RELEVANT_CONTEXT)
e2e        through the API: does the final answer contain the expected facts, and do the
           unanswerable questions get a refusal without made-up facts? Also measures latency

Results are printed as a table and saved to evals/results_<mode>.json.
"""
import argparse
import json
import re
import statistics
import time
import unicodedata
import uuid
from pathlib import Path

HERE = Path(__file__).parent
REFUSAL = ("does not", "doesn't", "not cover", "not discuss", "not mention",
           "no information", "not appear", "cannot find", "can't find", "unable")


# ------------------------------------------------------------------ text matching
def _variants(text: str) -> list[str]:
    """PDF text breaks lines inside words ("ma-\\nchine") and real hyphens ("byte-\\npair").
    Check both readings of every line-end hyphen."""
    t = unicodedata.normalize("NFKC", text).lower().replace("­", "")
    joined = re.sub(r"-\s*\n\s*", "", t)       # "ma-\nchine"  -> "machine"
    kept = re.sub(r"-\s*\n\s*", "-", t)        # "byte-\npair" -> "byte-pair"
    return [re.sub(r"\s+", " ", v).strip() for v in (joined, kept)]


def contains(text: str, phrase: str) -> bool:
    phrase = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", phrase).lower()).strip()
    return any(phrase in v for v in _variants(text))


def load_questions(only_ids: list[int] | None = None):
    data = json.loads((HERE / "questions.json").read_text())
    qs = [q for q in data["questions"] if not only_ids or q["id"] in only_ids]
    return data["arxiv_id"], qs


# ------------------------------------------------------------------ reporting
def report(mode: str, rows: list[dict]) -> None:
    print(f"\n{'id':>3}  {'result':<6} {'type':<12} detail")
    for r in rows:
        kind = "answerable" if r["answerable"] else "unanswerable"
        print(f"{r['id']:>3}  {'PASS' if r['ok'] else 'FAIL':<6} {kind:<12} {r['detail']}  | {r['question'][:55]}")

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    print("\nSummary")
    if ans:
        print(f"  answerable   : {sum(r['ok'] for r in ans)}/{len(ans)} passed")
    if una:
        print(f"  unanswerable : {sum(r['ok'] for r in una)}/{len(una)} passed")
    secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
    if secs:
        print(f"  time/question: mean {statistics.mean(secs):.2f}s, median {statistics.median(secs):.2f}s, max {max(secs):.2f}s")
    (HERE / f"results_{mode}.json").write_text(json.dumps(rows, indent=2))
    print(f"  saved evals/results_{mode}.json")


# ------------------------------------------------------------------ modes
def run_validate(pdf: str) -> None:
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
    _, qs = load_questions()
    pages = [p.get_text() for p in pymupdf.open(pdf)]
    bad = 0
    for q in qs:
        if not q["answerable"]:
            continue
        found = [i + 1 for i, t in enumerate(pages) if contains(t, q["evidence"])]
        bad += not found
        print(f"{q['id']:>3}  {'OK ' if found else 'MISSING'}  pages {found or '-'}  | {q['evidence']}")
    print(f"\n{bad} evidence phrase(s) not found. Fix those in questions.json before trusting the results."
          if bad else "\nAll evidence phrases found.")


def run_retrieval(arxiv_id: str, qs: list[dict]) -> list[dict]:
    from paper_agent.services.retrieval import retrieve

    rows = []
    for q in qs:
        t0 = time.perf_counter()
        _, passages = retrieve(q["question"], arxiv_id)
        secs = time.perf_counter() - t0
        if q["answerable"]:
            rank = next((i + 1 for i, p in enumerate(passages) if contains(p["text"], q["evidence"])), None)
            ok = rank is not None
            if ok:
                detail = f"evidence at rank {rank} of {len(passages)} (page {passages[rank - 1]['page']}, score {passages[rank - 1]['score']})"
            elif not passages:
                detail = "nothing returned (all passages below the score threshold)"
            else:
                detail = f"evidence missing from {len(passages)} passages (top score {max(p['score'] for p in passages)})"
        else:
            ok = not passages
            detail = "correctly returned NO_RELEVANT_CONTEXT" if ok else f"returned {len(passages)} passages (top score {max(p['score'] for p in passages)})"
        rows.append({"id": q["id"], "question": q["question"], "answerable": q["answerable"],
                     "ok": ok, "detail": detail, "seconds": round(secs, 3)})
    return rows


def run_e2e(arxiv_id: str, qs: list[dict], api: str, delay: float) -> list[dict]:
    import httpx

    rows = []
    for q in qs:
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{api}/chat", timeout=180, json={
                "message": q["question"], "thread_id": str(uuid.uuid4()), "arxiv_id": arxiv_id})
            r.raise_for_status()
            body = r.json()
        except Exception as e:
            rows.append({"id": q["id"], "question": q["question"], "answerable": q["answerable"],
                         "ok": False, "detail": f"request failed: {e}", "seconds": None, "answer": ""})
            continue
        secs = time.perf_counter() - t0
        answer, low = body["answer"], body["answer"].lower()
        if q["answerable"]:
            missing = [k for k in q["answer_contains"] if k.lower() not in low]
            grounded = any(contains(p["text"], q["evidence"]) for p in body["passages"])
            ok = not missing
            detail = ("answer has the expected facts" if ok else f"answer missing {missing}") + \
                     (", evidence retrieved" if grounded else ", evidence NOT retrieved")
        else:
            made_up = [k for k in q.get("must_not_contain", []) if k.lower() in low]
            refused = any(p in low for p in REFUSAL)
            ok = refused and not made_up
            detail = "refused" if ok else ("made-up facts: " + str(made_up) if made_up else "did not clearly refuse (read the answer)")
        rows.append({"id": q["id"], "question": q["question"], "answerable": q["answerable"],
                     "ok": ok, "detail": detail, "seconds": round(secs, 2), "answer": answer})
        if delay:
            time.sleep(delay)
    return rows


# ------------------------------------------------------------------ cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["validate", "retrieval", "e2e"])
    ap.add_argument("--pdf", default="data/pdfs/1706.03762v7.pdf", help="validate: path to the PDF")
    ap.add_argument("--api", default="http://localhost:8000", help="e2e: API base URL")
    ap.add_argument("--delay", type=float, default=2.0, help="e2e: seconds to wait between questions (rate limits)")
    ap.add_argument("--ids", type=int, nargs="*", help="run only these question ids")
    args = ap.parse_args()

    arxiv_id, qs = load_questions(args.ids)
    if args.mode == "validate":
        run_validate(args.pdf)
    elif args.mode == "retrieval":
        report("retrieval", run_retrieval(arxiv_id, qs))
    else:
        report("e2e", run_e2e(arxiv_id, qs, args.api, args.delay))


if __name__ == "__main__":
    main()
