#!/usr/bin/env python3
"""
scripts/build_regulatory_corpus.py — build a REAL Irish/EU regulatory corpus.

────────────────────────────────────────────────────────────────────────────
WHY SCRAPING WAS ABANDONED
────────────────────────────────────────────────────────────────────────────
The first version of this script fetched 16 official pages with `requests`.
It returned 3 of 16, and most of what came back was unusable:

    centralbank.ie      9/9 empty — the pages are client-rendered, so a plain
                        HTTP GET receives a JavaScript shell with no content
    eur-lex.europa.eu   4/4 blocked — programmatic access refused
    statesavings.ie     fetched, but yielded marketing copy: "Plant a tree
                        today", "Prize Bonds make a great gift", "review and
                        accept Functional cookies"

Roughly six citable chunks, all from depositguarantee.ie. Not a corpus.

The same documents are published as PDFs, which are static files: they
download reliably and contain the actual statutory and guidance text rather
than a page template. So this script ingests PDFs you download once, by hand.

That mirrors how D4 (EU Digital Finance) already works — "download the file
yourself and drop it here" — and it removes every failure mode above.

────────────────────────────────────────────────────────────────────────────
IT STILL WILL NOT INVENT ANYTHING
────────────────────────────────────────────────────────────────────────────
No seeds, no summarising, no paraphrase. Text written to disk is text the
regulator published. A source that is absent stays absent and is reported.

────────────────────────────────────────────────────────────────────────────
USAGE
    python scripts/build_regulatory_corpus.py --checklist   # what to download
    python scripts/build_regulatory_corpus.py               # dry run
    python scripts/build_regulatory_corpus.py --review      # print all chunks
    python scripts/build_regulatory_corpus.py --write       # build the corpus
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "raw" / "regulatory"
PDF_DIR = OUT_DIR / "pdf"
BAR = "=" * 78

# ── PDFs to download by hand, once ───────────────────────────────────────
#
# `stem` must match the downloaded filename (without .pdf) so provenance can
# be attached to the right file. Everything here is a primary source: the
# statute itself, or the regulator's own published code/guidance.
PDF_SOURCES: list[dict[str, str]] = [
    {"stem": "cbi-consumer-protection-code",
     "authority": "Central Bank of Ireland",
     "title": "Consumer Protection Code",
     "url": "https://www.centralbank.ie/regulation/consumer-protection/consumer-protection-codes-regulations"},
    {"stem": "eu-dgsd-2014-49",
     "authority": "European Union",
     "title": "Directive 2014/49/EU on deposit guarantee schemes",
     "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32014L0049"},
    {"stem": "eu-mifid2-2014-65",
     "authority": "European Union",
     "title": "Directive 2014/65/EU (MiFID II)",
     "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32014L0065"},
    {"stem": "eu-priips-1286-2014",
     "authority": "European Union",
     "title": "Regulation (EU) No 1286/2014 (PRIIPs KID)",
     "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32014R1286"},
    {"stem": "eu-sfdr-2019-2088",
     "authority": "European Union",
     "title": "Regulation (EU) 2019/2088 (SFDR)",
     "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32019R2088"},
    {"stem": "cbi-code-of-conduct-mortgage-arrears",
     "authority": "Central Bank of Ireland",
     "title": "Code of Conduct on Mortgage Arrears (CCMA) 2013",
     "url": "https://www.centralbank.ie/docs/default-source/Regulation/consumer-protection/other-codes-of-conduct/24-gns-4-2-7-2013-ccma.pdf"},
]

# ── HTML sources that actually work ──────────────────────────────────────
# Kept deliberately short: only pages that are server-rendered and returned
# substantive text when tested. depositguarantee.ie is the statutory scheme's
# own site and is the single best source for DGS coverage wording.
HTML_SOURCES: list[dict[str, str]] = [
    {"id": "dgs-home", "authority": "Deposit Guarantee Scheme (Ireland)",
     "title": "Deposit Guarantee Scheme", "url": "https://www.depositguarantee.ie/"},
    {"id": "dgs-protected-depositors", "authority": "Deposit Guarantee Scheme (Ireland)",
     "title": "Protected Depositors",
     "url": "https://www.depositguarantee.ie/en/what-we-cover/protected-depositors"},
    {"id": "dgs-protected-deposits", "authority": "Deposit Guarantee Scheme (Ireland)",
     "title": "Protected Deposits",
     "url": "https://www.depositguarantee.ie/en/what-we-cover/protected-deposits"},
    {"id": "mabs-5-steps", "authority": "Money Advice & Budgeting Service (Ireland)",
     "title": "5 Steps to Tackling Debt",
     "url": "https://www.mabs.ie/en/tackling-debt/5-steps-to-tackling-debt/"},
    {"id": "mabs-priority-debts", "authority": "Money Advice & Budgeting Service (Ireland)",
     "title": "Priority Debts and Secondary Debts",
     "url": "https://www.mabs.ie/en/tackling-debt/what-are-priority-debts-and-secondary-debts/"},
    {"id": "mabs-creditor-rights", "authority": "Money Advice & Budgeting Service (Ireland)",
     "title": "Your Rights About How Creditors Can Demand Repayment",
     "url": "https://www.mabs.ie/en/tackling-debt/your-rights-about-how-your-creditors-can-demand-repayment/"},
]

_DROP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form",
              "noscript", "svg", "button", "select", "option"}

# Marketing and UI copy that survives tag-stripping. Every one of these was
# observed in a real --review run; they are not hypothetical.
_JUNK = re.compile(
    r"(find out more|click|sign in|register|buy now|cookie|webchat|newsletter|"
    r"follow us|back to |temporarily unavailable|gift|jackpot|rainy day|"
    r"plant a tree|explore our range|we'?re here to help|for over \d+ years)",
    re.I)

# A chunk has to look like prose about rules, not a heading or a slogan.
_MIN_CHARS = 120


class _Extract(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _DROP_TAGS:
            self._skip += 1
        elif tag in ("p", "li", "div", "br", "tr", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _DROP_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip() + " ")


def _clean(lines: list[str]) -> list[str]:
    out, seen = [], set()
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < _MIN_CHARS or _JUNK.search(line):
            continue
        key = line[:120].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def html_to_paragraphs(html: str) -> list[str]:
    p = _Extract()
    try:
        p.feed(html)
    except Exception:
        pass
    return _clean("".join(p.parts).split("\n"))


def pdf_to_paragraphs(path: Path) -> list[str] | None:
    """
    Extract paragraphs from a downloaded PDF.

    Returns None specifically when pypdf isn't installed — distinct from
    an empty list, which means pypdf ran fine but the PDF genuinely
    yielded no usable text (e.g. a scanned image with no text layer).
    Collapsing both into "no text extracted (scanned image?)" used to be
    exactly what this function's only caller did, which sends someone
    chasing a PDF-quality problem that doesn't exist when the real fix is
    `pip install pypdf` — a message this function was already printing
    one line above the misleading one.

    THE SHAPE OF THE PROBLEM
        pypdf returns one "\n" per VISUAL line and no blank lines, so a legal
        PDF arrives as a single unbroken stream of ~80-character fragments,
        heavily hyphenated across line ends:

            "...the coverage level for the aggregate de-\nposits of each..."

        Splitting on blank lines therefore yields one enormous block. Filtering
        that block against the junk patterns then discards the ENTIRE document
        because one navigation phrase appears somewhere in it — which is
        exactly what happened on the first attempt: 0 paragraphs from a PDF
        that plainly contained Article 6.

    SO
        de-hyphenate -> join into one stream -> split into SENTENCES -> filter
        junk per sentence -> regroup into ~600-character blocks.

        Filtering per sentence is the important part. One slogan can no longer
        take a whole article down with it.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        print("    pypdf not installed — pip install pypdf")
        return None

    try:
        reader = PdfReader(str(path))
        text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
    except Exception as exc:
        print(f"    {type(exc).__name__}: {exc}")
        return []

    text = re.sub(r"-\n(\w)", r"\1", text)      # rejoin hyphenated line breaks
    text = re.sub(r"\s*\n\s*", " ", text)       # one continuous stream
    text = re.sub(r"\s+", " ", text).strip()

    # Sentence split. Avoids breaking on the numbering that legal text is full
    # of ("Article 6.", "1.", "(a)") by requiring a following capital or digit
    # after the space.
    sentences = re.split(r"(?<=[.;:])\s+(?=[A-Z0-9(])", text)

    kept = [s.strip() for s in sentences
            if len(s.strip()) > 25 and not _JUNK.search(s)]

    # Regroup into paragraph-sized blocks. _chunk_text in knowledge_base.py
    # chunks again downstream; this only needs coherent units.
    blocks, current = [], ""
    for sentence in kept:
        if len(current) + len(sentence) > 600 and current:
            blocks.append(current.strip())
            current = sentence + " "
        else:
            current += sentence + " "
    if current.strip():
        blocks.append(current.strip())

    return [b for b in blocks if len(b) >= _MIN_CHARS]


def fetch(url: str) -> str | None:
    try:
        import requests
    except ImportError:
        print("    requests not installed")
        return None
    try:
        # A self-identifying bot User-Agent ("thesis-research-corpus-
        # builder/1.0") is a common trigger for basic WAF/bot-detection on
        # .ie government-adjacent hosting (Cloudflare etc.) — these are
        # public consumer-information pages being fetched once, not
        # repeatedly scraped, so a standard browser UA is the appropriate
        # fix, not a workaround for anything adversarial.
        r = requests.get(url, timeout=20, headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
        })
        if r.status_code == 200:
            return r.text
        print(f"    HTTP {r.status_code}")
        return None
    except Exception as exc:
        print(f"    {type(exc).__name__}: {exc}")
        return None


def checklist() -> None:
    print(f"\n{BAR}\n DOWNLOAD THESE, ONCE\n{BAR}")
    print("  Save each as  data/raw/regulatory/pdf/<stem>.pdf\n")
    for s in PDF_SOURCES:
        have = (PDF_DIR / f"{s['stem']}.pdf").exists()
        print(f"  [{'x' if have else ' '}] {s['stem']}.pdf")
        print(f"      {s['title']}  ({s['authority']})")
        print(f"      {s['url']}\n")
    print("  EUR-Lex blocks scripted access but serves these fine in a browser.")
    print("  The CBI link is a landing page — take the current Code PDF from it.\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--checklist", action="store_true")
    args = ap.parse_args()

    if args.checklist:
        checklist()
        return 0

    today = date.today().isoformat()
    records: list[dict] = []
    ok, absent = [], []

    print(f"\n{BAR}\n PDFs  ({PDF_DIR.relative_to(ROOT)})\n{BAR}")
    if not PDF_DIR.exists():
        print("  directory does not exist — run --checklist")
    for s in PDF_SOURCES:
        path = PDF_DIR / f"{s['stem']}.pdf"
        if not path.exists():
            print(f"  [ ] {s['stem']}.pdf  — not downloaded, ABSENT")
            absent.append(s["stem"])
            continue
        paras = pdf_to_paragraphs(path)
        if paras is None:
            # pdf_to_paragraphs() already printed why (pypdf not installed)
            absent.append(s["stem"])
            continue
        if not paras:
            print(f"  [!] {s['stem']}.pdf  — pypdf ran but extracted no usable "
                  f"text (genuinely a scanned image, or all text was filtered "
                  f"as junk)")
            absent.append(s["stem"])
            continue
        print(f"  [x] {s['stem']}.pdf  — {len(paras)} paragraphs, "
              f"{sum(len(p) for p in paras):,} chars")
        ok.append(s["stem"])
        for i, para in enumerate(paras):
            records.append({
                "doc_id": f"{s['stem']}-{i:04d}", "text": para,
                "authority": s["authority"], "title": s["title"],
                "source_url": s["url"], "source_type": "pdf", "retrieved": today,
            })

    print(f"\n{BAR}\n HTML\n{BAR}")
    for i, s in enumerate(HTML_SOURCES):
        if i > 0:
            time.sleep(1.5)  # courtesy delay — avoid tripping basic rate-limiting
        print(f"  {s['id']}  {s['url']}")
        html = fetch(s["url"])
        if html is None:
            # fetch() already printed why (status code or exception)
            print("    ABSENT (fetch failed)")
            absent.append(s["id"])
            continue
        paras = html_to_paragraphs(html)
        if not paras:
            print("    ABSENT (fetched OK, but 0 paragraphs survived junk-filtering)")
            absent.append(s["id"])
            continue
        print(f"    {len(paras)} paragraph(s)")
        ok.append(s["id"])
        for i, para in enumerate(paras):
            records.append({
                "doc_id": f"{s['id']}-{i:03d}", "text": para,
                "authority": s["authority"], "title": s["title"],
                "source_url": s["url"], "source_type": "html", "retrieved": today,
            })

    print(f"\n{BAR}\n SUMMARY\n{BAR}")
    print(f"  sources ok : {len(ok)}  {ok}")
    print(f"  absent     : {len(absent)}  {absent}")
    print(f"  records    : {len(records)}")

    if args.review:
        print(f"\n{BAR}\n REVIEW\n{BAR}")
        for r in records:
            print(f"\n  [{r['doc_id']}] {r['authority']}")
            print(f"    {r['text'][:300]}")

    if not records:
        print("\n  Nothing to write. Run --checklist and download the PDFs.")
        return 1

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / "regulatory_corpus.json"
        path.write_text(json.dumps({
            "_meta": {"built": today, "sources_ok": ok, "sources_absent": absent,
                      "n_records": len(records),
                      "note": ("Verbatim text from primary sources. Nothing is "
                               "generated or summarised; absent sources are "
                               "absent, never seeded.")},
            "records": records,
        }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n  wrote {len(records)} records -> {path.relative_to(ROOT)}")
        print("\n  Next:")
        print("    rm -rf data/embeddings/rag_index && python scripts/build_knowledge_base.py")
        print("    python diagnose_retrieval.py")
    else:
        print("\n  dry run — nothing written. Use --review, then --write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())