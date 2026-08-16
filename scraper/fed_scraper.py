"""
Scrapes federalreserve.gov for:
  - FOMC statements     (monetarypolicy/fomccalendars.htm -> press releases)
  - FOMC minutes        (monetarypolicy/fomcminutes*.htm)
  - Fed speeches         (newsevents/speech/*.htm)

Design notes:
  - No API key needed; this is public HTML.
  - Fed pages are largely static server-rendered HTML -> requests + bs4 is enough,
    no need for a headless browser.
  - Rate-limit yourself (1 req/sec) to not get IP-throttled during a 3-day sprint.
  - Output: one JSON file per document in data/raw/, schema below, so the
    ingest stage doesn't need to know anything about HTML.

Document schema (data/raw/<type>_<date>_<slug>.json):
{
  "doc_type": "statement" | "minutes" | "speech",
  "date": "YYYY-MM-DD",
  "title": str,
  "speaker": str | null,       # speeches only
  "url": str,
  "text": str,                 # cleaned plain text
  "scraped_at": iso8601 str
}
"""
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"
HEADERS = {"User-Agent": "MacroPilot-research-bot/0.1 (student project; contact: set-your-email)"}
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
RATE_LIMIT_SEC = 1.0


def _get(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    time.sleep(RATE_LIMIT_SEC)
    return BeautifulSoup(resp.text, "html.parser")


def _clean_text(soup: BeautifulSoup) -> str:
    # Fed content pages wrap body copy in <div id="article"> or <div class="col-xs-12 col-sm-8 ...">
    container = soup.find("div", id="article") or soup.find("main") or soup.body
    if container is None:
        return ""
    for tag in container.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = container.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _slug(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
    return s[:60]


def _save(doc: dict) -> Path:
    fname = f"{doc['doc_type']}_{doc['date']}_{_slug(doc['title'])}.json"
    path = RAW_DIR / fname
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def scrape_fomc_statements(years: list[int]) -> list[Path]:
    """fomccalendars.htm lists full recent history on one page (not paginated by year),
    so fetch it once and filter matches by year instead of looping per year."""
    saved = []
    idx_url = f"{BASE}/monetarypolicy/fomccalendars.htm"
    try:
        soup = _get(idx_url)
    except requests.RequestException as e:
        print(f"[warn] could not fetch calendar index: {e}")
        return saved

    links = soup.find_all("a", href=re.compile(r"/newsevents/pressreleases/monetary\d{8}a\.htm"))
    seen_dates = set()
    for a in links:
        href = urljoin(BASE, a["href"])
        m = re.search(r"monetary(\d{8})a\.htm", href)
        if not m:
            continue
        date_raw = m.group(1)
        date = datetime.strptime(date_raw, "%Y%m%d").strftime("%Y-%m-%d")
        year = int(date_raw[:4])
        if year not in years or date in seen_dates:
            continue
        seen_dates.add(date)
        try:
            page = _get(href)
        except requests.RequestException as e:
            print(f"[warn] failed {href}: {e}")
            continue
        doc = {
            "doc_type": "statement",
            "date": date,
            "title": f"FOMC Statement {date}",
            "speaker": None,
            "url": href,
            "text": _clean_text(page),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }
        if doc["text"]:
            saved.append(_save(doc))
            print(f"[ok] statement {date}")
    return saved


def scrape_fomc_minutes(years: list[int]) -> list[Path]:
    saved = []
    idx_url = f"{BASE}/monetarypolicy/fomccalendars.htm"
    try:
        soup = _get(idx_url)
    except requests.RequestException as e:
        print(f"[warn] could not fetch minutes index: {e}")
        return saved

    links = soup.find_all("a", href=re.compile(r"/monetarypolicy/fomcminutes\d{8}\.htm"))
    seen_dates = set()
    for a in links:
        href = urljoin(BASE, a["href"])
        m = re.search(r"fomcminutes(\d{8})\.htm", href)
        date = datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d") if m else "unknown"
        year = int(m.group(1)[:4]) if m else None
        if year not in years or date in seen_dates:
            continue
        seen_dates.add(date)
        try:
            page = _get(href)
        except requests.RequestException as e:
            print(f"[warn] failed {href}: {e}")
            continue
        doc = {
            "doc_type": "minutes",
            "date": date,
            "title": f"FOMC Minutes {date}",
            "speaker": None,
            "url": href,
            "text": _clean_text(page),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }
        if doc["text"]:
            saved.append(_save(doc))
            print(f"[ok] minutes {date}")
    return saved


def scrape_speeches(years: list[int], limit: int | None = None) -> list[Path]:
    saved = []
    for year in years:
        idx_url = f"{BASE}/newsevents/speech/{year}-speeches.htm"
        try:
            soup = _get(idx_url)
        except requests.RequestException as e:
            print(f"[warn] could not fetch speech index for {year}: {e}")
            continue
        links = soup.find_all("a", href=re.compile(r"/newsevents/speech/\w+\d{8}a\.htm"))
        for a in links:
            href = urljoin(BASE, a["href"])
            m = re.search(r"(\w+)(\d{8})a\.htm", href)
            speaker_slug, date_raw = (m.group(1), m.group(2)) if m else (None, None)
            date = datetime.strptime(date_raw, "%Y%m%d").strftime("%Y-%m-%d") if date_raw else "unknown"
            try:
                page = _get(href)
            except requests.RequestException as e:
                print(f"[warn] failed {href}: {e}")
                continue
            title_tag = page.find("h3", class_="title") or page.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else f"Speech {date}"
            speaker_tag = page.find("p", class_="speaker")
            speaker = speaker_tag.get_text(strip=True) if speaker_tag else speaker_slug
            doc = {
                "doc_type": "speech",
                "date": date,
                "title": title,
                "speaker": speaker,
                "url": href,
                "text": _clean_text(page),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }
            if doc["text"]:
                saved.append(_save(doc))
                print(f"[ok] speech {date} - {speaker}")
            if limit and len(saved) >= limit:
                return saved
    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--speech-limit", type=int, default=60)
    parser.add_argument("--skip", nargs="*", default=[], choices=["statements", "minutes", "speeches"])
    args = parser.parse_args()

    if "statements" not in args.skip:
        scrape_fomc_statements(args.years)
    if "minutes" not in args.skip:
        scrape_fomc_minutes(args.years)
    if "speeches" not in args.skip:
        scrape_speeches(args.years, limit=args.speech_limit)
