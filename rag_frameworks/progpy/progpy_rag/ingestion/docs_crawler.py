"""
ingestion/docs_crawler.py

Crawls the ProgPy documentation site and extracts content per page/section.
Each API reference page (one per class) becomes its own chunk.
Guide pages are split by section.

Outputs RawChunks for the chunker to process.
"""

import re
import time
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass

from .github_crawler import RawChunk, detect_domain, detect_pattern


BASE_URL = "https://nasa.github.io/progpy"

# Pages to crawl — ordered from most to least important for RAG
PAGES = [
    # Core guides
    ("prog_models_guide.html",      "guide",    "modeling_simulation"),
    ("prog_algs_guide.html",        "guide",    "state_estimation_prediction"),

    # API reference — one page per class (most valuable for generation)
    ("api_ref/progpy/PrognosticModel.html",     "api",  "PrognosticsModel"),
    ("api_ref/progpy/CompositeModel.html",      "api",  "CompositeModel"),
    ("api_ref/progpy/LinearModel.html",         "api",  "LinearModel"),
    ("api_ref/progpy/DataModel.html",           "api",  "DataModel"),
    ("api_ref/progpy/EnsembleModel.html",       "api",  "EnsembleModel"),
    ("api_ref/progpy/MixtureOfExperts.html",    "api",  "MixtureOfExperts"),
    ("api_ref/progpy/Predictor.html",           "api",  "Predictors"),
    ("api_ref/progpy/StateEstimator.html",      "api",  "StateEstimators"),
    ("api_ref/progpy/SimResult.html",           "api",  "SimResult"),
    ("api_ref/progpy/UncertainData.html",       "api",  "UncertainData"),
    ("api_ref/progpy/Prediction.html",          "api",  "Prediction"),
    ("api_ref/progpy/IncludedModels.html",      "api",  "IncludedModels"),
    ("api_ref/progpy/Loading.html",             "api",  "Loading"),
    ("api_ref/progpy/Utils.html",               "api",  "Utils"),

    # Supplementary
    ("troubleshooting.html",    "guide",    "troubleshooting"),
    ("glossary.html",           "guide",    "glossary"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ProgPy-RAG-Ingestion/1.0)"
}


def fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch a page and return parsed BeautifulSoup, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"    Failed to fetch {url}: {e}")
        return None


def extract_main_content(soup: BeautifulSoup) -> str:
    """Pull the main article content, stripping nav/header/footer noise."""
    # ProgPy docs use a standard Sphinx layout
    main = (
        soup.find("article", {"role": "main"})
        or soup.find("div", {"class": "body"})
        or soup.find("main")
        or soup.find("div", {"id": "main-content"})
    )
    if main:
        return main.get_text(separator="\n", strip=True)
    # Fallback — strip known noise elements and take body text
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def split_into_sections(text: str) -> list[str]:
    """
    Split page text into sections by detecting header-like lines.
    A header line is short, followed by content.
    """
    lines = text.splitlines()
    sections = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Heuristic: short non-empty line followed by blank = section header
        if (
            len(stripped) > 0
            and len(stripped) < 80
            and stripped.isupper() is False
            and not stripped.startswith("(")
        ):
            if len("\n".join(current).strip()) > 100:
                sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    # Filter out tiny sections (nav artifacts)
    return [s for s in sections if len(s) > 100]


def crawl_page(
    page_path: str,
    chunk_type: str,
    name: str,
) -> list[RawChunk]:
    """Fetch one page and return chunks for it."""
    url = f"{BASE_URL}/{page_path}"
    print(f"    Fetching: {url}")

    soup = fetch_page(url)
    if not soup:
        return []

    content = extract_main_content(soup)
    source = f"docs:{page_path}"

    # API reference pages → one chunk per page (focused, not too long)
    if chunk_type == "api":
        return [RawChunk(
            text=content,
            source=source,
            chunk_type="api",
            name=name,
            domain=detect_domain(content),
            pattern=detect_pattern(content, name),
        )]

    # Guide pages → split into sections
    sections = split_into_sections(content)
    chunks = []
    for i, section in enumerate(sections):
        chunks.append(RawChunk(
            text=section,
            source=source,
            chunk_type="guide",
            name=f"{name}_section_{i}",
            domain=detect_domain(section),
            pattern=detect_pattern(section, name),
        ))
    return chunks


def crawl() -> list[RawChunk]:
    """
    Crawl all configured ProgPy doc pages.
    Returns a flat list of RawChunks.
    """
    all_chunks: list[RawChunk] = []

    for page_path, chunk_type, name in PAGES:
        chunks = crawl_page(page_path, chunk_type, name)
        all_chunks.extend(chunks)
        time.sleep(0.3)  # polite crawling

    print(f"  Extracted {len(all_chunks)} raw chunks from docs site")
    return all_chunks