"""
elm_client.py
=============
Thin client for the University of Edinburgh ELM platform, which exposes an
OpenAI-compatible gateway and issues API keys from inside ELM itself. Nothing
here is specific to ELM beyond the default configuration, so the same code
works against any OpenAI-compatible endpoint if the office later moves.

The model is used for three jobs and no others:

1. ``extract_profile``  - turn a free-text title and abstract into controlled
   vocabulary tags, which is the step that actually removes most mismatch,
   because students describe the same method in a dozen different ways.
2. ``screen_pair``      - rate a shortlisted student-supervisor pair, used only
   as a small tilt on top of the rule-based score.
3. ``audit_assignment`` - read the finished allocation and flag rows a human
   should look at again, with a written reason.

The allocation itself is never delegated to the model: the office has to be
able to explain an outcome to a student, and a deterministic objective plus a
capacity constraint is explainable in a way that a sampled generation is not.

Data protection: only pseudonymised identifiers leave this process. The
``Pseudonymiser`` replaces the target ID with a per-session token and keeps the
mapping in memory, so that names, matriculation numbers and student IDs are
never written into a prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

DEFAULT_BASE_URL = os.environ.get("ELM_BASE_URL", "")   # 从 ELM 内部取得后填入
DEFAULT_MODEL = os.environ.get("ELM_MODEL", "gpt-4o-mini")
CACHE_PATH = Path(os.environ.get("ELM_CACHE", ".elm_cache.json"))


# ---------------------------------------------------------------------------
# Pseudonymisation
# ---------------------------------------------------------------------------

class Pseudonymiser:
    """Swap real identifiers for session tokens before anything is sent out."""

    def __init__(self, salt: Optional[str] = None):
        self.salt = salt or os.urandom(8).hex()
        self._forward: Dict[str, str] = {}
        self._reverse: Dict[str, str] = {}

    def token(self, real_id: str) -> str:
        real_id = str(real_id)
        if real_id not in self._forward:
            digest = hashlib.sha256((self.salt + real_id).encode()).hexdigest()[:10]
            tok = f"ANON-{digest}"
            self._forward[real_id] = tok
            self._reverse[tok] = real_id
        return self._forward[real_id]

    def real(self, token: str) -> Optional[str]:
        return self._reverse.get(token)


def strip_identifiers(text: str) -> str:
    """Remove obvious personal identifiers from free text before sending."""
    if not text:
        return ""
    text = re.sub(r"\b[sS]\d{6,9}\b", "[ID]", text)                 # matriculation style
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "[EMAIL]", text)  # email
    text = re.sub(r"\b\d{7,12}\b", "[NUM]", text)
    return text


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class ELMConfig:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = os.environ.get("ELM_API_KEY", "")
    model: str = DEFAULT_MODEL
    timeout: int = 90
    max_workers: int = 6
    temperature: float = 0.0
    use_cache: bool = True


class ELMClient:
    def __init__(self, config: ELMConfig):
        self.config = config
        self._cache: Dict[str, str] = {}
        if config.use_cache and CACHE_PATH.exists():
            try:
                self._cache = json.loads(CACHE_PATH.read_text())
            except Exception:
                self._cache = {}

    # -- plumbing ----------------------------------------------------------

    def _key(self, messages: List[dict]) -> str:
        blob = json.dumps([self.config.model, messages], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _save_cache(self) -> None:
        if self.config.use_cache:
            try:
                CACHE_PATH.write_text(json.dumps(self._cache, ensure_ascii=False))
            except Exception:
                pass

    def chat(self, messages: List[dict], json_mode: bool = True, retries: int = 3) -> str:
        if not self.config.base_url or not self.config.api_key:
            raise RuntimeError("ELM base_url 或 api_key 未配置")
        key = self._key(messages)
        if self.config.use_cache and key in self._cache:
            return self._cache[key]

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(retries):
            try:
                resp = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {self.config.api_key}",
                             "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.config.timeout,
                )
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                self._cache[key] = content
                self._save_cache()
                return content
            except Exception as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"ELM 请求失败: {last_error}")

    def chat_json(self, messages: List[dict], fallback: dict) -> dict:
        try:
            raw = self.chat(messages, json_mode=True)
        except Exception as exc:
            return {**fallback, "error": str(exc)}
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {**fallback, "error": "模型输出不是合法 JSON"}

    def map_parallel(self, fn, items: Sequence, progress=None) -> List:
        results: List = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {pool.submit(fn, item): idx for idx, item in enumerate(items)}
            done = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    results[idx] = {"error": str(exc)}
                done += 1
                if progress:
                    progress(done, len(items))
        return results


# ---------------------------------------------------------------------------
# Job 1: normalise a student submission into controlled vocabulary
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """You support a university teaching operations office that allocates \
undergraduate business dissertations to supervisors. You are given one anonymised \
student submission. Map it onto the controlled vocabulary supplied. Never invent tags \
outside the vocabulary. Judge only what the text supports.

Return a JSON object with exactly these keys:
  "areas": array of vocabulary strings, at most 3, ordered by centrality
  "methods": array of vocabulary strings, at most 3
  "clarity": integer 1-5, where 5 means the topic is specific enough to supervise and \
1 means it is too vague to allocate
  "feasibility_flags": array of short strings for concerns such as \
"no data source", "scope too broad", "requires ethics approval", "not a business topic"
  "one_line_summary": a single sentence, at most 25 words
"""


def extract_profile(client: ELMClient, submission: dict,
                    area_vocab: Sequence[str], method_vocab: Sequence[str]) -> dict:
    user = json.dumps({
        "anon_id": submission.get("anon_id"),
        "project_title": strip_identifiers(str(submission.get("project_title", ""))),
        "project_name": strip_identifiers(str(submission.get("project_name", ""))),
        "abstract": strip_identifiers(str(submission.get("abstract", "")))[:3000],
        "references": strip_identifiers(str(submission.get("references", "")))[:1500],
        "area_vocabulary": list(area_vocab),
        "method_vocabulary": list(method_vocab),
    }, ensure_ascii=False)
    out = client.chat_json(
        [{"role": "system", "content": EXTRACT_SYSTEM},
         {"role": "user", "content": user}],
        fallback={"areas": [], "methods": [], "clarity": 3,
                  "feasibility_flags": [], "one_line_summary": ""},
    )
    out["anon_id"] = submission.get("anon_id")
    return out


# ---------------------------------------------------------------------------
# Job 2: screen a shortlisted pair
# ---------------------------------------------------------------------------

SCREEN_SYSTEM = """You assess whether a supervisor could reasonably supervise a given \
undergraduate dissertation topic. You are shown one anonymised topic and a shortlist of \
supervisor profiles. For each supervisor return a fit score from 0 to 100 and a reason of \
at most 20 words. Score on substantive capability only: does the supervisor's research area \
cover the topic, and can they supervise the stated method. Ignore workload and availability, \
which are handled elsewhere. Do not be generous: 50 means plausible but unremarkable.

Return JSON: {"ratings": [{"supervisor_id": str, "score": int, "reason": str}]}
"""


def screen_shortlist(client: ELMClient, student: dict, shortlist: List[dict]) -> dict:
    user = json.dumps({
        "topic": {
            "anon_id": student.get("anon_id"),
            "title": strip_identifiers(str(student.get("project_title", ""))),
            "abstract": strip_identifiers(str(student.get("abstract", "")))[:2000],
            "declared_areas": student.get("areas", ""),
            "declared_methods": student.get("methods", ""),
        },
        "supervisors": [
            {"supervisor_id": s["supervisor_id"],
             "research_areas": s.get("research_areas", ""),
             "methods": s.get("methods", ""),
             "keywords": str(s.get("keywords", ""))[:400]}
            for s in shortlist
        ],
    }, ensure_ascii=False)
    return client.chat_json(
        [{"role": "system", "content": SCREEN_SYSTEM},
         {"role": "user", "content": user}],
        fallback={"ratings": []},
    )


# ---------------------------------------------------------------------------
# Job 3: audit the finished allocation
# ---------------------------------------------------------------------------

AUDIT_SYSTEM = """You are the final check on a completed allocation of undergraduate \
dissertations to supervisors, produced by an optimisation model. Your job is to protect \
the students and the office from a bad pairing that the scoring rules did not catch. \
You are shown a batch of finished pairings.

For each pairing return one of three verdicts:
  "ok"      - the supervisor can clearly supervise this topic
  "query"   - defensible but worth a human glance, for example the method sits outside \
the supervisor's usual toolkit
  "reject"  - the supervisor could not reasonably supervise this topic

Be conservative with "reject": the model does not know the supervisor's full history, and \
a wrong rejection costs the office real work. State the reason in at most 25 words, and \
name the specific area or method that drives the verdict.

Return JSON: {"reviews": [{"anon_id": str, "verdict": "ok"|"query"|"reject", \
"reason": str, "confidence": "low"|"medium"|"high"}]}
"""


def audit_batch(client: ELMClient, pairs: List[dict]) -> List[dict]:
    user = json.dumps({"pairings": pairs}, ensure_ascii=False)
    out = client.chat_json(
        [{"role": "system", "content": AUDIT_SYSTEM},
         {"role": "user", "content": user}],
        fallback={"reviews": []},
    )
    return out.get("reviews", [])


def audit_allocation(client: ELMClient, rows: List[dict], batch_size: int = 8,
                     progress=None) -> List[dict]:
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    results = client.map_parallel(lambda b: audit_batch(client, b), batches, progress=progress)
    flat: List[dict] = []
    for r in results:
        if isinstance(r, list):
            flat.extend(r)
    return flat
