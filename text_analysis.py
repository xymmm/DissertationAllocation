"""
text_analysis.py
================
Text analysis over dissertation titles and proposals.

Three things are extracted here, all without any external model, so that the
office can run the whole pipeline with no network access and get the same
result every time:

1. **Key phrases.** The terms that distinguish one proposal from the rest of
   the cohort, obtained from term frequency weighted against the corpus. These
   drive part of the match score, and, more usefully, they are shown alongside
   each pairing as the evidence for it, so that a decision can be explained in
   the student's own words rather than as a number.

2. **Latent semantic similarity.** Term overlap alone misses a proposal that
   says "stock levels" where the supervisor's profile says "inventory". Reducing
   the term-document matrix to a smaller number of latent dimensions places
   those two in nearly the same direction, because they occur in the company of
   the same other words across the corpus, which recovers a good deal of the
   synonymy that a plain keyword match throws away.

3. **Requirements.** Whether the proposal implies primary data collection,
   fieldwork with human participants, access to company records, or a
   particular computational technique. These change what a supervisor needs to
   be able to offer, and they change what the office needs to chase, most
   obviously ethics approval, which is better discovered in February than in
   June.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Words that carry no signal in a dissertation proposal, over and above the
# ordinary English stop words: nearly every submission contains them.
DOMAIN_STOPWORDS = {
    "dissertation", "project", "study", "research", "aim", "aims", "objective",
    "objectives", "paper", "thesis", "chapter", "literature", "review",
    "analysis", "analyse", "analyze", "investigate", "investigates",
    "investigating", "examine", "examines", "examining", "explore", "explores",
    "exploring", "understand", "understanding", "data", "using", "use", "used",
    "based", "approach", "method", "methods", "methodology", "results",
    "findings", "contribution", "student", "university", "business", "topic",
    "will", "would", "also", "within", "across", "towards", "well",
}

REQUIREMENT_PATTERNS: Dict[str, List[str]] = {
    "primary data collection": [
        r"\bprimary data\b", r"\bcollect(?:ing|ed)? (?:my |our )?own data\b",
        r"\bfieldwork\b", r"\bquestionnaire\b", r"\bsurvey (?:of|among|with)\b",
        r"\binterview(?:s|ing)?\b", r"\bfocus group\b",
    ],
    "human participants": [
        r"\bparticipants?\b", r"\brespondents?\b", r"\bvolunteers?\b",
        r"\binterviewees?\b", r"\bconsent\b",
    ],
    "company or partner access": [
        r"\bcompany data\b", r"\bindustry partner\b", r"\bplacement\b",
        r"\bmy employer\b", r"\binternal (?:records|data)\b",
        r"\bcase (?:company|organisation|organization)\b",
    ],
    "restricted or paid data": [
        r"\bbloomberg\b", r"\brefinitiv\b", r"\bdatastream\b", r"\bcompustat\b",
        r"\bwrds\b", r"\bnielsen\b", r"\bproprietary (?:data|dataset)\b",
    ],
    "computational workload": [
        r"\bsimulation\b", r"\bagent[- ]based\b", r"\boptimisation\b", r"\boptimization\b",
        r"\bmachine learning\b", r"\bneural\b", r"\bdeep learning\b",
        r"\blarge language model\b", r"\bweb scraping\b", r"\bscrap(?:e|ing)\b",
    ],
    "secondary data only": [
        r"\bsecondary data\b", r"\bpublicly available\b", r"\bpublished (?:accounts|reports)\b",
        r"\bopen data\b", r"\bdatabase of\b",
    ],
}

ETHICS_TRIGGERS = {"primary data collection", "human participants"}


# ---------------------------------------------------------------------------
# Vectorisation shared by the phrase and similarity routines
# ---------------------------------------------------------------------------

@dataclass
class TextModel:
    """Fitted vectoriser plus the reduced space, if one could be built."""
    vectorizer: object
    matrix: object
    terms: np.ndarray
    reduced: Optional[np.ndarray] = None
    n_students: int = 0


def _is_cjk(corpus: Sequence[str]) -> bool:
    joined = " ".join(corpus)
    return len(CJK_RE.findall(joined)) / max(len(joined), 1) > 0.05


def fit_text_model(student_texts: Sequence[str], supervisor_texts: Sequence[str],
                   n_components: int = 120) -> Optional[TextModel]:
    """Fit a TF-IDF model over both sides and reduce it for semantic comparison."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
    except ImportError:
        return None

    corpus = list(student_texts) + list(supervisor_texts)
    if not any(t.strip() for t in corpus):
        return None

    if _is_cjk(corpus):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1,
                              sublinear_tf=True)
    else:
        stop = list(set(TfidfVectorizer(stop_words="english").get_stop_words())
                    | DOMAIN_STOPWORDS)
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2,
                              max_df=0.6, stop_words=stop, sublinear_tf=True)
    try:
        matrix = vec.fit_transform(corpus)
    except ValueError:
        return None
    if matrix.shape[1] == 0:
        return None

    reduced = None
    k = min(n_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
    if k >= 2:
        try:
            svd = TruncatedSVD(n_components=k, random_state=0)
            reduced = normalize(svd.fit_transform(matrix))
        except Exception:
            reduced = None

    return TextModel(vectorizer=vec, matrix=matrix,
                     terms=np.asarray(vec.get_feature_names_out()),
                     reduced=reduced, n_students=len(student_texts))


def similarity_matrices(model: TextModel) -> Tuple[np.ndarray, np.ndarray]:
    """Return the surface and the latent similarity between students and supervisors."""
    ns = model.n_students
    surface = (model.matrix[:ns] @ model.matrix[ns:].T).toarray()
    surface = np.clip(surface, 0.0, 1.0)
    if model.reduced is not None:
        latent = np.clip(model.reduced[:ns] @ model.reduced[ns:].T, 0.0, 1.0)
    else:
        latent = surface
    return surface, latent


def key_phrases(model: TextModel, top_n: int = 8) -> List[List[str]]:
    """The most distinctive terms of each student proposal.

    Multi-word phrases are preferred over the single words they contain, since
    "inventory management" tells the office something that "inventory" and
    "management" separately do not, and a phrase that is wholly contained in a
    longer retained phrase is dropped to keep the evidence readable.
    """
    out: List[List[str]] = []
    ns = model.n_students
    sub = model.matrix[:ns]
    for i in range(ns):
        row = sub.getrow(i)
        if row.nnz == 0:
            out.append([])
            continue
        idx = row.indices[np.argsort(-row.data)]
        ranked = [str(model.terms[j]) for j in idx[: top_n * 4]]
        kept: List[str] = []
        for phrase in ranked:
            if any(phrase != other and phrase in other for other in ranked[:top_n * 2]):
                continue
            if any(phrase in k or k in phrase for k in kept):
                continue
            kept.append(phrase)
            if len(kept) >= top_n:
                break
        out.append(kept)
    return out


def phrase_overlap(student_phrases: Sequence[Sequence[str]],
                   supervisor_texts: Sequence[str]) -> Tuple[np.ndarray, List[List[List[str]]]]:
    """Share of each student's key phrases that appear in a supervisor's profile.

    Returns both the score and, for every pair, the phrases that actually
    matched, which is what the office shows a student who asks why.
    """
    prepared = [" " + re.sub(r"\s+", " ", t.lower()) + " " for t in supervisor_texts]
    n, m = len(student_phrases), len(prepared)
    scores = np.zeros((n, m))
    evidence: List[List[List[str]]] = [[[] for _ in range(m)] for _ in range(n)]
    for i, phrases in enumerate(student_phrases):
        if not phrases:
            continue
        weights = np.array([1.0 / (1 + 0.25 * r) for r in range(len(phrases))])
        weights = weights / weights.sum()
        for j, text in enumerate(prepared):
            hit = 0.0
            for r, phrase in enumerate(phrases):
                if f" {phrase} " in text or phrase in text:
                    hit += weights[r]
                    evidence[i][j].append(phrase)
            scores[i, j] = min(hit, 1.0)
    return scores, evidence


# ---------------------------------------------------------------------------
# Requirements implied by a proposal
# ---------------------------------------------------------------------------

def detect_requirements(text: str) -> List[str]:
    found = []
    lowered = " " + str(text).lower() + " "
    for label, patterns in REQUIREMENT_PATTERNS.items():
        if any(re.search(p, lowered) for p in patterns):
            found.append(label)
    return found


def requirements_table(students: pd.DataFrame,
                       text_columns: Sequence[str] = ("project_title", "abstract")) -> pd.DataFrame:
    """Flag what each proposal will need, and what the office will have to chase."""
    rows = []
    for _, row in students.iterrows():
        text = " ".join(str(row.get(c, "")) for c in text_columns)
        flags = detect_requirements(text)
        rows.append({
            "student_id": row.get("student_id"),
            "requirements": "; ".join(flags),
            "ethics_review_likely": bool(ETHICS_TRIGGERS & set(flags)),
            "needs_data_access_check": "restricted or paid data" in flags
                                       or "company or partner access" in flags,
            "word_count": len(re.findall(r"\w+", text)),
        })
    return pd.DataFrame(rows)
