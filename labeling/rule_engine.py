import re
import pandas as pd
from typing import Dict, List

# Anchors/contexts
DELIVERY_ANCHOR = r"(ship(ping|ped)?|deliver(y|ed|ies)|arriv(e|ed|al)|shipment|courier|tracking)"
LATENCY_CONTEXT = r"(latenc(y|ies)|bluetooth|cursor|input|lag|frame|audio|video|motion|delay|stutter)"
PACKAGING = r"(box|package|packaging|parcel|envelope)"
DAMAGE = r"(damag(ed|e)?|crush(ed)?|dent(ed)?|torn|ripp?ed|open(ed)?|broken|leaking)"
ARRIVAL = r"(arriv(ed|e)|came|delivered|received|showed up)"
RECV_WORD = r"(received|got|sent|shipped|delivered)"
EXPECT_WORD = r"(order(ed)?|supposed to|should be|expected|as described|description)"
ITEM_WORD = r"(item|product|model|version|size|color)"

DELIVERY_LATE = rf"\b(late|delay(ed)?|slow)\b(?=.*\b{DELIVERY_ANCHOR}\b)|\b{DELIVERY_ANCHOR}\b(?=.*\b(late|delay(ed)?|slow)\b)"

ISSUES: Dict[str, Dict[str, List[str]]] = {
    "delivery_delay": {
        "include": [
            DELIVERY_LATE,
            r"\b(arrived)\b.*\b(way\s+too\s+late|too\s+late|late)\b",
            r"\b(missed delivery date|arrived after (the )?expected date|arrived after expected)\b",
            rf"\b(took)\s+\d+\s+(day|days|week|weeks)\b(?=.*\b{DELIVERY_ANCHOR}\b)",
            rf"\b(took)\s+(two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks)\b(?=.*\b{DELIVERY_ANCHOR}\b)",
            r"\b(never arrived|didn'?t arrive|lost package|not delivered)\b",
        ],
        "exclude": [
            r"\b(delivered promptly|delivered quickly|delivered fast)\b",
            r"\b(super\s+fast|very\s+fast)\s+(shipping|delivery)\b",
            r"\b(arrived early|came early|delivered early|next day)\b",
            r"\b(on time|right on time|as scheduled|within a few days)\b",
            r"\b(too late to return|late to return|late for return|missed (the )?return window)\b",
            r"\b(opened|tested|used|tried|registered|installed)\b.*\btoo late\b",
            r"\btoo late\b.*\b(open|test|use|try|register|install)\b",
            r"\b(can'?t return|couldn'?t return)\b",
            rf"\b{LATENCY_CONTEXT}\b",
        ],
    },
    "damaged_packaging": {
        "include": [
            rf"\b{ARRIVAL}\b.*\b{PACKAGING}\b.*\b{DAMAGE}\b",
            rf"\b{PACKAGING}\b.*\b{DAMAGE}\b.*\b(on arrival|when it arrived|arrived)\b",
            r"\b(open box|seal was broken|broken seal)\b",
        ],
        "exclude": [
            r"\b(cheap\s+plastic|flimsy|build quality)\b",
            r"\b(advertising and packaging|packaging (says|states|claims)|packaging exclaims)\b",
            r"\b(opened the package|opened the packaging)\b(?!.*\b(arrived open|broken seal)\b)",
        ],
    },
    "defective_product": {
        "include": [
            r"\b(DOA|dead on arrival)\b",
            r"\b(defect(ive)?|faulty|malfunction(ing)?)\b",
            r"\b(stopped working|quit working|died|failed)\b",
            r"\b(doesn'?t work|won'?t work|not working|won'?t turn on|won'?t power on|no power)\b",
            r"\b(broke|broken)\b.*\b(after|within)\b.*\b(\d+\s*(day|days|week|weeks|month|months))\b",
        ],
        "exclude": [
            r"\b(box|package|packaging|parcel)\b.*\b(damag(ed|e)?|crush(ed)?|dent(ed)?|torn|ripp?ed|open(ed)?|broken)\b",
            r"\b(not compatible|incompatible|doesn'?t fit|won'?t fit|too (big|small))\b",
            r"\b(won'?t work|doesn'?t work)\b.*\b(with|on)\b.*\b(my|this|that|a|an)\b",
            r"\b(can'?t (connect|pair|sync)|won'?t (connect|pair|sync)|connection issues?)\b",
            r"\b(set ?up|setup|install|installation|driver|firmware|update)\b.*\b(issue|problem|fail)\b",
        ],
    },
    "wrong_item": {
        "include": [
            rf"\b{RECV_WORD}\b.*\bwrong\b.*\b{ITEM_WORD}\b",
            r"\b(wrong item|wrong product|sent me the wrong|not what i ordered)\b",
            rf"\border(ed)?\b.*\bbut\b.*\b{RECV_WORD}\b.*\b{ITEM_WORD}\b",
            rf"\b{RECV_WORD}\b.*\b(instead of|rather than)\b.*\b{ITEM_WORD}\b",
        ],
        "exclude": [
            r"\b(i ordered the wrong|my mistake|i picked the wrong|i accidentally ordered)\b",
            r"\b(open box|seal was broken|used|returned item|previously opened)\b",
        ],
    },
    "missing_accessories": {
        "include": [
            r"\b(missing|didn'?t come with|didn'?t include|not included)\b.*\b(cable|charger|adapter|remote|manual|screws|parts?)\b",
            r"\b(no)\s+\b(cable|charger|adapter|remote|manual)\b(?=.*\b(in the box|in-box|package|packaging)\b)",
            r"\b(missing parts?)\b",
        ],
        "exclude": [
            r"\b(comes with|included|it includes|it came with|packaged with)\b.*\b(cable|charger|adapter|remote|manual|screws)\b",
            r"\b(no)\s+\b(remote|manual)\b.*\b(needed|required)\b",
        ],
    },
}

ISSUE_ORDER = ["wrong_item", "missing_accessories", "damaged_packaging", "defective_product", "delivery_delay"]

def compile_rules(issues: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[re.Pattern]]]:
    return {
        issue: {
            "include": [re.compile(p, re.I) for p in spec["include"]],
            "exclude": [re.compile(p, re.I) for p in spec.get("exclude", [])],
        }
        for issue, spec in issues.items()
    }

COMPILED = compile_rules(ISSUES)

def flag_text(text: str, include_patterns: List[re.Pattern], exclude_patterns: List[re.Pattern]) -> int:
    t = text or ""
    if any(p.search(t) for p in exclude_patterns):
        return 0
    return int(any(p.search(t) for p in include_patterns))

def match_issues(text: str) -> list[str]:
    hits: list[str] = []
    for issue in ISSUE_ORDER:
        spec = COMPILED[issue]
        if any(rx.search(text) for rx in spec["exclude"]):
            continue
        if any(rx.search(text) for rx in spec["include"]):
            hits.append(issue)
    return hits

def label_issues(df: pd.DataFrame, text_col: str = "text_full_clean") -> pd.DataFrame:
    df = df.copy()
    df["issues"] = df[text_col].apply(match_issues)
    for issue in ISSUES.keys():
        df[issue] = df["issues"].apply(lambda xs, i=issue: int(i in xs))
    return df
