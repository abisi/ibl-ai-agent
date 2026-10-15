"""Exhaustive deck additions for 155 (user 2026-10-05: "now make more exhaustive slides"). SSL_155_FULL=1 makes 155 write
results_full.pptx: the main deck plus, at the matching places, one slide per figure of the full report (report_full.md, built by
report/build_report.py) and its results text, the expert-session control (156), discussion, caveats, open questions and the
supplementary figures. Text and numbers are taken from report_full.md (no typed numbers); take-home lines are qualitative.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TAKE = {
    "I.1": "Choice is decodable from early whole-brain activity in both cohorts, partly from the pre-stimulus state; the cohorts "
           "diverge only late, once licks are under way.",
    "I.2": "Stimulus modality is decodable within 10 ms in both cohorts; R- is higher only later, once lick-related activity differs.",
    "I.3": "Before the first lick, modality is decodable equally in both cohorts; after it, R- is higher.",
    "I.4": "Performance state is weakly decodable, and in R- it tracks behavioural d'.",
    "II.1": "With halves matched in hits and misses, hit / miss decoding does not change in either cohort.",
    "II.2": "Change-point definitions of the learning trial give the largest R+ gains, but each definition selects its own sessions.",
    "II.3": "Against placebo splits of the same session, the R+ change at the whisker change point is special, and larger than in R-.",
    "II.4": "Behaviour scores relate to split effects only in R-, mainly for pre-lick modality decoding.",
    "II.5": "Single-trial margins follow the learning curve, with opposite signs in the two cohorts.",
    "II.6": "At the stored learning trial, R+ decoding exceeds placebo splits and differs from R-, more weakly than at the change point.",
    "II.7": "The change-point effect holds across windows and is carried by step learners.",
    "II.8": "With equal hits before and after, hit / miss decoding does not change; R+ pre-lick modality decoding declines.",
    "III": "Every Part III analysis uses the same tracked units; inference rests on the passive pre -> post change beyond session time.",
    "III.2": "The coding direction rotates between halves in both cohorts; beyond session time, R- passive whisker responses turn away from it.",
    "III.3": "The choice-decoder readout of passive whisker trials falls in R-; part of it is session time and state.",
    "III.4": "Timing and reward rate differ between cohorts by design; as covariates they absorb part of the readout difference.",
    "III.5": "Session time explains part of the post-task changes, most for the decoder; beyond it, the R- whisker response moves away "
             "from the hit / miss axes relative to the auditory response.",
    "III.6": "Both stimuli shrink along their own patterns in both cohorts; only the R- whisker response moves toward the miss level "
             "on the choice axis.",
    "III.7": "The R- direction appears in most area groups; single cells are small and uncorrected.",
    "III.8": "Exposure scales the shrinkage along the sensory axes, not the change along the choice axis.",
}
EXTRA_FIGS = {"I.2": ["113_modality_stim_set1_session_wide.png"], "I.4": ["113_perfstate_set1_session_wide.png"]}
LATEX = [(r"\Delta", "Δ"), (r"\pi", "π"), (r"\theta", "θ"), (r"\beta", "β"), (r"\varepsilon", "ε"), (r"\epsilon", "ε"),
         (r"\sigma", "σ"), (r"\rho", "ρ"), (r"\tau", "τ"), (r"\langle", "⟨"), (r"\rangle", "⟩"), (r"\cdot", "·"), (r"\propto", "∝"),
         (r"\times", "×"), (r"\geq", "≥"), (r"\ge", "≥"), (r"\leq", "≤"), (r"\le", "≤"), (r"\to", "→"), (r"\pm", "±"),
         (r"\approx", "≈"), (r"\sum", "Σ"), (r"\cos", "cos"), (r"\|", "‖"), (r"\,", " "), (r"\;", " "), (r"\quad", "  ")]


# ----------------------------------------------------------------------------------------------------------- parsing
def clean(t):
    t = re.sub(r"```\{=latex\}.*?```", " ", t, flags=re.S)
    t = re.sub(r"!\[.*?\]\(.*?\)(\{.*?\})?", " ", t, flags=re.S)
    t = re.sub(r"\s*\((Supplementary )?(Figures?|Tables?) [^()]*\)", "", t)
    t = re.sub(r"\$+([^$]*)\$+", lambda m: latex(m.group(1)), t)
    t = t.replace("**", "").replace("`", "")
    t = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)\*(?!\w)", r"\1", t)
    t = re.sub(r"= <\s?", "< ", t)
    return re.sub(r"\s+", " ", t).strip()


def latex(s):
    for a, b in LATEX:
        s = s.replace(a, b)
    s = re.sub(r"\\(mathbf|mathrm|boldsymbol|hat|widehat|bar|text|operatorname|mathbb)\{([^{}]*)\}", r"\2", s)
    s = re.sub(r"\\(d?frac)\{([^{}]*)\}\{([^{}]*)\}", r"(\2)/(\3)", s)
    s = re.sub(r"[_^]\{([^{}]*)\}", r"\1", s).replace("_", "").replace("^", "")
    return s.replace("\\", "").replace("{", "").replace("}", "")


def sentences(t):
    """split at full stops only (not at ';', not before 'R+' / 'R-'); fragments < 45 characters join the next sentence"""
    t = t.replace("e.g. ", "e.g.\u00a0").replace("i.e. ", "i.e.\u00a0").replace(" vs. ", " vs\u00a0")
    raw = [x.replace("\u00a0", " ") for x in re.split(r"(?<=\.)\s+(?=[A-Z(*])(?!R[+-])", t) if x.strip()]
    out, carry = [], ""
    for x in raw:
        x = (carry + " " + x).strip() if carry else x
        if len(x) < 45:
            carry = x
        else:
            out.append(x); carry = ""
    if carry:
        out.append(carry)
    return out


def paras_of(blk):
    out = []
    for p in blk.split("\n\n")[1:]:
        q = p.strip()
        if not q or q.startswith(("|", ":", "Table", "```", "!")) or q.startswith("\\"):
            continue
        c = clean(q)
        if len(c) > 30:
            out.append(c)
    return out


def load(md_path):
    md = md_path.read_text(encoding="utf-8")
    sec = lambda a, b: md[md.index(a): md.index(b, md.index(a) + 1)]
    res = sec("\n# Results", "\n# Discussion")
    subs = []
    for blk in re.split(r"\n(?=##+ )", res):
        h = blk.split("\n")[0]
        m = re.match(r"#+ (Part (I{1,3})\.|([IV]+\.\d+)) ?(.*)", h)
        if not m:
            continue
        key = m.group(3) or (m.group(2) if m.group(2) == "III" else None)
        if key is None:
            continue
        figs = [(f, clean(c)) for c, f in re.findall(r"!\[(.*?)\]\(figures/([^)]+)\)", blk, flags=re.S)]
        subs.append(dict(key=key, title=clean(m.group(4)), figs=figs, paras=paras_of(blk)))
    supp = [(f, clean(c)) for c, f in re.findall(r"!\[(.*?)\]\(figures/([^)]+)\)", sec("\n# Supplementary", "\n# Appendix"), flags=re.S)]
    items = lambda s, pat: [clean(x) for x in re.split(pat, s)[1:] if clean(x)]
    return dict(subs=subs, supp=supp,
                key=items(sec("\n# Key results", "\n# Introduction"), r"\n\d+\.\s+"),
                disc=paras_of(sec("\n# Discussion", "\n# Caveats")),
                cav=items(sec("\n# Caveats", "\n# Open"), r"\n\*\s+"),
                open=items(sec("\n# Open", "\n# Supplementary"), r"\n\d+\.\s+"),
                data=paras_of(sec("\n## Data, inclusion", "\n## Decodings")), stats=paras_of(sec("\n## Statistics", "\n# Results")))


# ----------------------------------------------------------------------------------------------------------- slides
def short(cap, n=150):
    c = re.sub(r"^Figure S?\d+\.\s*", "", cap)
    c = re.split(r"(?<=\.)\s", c)[0]
    if c.count("(") > c.count(")"):
        c = c[: c.rfind("(")].rstrip(" ,;")
    return c if len(c) <= n else c[: n - 3].rsplit(" ", 1)[0] + "..."


def bands(D, path, target=0.42):
    """split a tall figure into horizontal bands (cut at the whitest row near each ideal cut) so each fills the slide width"""
    from PIL import Image
    src = Image.open(path); w, h = src.size
    n = int(np.ceil((h / w) / target))
    if n <= 1:
        return [path]
    blank = (np.asarray(src.convert("L"), float) > 245).mean(1) > 0.998
    runs, i = [], 0                                    # (start, end) of blank row runs
    while i < h:
        if blank[i]:
            j = i
            while j < h and blank[j]:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    cuts = [0]
    for k in range(1, n):
        c = h * k / n; lo, hi = cuts[-1] + 0.15 * h / n, c + 0.4 * h / n
        cand = [(b - a, (a + b) // 2) for a, b in runs if lo <= (a + b) / 2 <= hi]
        # longest blank gap near the ideal cut (gaps between panel rows are longer than those inside a panel)
        cuts.append(max(cand, key=lambda t: t[0] - 0.02 * abs(t[1] - c))[1] if cand else int(c))
    cuts.append(h)
    out = []
    for i in range(n):
        q = D.OUTF / "bands" / f"{path.stem}_{i + 1}of{n}.png"; q.parent.mkdir(parents=True, exist_ok=True)
        src.crop((0, cuts[i], w, cuts[i + 1])).save(q, dpi=src.info.get("dpi", (300, 300)))
        out.append(q)
    return out


def fig_slide(D, prs, title, path, cap, take):
    parts = bands(D, path)
    for i, q in enumerate(parts):
        s = D.new_slide(prs, title, short(cap) + (f"  (panels {i + 1}/{len(parts)})" if len(parts) > 1 else ""))
        D.picture(s, q, 0.4, 1.2, w=12.5, h=4.75 if take else 5.75, scale=20, center=True)
        if take:
            D.takehome(s, 0.4, 6.1, 12.5, 0.9, take)


def text_slides(D, prs, title, items, size=13, budget=1650, max_items=9):
    pages, cur, n = [], [], 0
    for it in items:
        if cur and (n + len(it) > budget or len(cur) >= max_items):
            pages.append(cur); cur, n = [], 0
        cur.append(it); n += len(it)
    if cur:
        pages.append(cur)
    for i, pg in enumerate(pages):
        s = D.new_slide(prs, title + (f" ({i + 1}/{len(pages)})" if len(pages) > 1 else ""))
        D.bullets(s, 0.5, 1.3, 12.3, 5.7, pg, size=size)


def subsection(D, prs, R, sub):
    figs = list(sub["figs"]) + [(f, cap) for f, cap in R["supp"] if f in EXTRA_FIGS.get(sub["key"], [])]
    take = TAKE.get(sub["key"], "")
    title = sub["title"] or "Stimulus-onset geometry: R- whisker responses decouple from the lick axis"
    for f, cap in figs:
        p = D.RDIR / "figures" / f
        if p.exists():
            fig_slide(D, prs, f"{sub['key']}  {title}", p, cap, take)
    sents = [x for para in sub["paras"] for x in sentences(para)]
    if sents:
        text_slides(D, prs, f"{sub['key']}  results", sents)


def part(D, prs, R, roman):
    for sub in R["subs"]:
        if sub["key"] == roman or sub["key"].startswith(roman + "."):
            subsection(D, prs, R, sub)


def overview(D, prs, R):
    text_slides(D, prs, "Key results", R["key"], size=12, budget=1900, max_items=6)
    text_slides(D, prs, "Data, inclusion and statistics", [x for p in R["data"] + R["stats"] for x in sentences(p)], size=12, budget=1900)


def expert_slide(D, prs):
    path = D.PUB / "156_expert_control.png"
    st = D.EA / "156_stats.csv"
    if not (path.exists() and st.exists()):
        return
    T = pd.read_csv(st)
    g = lambda m, test, grp: T[(T.measure == m) & (T.test == test) & (T.group == grp)].iloc[0]
    P = lambda r: f"{D.H.pnum(r.p_nonparam)} | {D.H.pnum(r.p_param)}"
    rows = []
    for m, lab in (("shift_excess_dWR", "lick axis, whisker cosine"), ("shift_excess_dWAP", "lick axis, W - A projection"),
                   ("shift_excess_dWA", "decoder, W - A readout"), ("dx_W", "state space, whisker along the choice axis")):
        a, b = g(m, "R+ vs R- (Mann-Whitney | Welch)", "learning"), g(m, "R+ vs R- (Mann-Whitney | Welch)", "expert")
        i = g(m, "cohort x stage (permutation | OLS)", "interaction")
        rows.append(f"{lab}: R+ vs R- learning p = {P(a)}, expert p = {P(b)}; cohort x stage p = {P(i)}.")
    n = T[(T.measure == "shift_excess_dWR") & (T.test == "vs 0 (Wilcoxon | t)")].set_index("group").n
    rows.append(f"Expert lick-axis sessions: R+ {int(n['R+ expert'])}, R- {int(n['R- expert'])} (expert R- mice rarely lick to the "
                "whisker, so few sessions have enough hits for a hit / miss axis).")
    s = D.new_slide(prs, "Expert sessions as a control", "Same pipeline on expert days; filled: learning, open: expert (156)")
    D.picture(s, path, 0.3, 1.2, w=7.0, h=5.9, scale=20)
    D.bullets(s, 7.5, 1.3, 5.5, 3.9, rows, size=12)
    D.takehome(s, 7.5, 5.4, 5.5, 1.4, "No cohort difference on expert days; the contrast measures look learning-specific (interaction "
               "p ~ 0.05-0.09), but the R- expert group is small.")


def ending(D, prs, R):
    text_slides(D, prs, "Discussion", [x for p in R["disc"] for x in sentences(p)])
    text_slides(D, prs, "Caveats and limitations", R["cav"], size=12, budget=1900)
    text_slides(D, prs, "Open questions and next steps", R["open"], size=13)


def supplementary(D, prs, R):
    used = {f for v in EXTRA_FIGS.values() for f in v}
    for f, cap in R["supp"]:
        p = D.RDIR / "figures" / f
        if f not in used and p.exists():
            fig_slide(D, prs, "Supplementary: " + short(cap, 70), p, cap, "")
