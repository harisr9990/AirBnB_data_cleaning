"""
TidyLab - Smart Data Cleaning Studio
------------------------------------
Install:  pip install customtkinter pandas numpy matplotlib openpyxl pillow
Run:      python tidylab.py
"""
import difflib
import html as htmllib
import os
import re
import time
import warnings
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- palette
BG, RAIL, CARD, CARD2, LINE = "#0d1117", "#090c12", "#161c26", "#1f2733", "#2b3647"
TXT, MUTED = "#e6edf3", "#8b98ab"
ACCENT, GOOD, WARN, BAD, BLUE = "#8b5cf6", "#2dd4bf", "#fbbf24", "#fb7185", "#60a5fa"

PLACEHOLDERS = {"", "error", "unknown", "n/a", "na", "null", "none", "nan", "?", "-", "--", "#n/a"}
NONNEG = re.compile(r"(price|fee|cost|amount|total|spent|quantity|qty|count|nights|age|salary|reviews|number)", re.I)
ID_LIKE = re.compile(r"(^|[\s_])(id|code|zip|lat|long|latitude|longitude|year)($|[\s_])", re.I)
BOOLMAP = {"true": True, "yes": True, "t": True, "y": True, "false": False, "no": False, "f": False, "n": False}


# ===================================================================== ENGINE
def is_num(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def is_text(s):
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)
                or pd.api.types.is_bool_dtype(s))


def text_cols(df):
    return [c for c in df.columns if is_text(df[c])]


def ph_mask(s):
    return s.map(lambda v: isinstance(v, str) and v.strip().lower() in PLACEHOLDERS)


def num_clean(s):
    return pd.to_numeric(s.astype("string").str.replace(r"[$€£¥,%\s]", "", regex=True), errors="coerce")


def infer_type(s):
    """Does this text column secretly hold numbers, dates or booleans?"""
    v = s.dropna()
    v = v[~ph_mask(v)]
    if v.empty:
        return None
    if len(v) > 3000:
        v = v.sample(3000, random_state=0)
    low = v.astype(str).str.strip().str.lower()
    if low.nunique() <= 3 and set(low.unique()) <= set(BOOLMAP):
        return "bool"
    if num_clean(v).notna().mean() >= 0.9:
        return "number"
    if low.str.contains(r"\d").mean() > 0.9 and low.str.contains(r"[-/:]").mean() > 0.9:
        if pd.to_datetime(v, errors="coerce").notna().mean() >= 0.85:
            return "date"
    return None


def convert(s, kind):
    if kind == "number":
        return num_clean(s)
    if kind == "date":
        return pd.to_datetime(s, errors="coerce")
    low = s.astype("string").str.strip().str.lower()
    return low.map(BOOLMAP).astype("boolean")


def find_typos(s):
    """rare spelling -> frequent spelling (lowercase), e.g. brookln -> brooklyn"""
    vc = s.dropna().astype(str).str.lower().value_counts()
    if len(vc) < 2 or len(vc) > 80:
        return {}
    items, out = list(vc.items()), {}
    for val, cnt in items:
        if len(val) < 5:
            continue
        cands = [k for k, c in items if c >= 20 * cnt and k != val]
        m = difflib.get_close_matches(val, cands, n=1, cutoff=0.85)
        if m:
            out[val] = m[0]
    return out


def inconsistent(s):
    v = s.dropna().astype(str)
    raw_n = v.nunique()
    norm_n = v.str.replace(r"\s+", " ", regex=True).str.strip().str.lower().nunique()
    return raw_n > norm_n or bool(find_typos(s))


def profile_table(df):
    rows, n = [], len(df)
    for c in df.columns:
        s = df[c]
        txt = is_text(s)
        ph = int(ph_mask(s).sum()) if txt else 0
        miss = int(s.isnull().sum()) + ph
        pct = 100 * miss / max(n, 1)
        nun = int(s.nunique(dropna=True))
        issues, conv, incons = [], False, False
        cat = txt and 2 <= nun <= 80
        if txt:
            k = infer_type(s)
            if k:
                conv = True
                issues.append({"number": "stored as text (number)", "date": "stored as text (date)",
                               "bool": "stored as text (yes/no)"}[k])
        if ph:
            issues.append(f"{ph} placeholders")
        if pct > 50:
            issues.append("mostly empty")
        if nun <= 1 and n > 1:
            issues.append("constant")
        if cat and inconsistent(s):
            incons = True
            issues.append("inconsistent spelling/case")
        if is_num(s) and NONNEG.search(str(c)) and bool((s < 0).any()):
            issues.append("negative values")
        ex = s.dropna().iloc[0] if s.notna().any() else ""
        kind = ("number" if is_num(s) else "date" if pd.api.types.is_datetime64_any_dtype(s)
                else "boolean" if pd.api.types.is_bool_dtype(s) else "text")
        rows.append({"Column": c, "Type": kind, "Missing %": round(pct, 1), "Unique": nun,
                     "Issues": ("⚠ " + ", ".join(issues)) if issues else "✔ clean",
                     "Example": str(ex)[:28], "_miss": miss, "_conv": conv, "_cat": cat,
                     "_incons": incons, "_issues": issues})
    return rows


def analyze(df):
    prof = profile_table(df)
    dups = int(df.duplicated().sum())
    cells = max(len(df) * max(df.shape[1], 1), 1)
    comp = max(0.0, 1 - sum(r["_miss"] for r in prof) / cells)
    uniq = 1 - dups / max(len(df), 1)
    valid = 1 - sum(1 for r in prof if r["_conv"]) / max(len(prof), 1)
    cat = [r for r in prof if r["_cat"]]
    cons = 1 - sum(1 for r in cat if r["_incons"]) / len(cat) if cat else 1.0
    health = {"Overall": 100 * (0.4 * comp + 0.2 * uniq + 0.2 * valid + 0.2 * cons),
              "Completeness": comp * 100, "Uniqueness": uniq * 100,
              "Validity": valid * 100, "Consistency": cons * 100}
    stats = {"Records": len(df), "Columns": df.shape[1],
             "Missing cells": sum(r["_miss"] for r in prof), "Duplicate rows": dups}
    return prof, stats, health


def suggestions(prof, stats):
    out = []
    conv = [r["Column"] for r in prof if r["_conv"]]
    if conv:
        out.append(("🔢", f"{len(conv)} column(s) look like numbers/dates stored as text: {', '.join(map(str, conv[:5]))}"))
    if stats["Duplicate rows"]:
        out.append(("♊", f"{stats['Duplicate rows']:,} duplicate rows found"))
    if stats["Missing cells"]:
        out.append(("🕳️", f"{stats['Missing cells']:,} missing or placeholder cells"))
    sparse = [r["Column"] for r in prof if "mostly empty" in r["_issues"]]
    if sparse:
        out.append(("🗑️", f"Mostly-empty columns: {', '.join(map(str, sparse[:5]))}"))
    const = [r["Column"] for r in prof if "constant" in r["_issues"]]
    if const:
        out.append(("🧊", f"Constant columns carry no information: {', '.join(map(str, const[:5]))}"))
    inc = [r["Column"] for r in prof if r["_incons"]]
    if inc:
        out.append(("✏️", f"Inconsistent spelling/case or typos in: {', '.join(map(str, inc[:5]))}"))
    neg = [r["Column"] for r in prof if "negative values" in r["_issues"]]
    if neg:
        out.append(("➖", f"Impossible negative values in: {', '.join(map(str, neg[:5]))}"))
    return out or [("✅", "No obvious problems found. This dataset is already clean.")]


def outlier_stats(df):
    res = {}
    for c in df.columns:
        s = df[c]
        if not is_num(s) or ID_LIKE.search(str(c)):
            continue
        v = s.dropna()
        if v.nunique() <= 10:
            continue
        q1, q3 = v.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((v < lo) | (v > hi)).sum())
        res[c] = {"n": n, "pct": 100 * n / len(v), "lo": float(lo), "hi": float(hi)}
    return res


def log(a, step, col, detail, n):
    a.append({"Step": step, "Column": col, "What changed": detail, "Cells": int(n)})


# ----------------------------------------------------------- pipeline steps
def s_names(df, o, a):
    old, new, seen = list(df.columns), [], {}
    for c in old:
        n = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(c).strip())
        n = re.sub(r"[^0-9a-zA-Z]+", "_", n).strip("_").lower() or "column"
        k = seen.get(n, 0)
        seen[n] = k + 1
        new.append(n if k == 0 else f"{n}_{k + 1}")
    ch = [(x, y) for x, y in zip(old, new) if x != y]
    df = df.copy()
    df.columns = new
    if ch:
        log(a, "Column names", "(all)", f"Renamed {len(ch)} columns, e.g. {ch[0][0]} → {ch[0][1]}", len(ch))
    return df


def s_trim(df, o, a):
    df = df.copy()
    for c in text_cols(df):
        old = df[c]
        new = old.map(lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v)
        n = int(((new != old) & old.notna()).sum())
        if n:
            df[c] = new
            log(a, "Whitespace", c, "Trimmed / collapsed extra spaces", n)
    return df


def s_placeholders(df, o, a):
    df = df.copy()
    for c in text_cols(df):
        m = ph_mask(df[c])
        n = int(m.sum())
        if n:
            df.loc[m, c] = np.nan
            log(a, "Placeholders", c, "ERROR / UNKNOWN / N/A / blank → missing", n)
    return df


def s_duplicates(df, o, a):
    n = int(df.duplicated().sum())
    if n:
        df = df.drop_duplicates().reset_index(drop=True)
        log(a, "Duplicates", "(rows)", "Removed exact duplicate rows", n)
    return df


def s_types(df, o, a):
    df = df.copy()
    for c in text_cols(df):
        kind = infer_type(df[c])
        if not kind:
            continue
        filled = int(df[c].notna().sum())
        new = convert(df[c], kind)
        bad = filled - int(new.notna().sum())
        df[c] = new
        extra = f", {bad} invalid → missing" if bad else ""
        label = {"number": "number (removed $ , %)", "date": "date", "bool": "true/false"}[kind]
        log(a, "Smart types", c, f"text → {label}{extra}", filled)
    return df


def s_invalid(df, o, a):
    df = df.copy()
    for c in df.columns:
        if not is_num(df[c]):
            continue
        bad = None
        if NONNEG.search(str(c)):
            bad = df[c] < 0
            why = "negative value → missing"
        elif str(c).endswith("365"):
            bad = df[c] > 365
            why = "more than 365 days → missing"
        if bad is not None and int(bad.sum()):
            n = int(bad.sum())
            df.loc[bad, c] = np.nan
            log(a, "Impossible values", c, why, n)
    return df


def s_categories(df, o, a):
    df = df.copy()
    for c in text_cols(df):
        s = df[c]
        if not 2 <= s.nunique() <= 80:
            continue
        key = s.map(lambda v: v.lower() if isinstance(v, str) else v)
        canon = s.groupby(key).agg(lambda x: x.value_counts().idxmax()).to_dict()
        new = key.map(lambda k: canon.get(k, k) if isinstance(k, str) else k)
        diff = (new != s) & s.notna()
        if int(diff.sum()):
            log(a, "Categories", c, f"Unified case, e.g. {s[diff].iloc[0]} → {new[diff].iloc[0]}", diff.sum())
        typos = find_typos(new)
        if typos:
            spell = {v.lower(): v for v in new.dropna().unique()}
            rep = {spell[t]: spell[g] for t, g in typos.items() if t in spell and g in spell}
            cnt = int(new.isin(list(rep)).sum())
            new = new.map(lambda v: rep.get(v, v))
            log(a, "Typos fixed", c, ", ".join(f"{k} → {v}" for k, v in rep.items()), cnt)
        df[c] = new
    return df


def s_sparse(df, o, a):
    df = df.copy()
    for c in list(df.columns):
        pct = df[c].isnull().mean() * 100
        if pct >= o["drop_thr"]:
            df = df.drop(columns=c)
            log(a, "Dropped columns", c, f"{pct:.0f}% empty", len(df))
        elif df[c].nunique(dropna=True) <= 1 and len(df) > 1:
            df = df.drop(columns=c)
            log(a, "Dropped columns", c, "same value in every row (no information)", len(df))
    return df


def s_missing(df, o, a):
    df = df.copy()
    for c in list(df.columns):
        s = df[c]
        n = int(s.isnull().sum())
        if not n:
            continue
        if is_num(s):
            v = s.median() if o["num_fill"] == "Median" else s.mean()
            how = f"{o['num_fill'].lower()} ({round(float(v), 2)})"
        elif pd.api.types.is_datetime64_any_dtype(s):
            v = s.median()
            how = f"median date ({str(v)[:10]})"
        elif o["txt_fill"] == "Unknown" and not pd.api.types.is_bool_dtype(s):
            v, how = "Unknown", '"Unknown"'
        else:
            m = s.mode()
            if m.empty:
                continue
            v = m.iloc[0]
            how = f"most frequent value ({str(v)[:20]})"
        df[c] = s.fillna(v)
        log(a, "Missing values", c, f"Filled with {how}", n)
    return df


def apply_outliers(df, mode, a):
    st = outlier_stats(df)
    if mode.startswith("Cap"):
        df = df.copy()
        for c, d in st.items():
            if d["n"]:
                df[c] = df[c].clip(d["lo"], d["hi"])
                log(a, "Outliers", c, f"Capped to IQR range {d['lo']:.1f} … {d['hi']:.1f}", d["n"])
    elif mode.startswith("Remove"):
        mask = pd.Series(False, index=df.index)
        for c, d in st.items():
            mask |= (df[c] < d["lo"]) | (df[c] > d["hi"])
        n = int(mask.sum())
        if n:
            df = df[~mask].reset_index(drop=True)
            log(a, "Outliers", "(rows)", "Removed rows containing IQR outliers", n)
    return df


def s_outliers(df, o, a):
    return apply_outliers(df, o["out_mode"], a)


PIPELINE = [
    ("names", "Standardize column names", "Employee Name → employee_name", s_names),
    ("trim", "Trim & collapse whitespace", "' Tea  ' → 'Tea'", s_trim),
    ("placeholders", "Placeholders → missing", "ERROR, UNKNOWN, N/A, ?, -", s_placeholders),
    ("duplicates", "Remove duplicate rows", "exact copies of a row", s_duplicates),
    ("types", "Smart type detection", "'$1,200' → 1200, text dates → dates", s_types),
    ("invalid", "Fix impossible values", "negative prices, 3677 days in a year", s_invalid),
    ("categories", "Unify categories & fix typos", "brookln → Brooklyn", s_categories),
    ("sparse", "Drop empty / constant columns", "columns that add nothing", s_sparse),
    ("missing", "Fill missing values", "median / mean / most frequent", s_missing),
    ("outliers", "Outlier handling (IQR)", "report, cap or remove", s_outliers),
]


def read_file(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


# ======================================================================= GUI
def mkfig(w, h):
    f = Figure(figsize=(w, h), dpi=100)
    f.patch.set_facecolor(CARD)
    return f


def sty(ax):
    ax.set_facecolor(CARD2)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(TXT)
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for sp in ax.spines.values():
        sp.set_color(LINE)


def score_color(v):
    return GOOD if v >= 85 else WARN if v >= 65 else BAD


class Ring(tk.Canvas):
    def __init__(self, master, score, label, size=150):
        super().__init__(master, width=size, height=size, bg=CARD, highlightthickness=0)
        p = 14
        self.create_oval(p, p, size - p, size - p, outline=LINE, width=12)
        if score is not None:
            self.create_arc(p, p, size - p, size - p, start=90, extent=-min(359.9, 3.6 * score),
                            style="arc", outline=score_color(score), width=12)
        self.create_text(size / 2, size / 2 - 6, text="—" if score is None else f"{score:.0f}",
                         fill=TXT, font=("Segoe UI", 30, "bold"))
        self.create_text(size / 2, size / 2 + 26, text=label, fill=MUTED, font=("Segoe UI", 9, "bold"))


NAV = [("Overview", "🏠"), ("Profiler", "🔍"), ("Cleaning Studio", "🧼"), ("Change Log", "📜"),
       ("Before / After", "⚖️"), ("Explorer", "🗂️"), ("Insights", "📊"), ("Outliers", "🎯"),
       ("Export", "📤")]
SUBTITLE = {
    "Overview": "Health score and smart suggestions for your dataset",
    "Profiler": "Column-by-column X-ray with problems highlighted",
    "Cleaning Studio": "Build your cleaning pipeline and run it",
    "Change Log": "Every single change the cleaner made (audit trail)",
    "Before / After": "How the dataset improved",
    "Explorer": "Search and browse the data",
    "Insights": "Charts of the cleaned dataset",
    "Outliers": "IQR outlier detection with one-click fixes",
    "Export": "Save the cleaned data, a report and images",
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")
        self.title("TidyLab - Smart Data Cleaning Studio")
        self.geometry("1320x800")
        self.minsize(1100, 700)
        self.configure(fg_color=BG)
        self.raw = self.df = None
        self.prof_raw = self.prof_new = self.st_raw = self.st_new = self.h_raw = self.h_new = None
        self.audit, self.run_log, self.outl = [], [], {}
        self.file_name, self.page, self.shot_n = "", "Overview", 0
        self.step_vars = {k: ctk.BooleanVar(value=True) for k, *_ in PIPELINE}
        self.num_fill = ctk.StringVar(value="Median")
        self.txt_fill = ctk.StringVar(value="Most frequent")
        self.out_mode = ctk.StringVar(value="Report only")
        self.drop_thr = ctk.DoubleVar(value=50)
        self.nav = {}
        self._build_rail()
        main = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        main.pack(side="right", fill="both", expand=True)
        top = ctk.CTkFrame(main, fg_color=BG, height=78, corner_radius=0)
        top.pack(fill="x", padx=28, pady=(14, 0))
        t = ctk.CTkFrame(top, fg_color="transparent")
        t.pack(side="left")
        self.h_title = ctk.CTkLabel(t, text="", font=("Segoe UI", 26, "bold"), text_color=TXT)
        self.h_title.pack(anchor="w")
        self.h_sub = ctk.CTkLabel(t, text="", font=("Segoe UI", 12), text_color=MUTED)
        self.h_sub.pack(anchor="w")
        ctk.CTkButton(top, text="↺  Reset", width=90, height=34, fg_color=CARD2, hover_color=LINE,
                      command=self.reset).pack(side="right", padx=(8, 0))
        ctk.CTkButton(top, text="📸  Screenshot", width=130, height=34, fg_color=ACCENT,
                      hover_color="#7c3aed", command=self.screenshot).pack(side="right")
        self.toast_lbl = ctk.CTkLabel(top, text="", font=("Segoe UI", 12), text_color=GOOD)
        self.toast_lbl.pack(side="right", padx=16)
        self.body = ctk.CTkFrame(main, fg_color=BG, corner_radius=0)
        self.body.pack(fill="both", expand=True, padx=28, pady=14)
        self.show("Overview")

    # ------------------------------------------------------------- chrome
    def _build_rail(self):
        rail = ctk.CTkFrame(self, width=232, fg_color=RAIL, corner_radius=0)
        rail.pack(side="left", fill="y")
        rail.pack_propagate(False)
        ctk.CTkLabel(rail, text="🧼 TidyLab", font=("Segoe UI", 24, "bold"), text_color="#c4b5fd").pack(
            anchor="w", padx=20, pady=(26, 0))
        ctk.CTkLabel(rail, text="Smart Data Cleaning Studio", font=("Segoe UI", 10),
                     text_color=MUTED).pack(anchor="w", padx=22, pady=(0, 22))
        for name, icon in NAV:
            b = ctk.CTkButton(rail, text=f"  {icon}   {name}", anchor="w", height=40, corner_radius=10,
                              fg_color="transparent", hover_color=CARD2, font=("Segoe UI", 13),
                              command=lambda n=name: self.show(n))
            b.pack(fill="x", padx=12, pady=1)
            self.nav[name] = b
        self.file_lbl = ctk.CTkLabel(rail, text="No dataset", font=("Segoe UI", 10), text_color=MUTED,
                                     wraplength=200)
        self.file_lbl.pack(side="bottom", pady=(0, 18))
        ctk.CTkButton(rail, text="📂  Open Dataset", height=44, fg_color=ACCENT, hover_color="#7c3aed",
                      font=("Segoe UI", 13, "bold"), command=self.load).pack(side="bottom", fill="x",
                                                                            padx=14, pady=(0, 8))

    def show(self, page):
        for w in self.body.winfo_children():
            w.destroy()
        self.page = page
        for n, b in self.nav.items():
            on = n == page
            b.configure(fg_color="#2a2145" if on else "transparent", text_color="#c4b5fd" if on else TXT)
        self.h_title.configure(text=page)
        self.h_sub.configure(text=SUBTITLE[page])
        {"Overview": self.pg_overview, "Profiler": self.pg_profiler, "Cleaning Studio": self.pg_studio,
         "Change Log": self.pg_log, "Before / After": self.pg_compare, "Explorer": self.pg_explorer,
         "Insights": self.pg_insights, "Outliers": self.pg_outliers, "Export": self.pg_export}[page]()

    def toast(self, msg, color=GOOD):
        self.toast_lbl.configure(text=msg, text_color=color)
        self.after(4000, lambda: self.toast_lbl.winfo_exists() and self.toast_lbl.configure(text=""))

    def card(self, parent, **kw):
        return ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14, **kw)

    def lbl(self, parent, text, size=12, bold=False, color=TXT, **kw):
        return ctk.CTkLabel(parent, text=text, font=("Segoe UI", size, "bold" if bold else "normal"),
                            text_color=color, **kw)

    def gate(self, need_clean=False):
        if self.raw is None or (need_clean and self.df is None):
            c = self.card(self.body)
            c.pack(pady=80)
            if self.raw is None:
                self.lbl(c, "📂", 44).pack(padx=70, pady=(30, 4))
                self.lbl(c, "No dataset loaded yet", 18, True).pack()
                self.lbl(c, "Open a CSV or Excel file to start", 12, color=MUTED).pack(pady=(2, 14))
                ctk.CTkButton(c, text="Open Dataset", fg_color=ACCENT, command=self.load).pack(pady=(0, 30))
            else:
                self.lbl(c, "🧼", 44).pack(padx=70, pady=(30, 4))
                self.lbl(c, "Run the cleaning first", 18, True).pack()
                self.lbl(c, "This page compares the data before and after cleaning", 12,
                         color=MUTED).pack(pady=(2, 14))
                ctk.CTkButton(c, text="Go to Cleaning Studio", fg_color=ACCENT,
                              command=lambda: self.show("Cleaning Studio")).pack(pady=(0, 30))
            return False
        return True

    def table(self, parent, df, max_rows=300, max_cols=None, height=12):
        st = ttk.Style()
        st.theme_use("clam")
        st.configure("T.Treeview", background=CARD, fieldbackground=CARD, foreground=TXT, rowheight=25,
                     borderwidth=0)
        st.configure("T.Treeview.Heading", background=CARD2, foreground=TXT, relief="flat",
                     font=("Segoe UI", 10, "bold"))
        st.map("T.Treeview", background=[("selected", ACCENT)])
        box = tk.Frame(parent, bg=CARD)
        cols = list(df.columns)[:max_cols] if max_cols else list(df.columns)
        tree = ttk.Treeview(box, columns=cols, show="headings", style="T.Treeview", height=height)
        sy = ttk.Scrollbar(box, orient="vertical", command=tree.yview)
        sx = ttk.Scrollbar(box, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        box.grid_rowconfigure(0, weight=1)
        box.grid_columnconfigure(0, weight=1)
        tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        sample = df[cols].head(60).astype(str).fillna("")
        for c in cols:
            tree.heading(c, text=str(c))
            longest = max([len(str(c)) + 2] + [len(str(v)) for v in sample[c]])
            tree.column(c, width=max(70, min(360, longest * 8)), anchor="w", stretch=False)
        for r in df[cols].head(max_rows).itertuples(index=False):
            tree.insert("", "end", values=["" if pd.isna(v) else str(v)[:45] for v in r])
        return box

    def embed(self, fig, parent):
        cv = FigureCanvasTkAgg(fig, master=parent)
        cv.get_tk_widget().configure(bg=CARD)
        cv.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        cv.draw()
        return cv

    # ----------------------------------------------------------- load etc.
    def load(self):
        path = filedialog.askopenfilename(
            title="Select dataset", filetypes=[("Data files", "*.csv *.xlsx *.xls"), ("All", "*.*")])
        if not path:
            return
        self.toast("Reading file…", WARN)
        self.update()
        try:
            raw = read_file(path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")
            return
        self.toast("Analyzing data quality…", WARN)
        self.update()
        self.raw, self.file_name = raw, os.path.basename(path)
        self.prof_raw, self.st_raw, self.h_raw = analyze(raw)
        self.df = self.prof_new = self.st_new = self.h_new = None
        self.audit, self.run_log, self.outl = [], [], {}
        self.file_lbl.configure(text=f"{self.file_name}\n{len(raw):,} rows × {raw.shape[1]} cols")
        self.show("Overview")
        self.toast("Dataset loaded ✔")

    def reset(self):
        if self.raw is None:
            return
        if messagebox.askyesno("Reset", "Discard the cleaned data and start again from the raw file?"):
            self.df = self.prof_new = self.st_new = self.h_new = None
            self.audit, self.run_log, self.outl = [], [], {}
            self.show("Overview")

    def screenshot(self):
        try:
            from PIL import ImageGrab
            self.update_idletasks()
            self.update()
            folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
            os.makedirs(folder, exist_ok=True)
            self.shot_n += 1
            name = re.sub(r"[^a-z0-9]+", "_", self.page.lower()).strip("_")
            path = os.path.join(folder, f"{self.shot_n:02d}_{name}.png")
            x, y, w, h = self.winfo_rootx(), self.winfo_rooty(), self.winfo_width(), self.winfo_height()
            ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(path)
            self.toast(f"Saved screenshots/{os.path.basename(path)}")
        except Exception as e:
            self.toast(f"Screenshot failed: {e}", BAD)

    # ============================================================== OVERVIEW
    def pg_overview(self):
        if not self.gate():
            return
        row = ctk.CTkFrame(self.body, fg_color="transparent")
        row.pack(fill="x")
        items = [("RECORDS", "Records", TXT), ("COLUMNS", "Columns", BLUE),
                 ("MISSING CELLS", "Missing cells", BAD), ("DUPLICATE ROWS", "Duplicate rows", WARN)]
        for i, (title, key, col) in enumerate(items):
            row.grid_columnconfigure(i, weight=1)
            c = self.card(row)
            c.grid(row=0, column=i, padx=6, sticky="ew")
            self.lbl(c, title, 10, True, MUTED).pack(pady=(14, 0))
            self.lbl(c, f"{self.st_raw[key]:,}", 28, True, col).pack()
            if self.st_new:
                self.lbl(c, f"→ {self.st_new[key]:,} after cleaning", 11, color=GOOD).pack(pady=(0, 12))
            else:
                self.lbl(c, "raw file", 11, color=MUTED).pack(pady=(0, 12))
        mid = ctk.CTkFrame(self.body, fg_color="transparent")
        mid.pack(fill="x", pady=12)
        mid.grid_columnconfigure(1, weight=1)
        hc = self.card(mid)
        hc.grid(row=0, column=0, padx=6, sticky="ns")
        self.lbl(hc, "Data Health Score", 14, True).pack(anchor="w", padx=20, pady=(16, 0))
        rr = ctk.CTkFrame(hc, fg_color="transparent")
        rr.pack(padx=14, pady=10)
        Ring(rr, self.h_raw["Overall"], "BEFORE").pack(side="left", padx=6)
        Ring(rr, self.h_new["Overall"] if self.h_new else None, "AFTER").pack(side="left", padx=6)
        sc = self.card(mid)
        sc.grid(row=0, column=1, padx=6, sticky="nsew")
        self.lbl(sc, "Quality breakdown", 14, True).pack(anchor="w", padx=20, pady=(16, 6))
        for k in ["Completeness", "Uniqueness", "Validity", "Consistency"]:
            r = ctk.CTkFrame(sc, fg_color="transparent")
            r.pack(fill="x", padx=20, pady=5)
            self.lbl(r, k, 12, color=MUTED, width=110, anchor="w").pack(side="left")
            v = (self.h_new or self.h_raw)[k]
            pb = ctk.CTkProgressBar(r, height=10, progress_color=score_color(v), fg_color=CARD2)
            pb.set(v / 100)
            pb.pack(side="left", fill="x", expand=True, padx=10)
            txt = f"{self.h_raw[k]:.0f}" + (f" → {self.h_new[k]:.0f}" if self.h_new else "")
            self.lbl(r, txt, 12, True, width=80, anchor="e").pack(side="right")
        sg = self.card(self.body)
        sg.pack(fill="both", expand=True, padx=6)
        self.lbl(sg, "💡 Smart suggestions", 14, True).pack(anchor="w", padx=20, pady=(14, 6))
        for icon, text in suggestions(self.prof_raw, self.st_raw):
            self.lbl(sg, f"{icon}  {text}", 12, color=TXT, anchor="w", justify="left",
                     wraplength=900).pack(anchor="w", padx=24, pady=3)
        ctk.CTkButton(sg, text="Open Cleaning Studio →", fg_color=ACCENT, hover_color="#7c3aed",
                      command=lambda: self.show("Cleaning Studio")).pack(anchor="w", padx=22, pady=14)

    # ============================================================== PROFILER
    def pg_profiler(self):
        if not self.gate():
            return
        tv = ctk.CTkTabview(self.body, fg_color=CARD, segmented_button_selected_color=ACCENT)
        tv.pack(fill="both", expand=True)
        t1, t2 = tv.add("Columns"), tv.add("Missing map")
        holder = ctk.CTkFrame(t1, fg_color="transparent")

        def draw(src):
            for w in holder.winfo_children():
                w.destroy()
            prof = self.prof_new if (src == "Cleaned" and self.prof_new) else self.prof_raw
            d = pd.DataFrame(prof)[["Column", "Type", "Missing %", "Unique", "Issues", "Example"]]
            self.table(holder, d, max_rows=500, height=16).pack(fill="both", expand=True)

        seg = ctk.CTkSegmentedButton(t1, values=["Raw", "Cleaned"], command=draw, selected_color=ACCENT)
        seg.set("Raw")
        seg.pack(anchor="w", pady=(4, 8))
        holder.pack(fill="both", expand=True)
        draw("Raw")
        fig = mkfig(9, 4.6)
        ax = fig.add_subplot(111)
        sty(ax)
        d = self.raw
        idx = np.linspace(0, len(d) - 1, min(len(d), 300)).astype(int)
        sm = d.iloc[idx]
        m = np.array([(sm[c].isnull() | (ph_mask(sm[c]) if is_text(sm[c]) else False)).values
                      for c in sm.columns])
        ax.imshow(m, aspect="auto", cmap=ListedColormap([CARD2, BAD]), interpolation="nearest")
        ax.set_yticks(range(len(sm.columns)))
        ax.set_yticklabels([str(c)[:22] for c in sm.columns], fontsize=7)
        ax.set_xlabel("300 sampled rows   (red = missing or placeholder)")
        ax.set_title("Where is the data missing?")
        fig.tight_layout()
        self.embed(fig, t2)

    # ========================================================= CLEANING STUDIO
    def options(self):
        return {"num_fill": self.num_fill.get(), "txt_fill": self.txt_fill.get(),
                "out_mode": self.out_mode.get(), "drop_thr": float(self.drop_thr.get())}

    def pg_studio(self):
        if not self.gate():
            return
        self.body.grid_columnconfigure(1, weight=1)
        self.body.grid_rowconfigure(0, weight=1)
        left = ctk.CTkScrollableFrame(self.body, width=380, fg_color=CARD, corner_radius=14,
                                      label_text="Pipeline  (switch steps on / off)",
                                      label_font=("Segoe UI", 13, "bold"), label_fg_color=CARD2)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        for i, (k, label, desc, _) in enumerate(PIPELINE, 1):
            f = ctk.CTkFrame(left, fg_color="transparent")
            f.pack(fill="x", pady=4)
            ctk.CTkSwitch(f, text=f"{i}. {label}", variable=self.step_vars[k], progress_color=ACCENT,
                          font=("Segoe UI", 12, "bold")).pack(anchor="w")
            self.lbl(f, desc, 10, color=MUTED).pack(anchor="w", padx=46)
        self.lbl(left, "Settings", 13, True).pack(anchor="w", pady=(14, 4))
        for text, var, vals in [("Numeric fill", self.num_fill, ["Median", "Mean"]),
                                ("Text fill", self.txt_fill, ["Most frequent", "Unknown"]),
                                ("Outliers", self.out_mode, ["Report only", "Cap (winsorize)", "Remove rows"])]:
            r = ctk.CTkFrame(left, fg_color="transparent")
            r.pack(fill="x", pady=3)
            self.lbl(r, text, 12, color=MUTED, width=90, anchor="w").pack(side="left")
            ctk.CTkOptionMenu(r, values=vals, variable=var, fg_color=CARD2, button_color=ACCENT,
                              width=170).pack(side="right")
        r = ctk.CTkFrame(left, fg_color="transparent")
        r.pack(fill="x", pady=6)
        thr = self.lbl(r, f"Drop columns ≥ {self.drop_thr.get():.0f}% empty", 12, color=MUTED)
        thr.pack(anchor="w")
        ctk.CTkSlider(r, from_=10, to=100, number_of_steps=18, variable=self.drop_thr, progress_color=ACCENT,
                      command=lambda v: thr.configure(text=f"Drop columns ≥ {v:.0f}% empty")).pack(fill="x")
        right = ctk.CTkFrame(self.body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        ctk.CTkButton(right, text="▶   RUN CLEANING PIPELINE", height=50, fg_color=ACCENT,
                      hover_color="#7c3aed", font=("Segoe UI", 15, "bold"),
                      command=self.run_cleaning).pack(fill="x")
        self.w_status = self.lbl(right, "Ready." if self.df is None else "Cleaning finished ✔", 12,
                                 color=MUTED)
        self.w_status.pack(anchor="w", pady=(10, 2))
        self.w_prog = ctk.CTkProgressBar(right, height=12, progress_color=GOOD, fg_color=CARD2)
        self.w_prog.set(1 if self.df is not None else 0)
        self.w_prog.pack(fill="x")
        if self.h_new:
            sc = self.card(right)
            sc.pack(fill="x", pady=12)
            rr = ctk.CTkFrame(sc, fg_color="transparent")
            rr.pack(pady=8)
            Ring(rr, self.h_raw["Overall"], "BEFORE", 120).pack(side="left", padx=10)
            self.lbl(rr, "➜", 28, color=MUTED).pack(side="left")
            Ring(rr, self.h_new["Overall"], "AFTER", 120).pack(side="left", padx=10)
            self.lbl(rr, f"{len(self.audit)} operations\n{sum(a['Cells'] for a in self.audit):,} cells touched\n"
                         f"{self.st_raw['Records'] - self.st_new['Records']:,} rows removed", 12,
                     color=TXT, justify="left").pack(side="left", padx=20)
        self.w_log = ctk.CTkTextbox(right, fg_color=CARD, font=("Consolas", 12), corner_radius=14)
        self.w_log.pack(fill="both", expand=True, pady=(12, 0))
        self.w_log.insert("end", "\n".join(self.run_log) or "Live log appears here when you run the pipeline.")

    def run_cleaning(self):
        o = self.options()
        steps = [(k, lbl, fn) for k, lbl, _, fn in PIPELINE if self.step_vars[k].get()]
        if not steps:
            messagebox.showinfo("Nothing to do", "Switch on at least one step.")
            return
        df, audit, t0 = self.raw.copy(), [], time.time()
        self.w_log.delete("1.0", "end")
        self.run_log = []
        try:
            for i, (k, label, fn) in enumerate(steps):
                self.w_status.configure(text=f"Step {i + 1}/{len(steps)}: {label}…", text_color=WARN)
                self.w_prog.set(i / len(steps))
                self.update()
                n0 = len(audit)
                df = fn(df, o, audit)
                line = f"✔ {label:<34} {len(audit) - n0:>3} changes logged"
                self.run_log.append(line)
                self.w_log.insert("end", line + "\n")
                self.w_log.see("end")
                self.update()
        except Exception as e:
            messagebox.showerror("Cleaning failed", f"{type(e).__name__}: {e}")
            return
        self.w_status.configure(text="Analyzing cleaned data…", text_color=WARN)
        self.update()
        self.df, self.audit = df, audit
        self.prof_new, self.st_new, self.h_new = analyze(df)
        self.outl = outlier_stats(df)
        self.run_log.append(f"\n🎉 Done in {time.time() - t0:.1f}s. Health {self.h_raw['Overall']:.0f} → "
                            f"{self.h_new['Overall']:.0f}")
        self.show("Cleaning Studio")
        self.toast("Cleaning finished ✔")

    # =============================================================== CHANGE LOG
    def pg_log(self):
        if not self.gate(True):
            return
        a = pd.DataFrame(self.audit, columns=["Step", "Column", "What changed", "Cells"])
        bar = ctk.CTkFrame(self.body, fg_color="transparent")
        bar.pack(fill="x", pady=(0, 8))
        self.lbl(bar, f"{len(a)} operations · {int(a['Cells'].sum()):,} cells affected" if len(a) else
                 "The pipeline made no changes.", 13, True).pack(side="left")
        holder = self.card(self.body)
        holder.pack(fill="both", expand=True)
        steps = ["All steps"] + sorted(a["Step"].unique().tolist())

        def draw(*_):
            for w in holder.winfo_children():
                w.destroy()
            d = a if flt.get() == "All steps" else a[a["Step"] == flt.get()]
            self.table(holder, d, max_rows=1000, height=18).pack(fill="both", expand=True, padx=8, pady=8)

        flt = ctk.StringVar(value="All steps")
        ctk.CTkOptionMenu(bar, values=steps, variable=flt, command=draw, fg_color=CARD2,
                          button_color=ACCENT).pack(side="right")
        draw()

    # ============================================================ BEFORE / AFTER
    def pg_compare(self):
        if not self.gate(True):
            return
        sc = ctk.CTkScrollableFrame(self.body, fg_color="transparent")
        sc.pack(fill="both", expand=True)
        top = ctk.CTkFrame(sc, fg_color="transparent")
        top.pack(fill="x")
        top.grid_columnconfigure(0, weight=1, uniform="cmp")
        top.grid_columnconfigure(1, weight=1, uniform="cmp")
        mc = self.card(top)
        mc.grid(row=0, column=0, padx=6, sticky="nsew")
        for j, h in enumerate(["Measure", "Before", "After", "Change"]):
            mc.grid_columnconfigure(j, weight=1)
            self.lbl(mc, h, 11, True, ACCENT).grid(row=0, column=j, padx=14, pady=(14, 6), sticky="w")
        rows = [(k, self.st_raw[k], self.st_new[k], k in ("Missing cells", "Duplicate rows")) for k in self.st_raw]
        rows.append(("Health score", round(self.h_raw["Overall"]), round(self.h_new["Overall"]), None))
        for i, (k, b, af, lower_better) in enumerate(rows, 1):
            d = af - b
            good = (d < 0) if lower_better else (d > 0) if lower_better is None else None
            col = GOOD if good else MUTED if good is None and d == 0 else BAD if good is False else TXT
            self.lbl(mc, k, 12).grid(row=i, column=0, padx=14, pady=7, sticky="w")
            self.lbl(mc, f"{b:,}", 12, color=BAD if lower_better else TXT).grid(row=i, column=1, padx=14, sticky="w")
            self.lbl(mc, f"{af:,}", 12, True, GOOD if lower_better is not False else TXT).grid(
                row=i, column=2, padx=14, sticky="w")
            self.lbl(mc, ("▲ +" if d > 0 else "▼ ") + f"{d:,}" if d else "—", 12, True, col).grid(
                row=i, column=3, padx=14, sticky="w")
        cc = self.card(top)
        cc.grid(row=0, column=1, padx=6, sticky="nsew")
        fig = mkfig(4.6, 2.9)
        ax = fig.add_subplot(111)
        sty(ax)
        ks = ["Completeness", "Uniqueness", "Validity", "Consistency"]
        y = np.arange(len(ks))
        ax.barh(y - 0.2, [self.h_raw[k] for k in ks], 0.38, color=BAD, label="Before")
        ax.barh(y + 0.2, [self.h_new[k] for k in ks], 0.38, color=GOOD, label="After")
        ax.set_yticks(y)
        ax.set_yticklabels(ks)
        ax.set_xlim(0, 105)
        ax.set_title("Quality scores")
        ax.legend(fontsize=7, facecolor=CARD, labelcolor=TXT, loc="lower right")
        fig.tight_layout()
        self.embed(fig, cc)
        pr = ctk.CTkFrame(sc, fg_color="transparent")
        pr.pack(fill="x", pady=12)
        for j, (name, d) in enumerate([("RAW DATA (before)", self.raw), ("CLEANED DATA (after)", self.df)]):
            pr.grid_columnconfigure(j, weight=1)
            c = self.card(pr)
            c.grid(row=0, column=j, padx=6, sticky="nsew")
            self.lbl(c, name, 12, True, BAD if j == 0 else GOOD).pack(anchor="w", padx=14, pady=(12, 4))
            self.table(c, d, max_rows=12, max_cols=7, height=9).pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # ================================================================ EXPLORER
    def pg_explorer(self):
        if not self.gate():
            return
        bar = ctk.CTkFrame(self.body, fg_color="transparent")
        bar.pack(fill="x", pady=(0, 8))
        src = ctk.StringVar(value="Cleaned" if self.df is not None else "Raw")
        q = ctk.StringVar()
        col = ctk.StringVar(value="All columns")
        info = self.lbl(bar, "", 11, color=MUTED)
        holder = self.card(self.body)
        holder.pack(fill="both", expand=True)

        def data():
            return self.df if (src.get() == "Cleaned" and self.df is not None) else self.raw

        def draw(*_):
            d = data()
            for w in holder.winfo_children():
                w.destroy()
            sub, note = d, ""
            text = q.get().strip()
            if text:
                part = d.head(30000)
                if col.get() == "All columns" or col.get() not in part.columns:
                    m = part.astype(str).apply(lambda s: s.str.contains(text, case=False, na=False, regex=False)).any(axis=1)
                else:
                    m = part[col.get()].astype(str).str.contains(text, case=False, na=False, regex=False)
                sub, note = part[m], " (searched first 30,000 rows)"
            info.configure(text=f"{min(len(sub), 300):,} of {len(sub):,} matching rows shown{note}")
            self.table(holder, sub, max_rows=300, height=18).pack(fill="both", expand=True, padx=8, pady=8)

        def change_src(v):
            src.set(v)
            cm.configure(values=["All columns"] + [str(c) for c in data().columns])
            col.set("All columns")
            draw()

        seg = ctk.CTkSegmentedButton(bar, values=["Raw", "Cleaned"], command=change_src, selected_color=ACCENT)
        seg.set(src.get())
        seg.pack(side="left")
        e = ctk.CTkEntry(bar, textvariable=q, placeholder_text="🔎 Search any value…", width=260)
        e.pack(side="left", padx=10)
        e.bind("<Return>", draw)
        cm = ctk.CTkOptionMenu(bar, values=["All columns"] + [str(c) for c in data().columns], variable=col,
                               fg_color=CARD2, button_color=ACCENT, width=170, command=draw)
        cm.pack(side="left")
        info.pack(side="right")
        draw()

    # ================================================================ INSIGHTS
    def pg_insights(self):
        if not self.gate(True):
            return
        d = self.df
        nums = [c for c in d.columns if is_num(d[c]) and d[c].nunique() > 10 and not ID_LIKE.search(str(c))]
        cats = [c for c in text_cols(d) if 2 <= d[c].nunique() <= 15]
        bar = ctk.CTkFrame(self.body, fg_color="transparent")
        bar.pack(fill="x", pady=(0, 8))
        sel_n = ctk.StringVar(value=nums[0] if nums else "")
        sel_c = ctk.StringVar(value=cats[0] if cats else "")
        box = self.card(self.body)
        box.pack(fill="both", expand=True)
        state = {}

        def draw(*_):
            for w in box.winfo_children():
                w.destroy()
            fig = mkfig(10, 6)
            a1, a2, a3, a4 = fig.subplots(2, 2).flat
            for ax in (a1, a2, a3, a4):
                sty(ax)
            worst = sorted(self.prof_raw, key=lambda r: -r["Missing %"])[:8]
            a1.barh([str(r["Column"])[:18] for r in worst][::-1], [r["Missing %"] for r in worst][::-1], color=BAD)
            a1.set_title("Most missing data BEFORE cleaning (%)")
            if sel_c.get() in d.columns:
                d[sel_c.get()].value_counts().head(10).plot(kind="bar", ax=a2, color=BLUE)
                a2.set_title(f"Counts: {sel_c.get()}")
                a2.set_xlabel("")
                a2.tick_params(axis="x", rotation=40)
            else:
                a2.set_title("No category column")
            if sel_n.get() in d.columns:
                v = d[sel_n.get()].dropna()
                a3.hist(v, bins=30, color=ACCENT)
                a3.axvline(v.median(), color=GOOD, lw=1.5, label="median")
                a3.legend(fontsize=7, facecolor=CARD, labelcolor=TXT)
                a3.set_title(f"Distribution: {sel_n.get()}")
            else:
                a3.set_title("No numeric column")
            use = [c for c in nums if d[c].std() > 0][:8]
            if len(use) >= 2:
                cr = d[use].corr()
                a4.imshow(cr, cmap="coolwarm", vmin=-1, vmax=1)
                a4.set_xticks(range(len(use)))
                a4.set_yticks(range(len(use)))
                a4.set_xticklabels([str(c)[:10] for c in use], rotation=45, ha="right", fontsize=7)
                a4.set_yticklabels([str(c)[:10] for c in use], fontsize=7)
                for i in range(len(use)):
                    for j in range(len(use)):
                        a4.text(j, i, f"{cr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white")
                a4.set_title("Correlation heatmap")
            else:
                a4.set_title("Not enough numeric columns")
            fig.tight_layout()
            state["cv"] = self.embed(fig, box)

        if cats:
            self.lbl(bar, "Category:", 12, color=MUTED).pack(side="left")
            ctk.CTkOptionMenu(bar, values=cats, variable=sel_c, command=draw, fg_color=CARD2,
                              button_color=ACCENT, width=170).pack(side="left", padx=(6, 18))
        if nums:
            self.lbl(bar, "Number:", 12, color=MUTED).pack(side="left")
            ctk.CTkOptionMenu(bar, values=nums, variable=sel_n, command=draw, fg_color=CARD2,
                              button_color=ACCENT, width=190).pack(side="left", padx=6)
        draw()

    # ================================================================ OUTLIERS
    def pg_outliers(self):
        if not self.gate(True):
            return
        self.body.grid_columnconfigure(1, weight=1)
        self.body.grid_rowconfigure(0, weight=1)
        left = self.card(self.body)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        self.lbl(left, "IQR outlier scan", 14, True).pack(anchor="w", padx=16, pady=(14, 2))
        self.lbl(left, "Values outside Q1 − 1.5×IQR … Q3 + 1.5×IQR", 10, color=MUTED).pack(anchor="w", padx=16)
        if not self.outl:
            self.lbl(left, "No numeric columns to check.", 12, color=MUTED).pack(padx=40, pady=40)
        else:
            t = pd.DataFrame([{"Column": c, "Outliers": d["n"], "% of rows": round(d["pct"], 1),
                               "Low": round(d["lo"], 1), "High": round(d["hi"], 1)} for c, d in self.outl.items()])
            self.table(left, t, height=12).pack(padx=10, pady=10)
        bt = ctk.CTkFrame(left, fg_color="transparent")
        bt.pack(fill="x", padx=14, pady=(0, 14))
        ctk.CTkButton(bt, text="Cap all outliers", fg_color=ACCENT, command=lambda: self.fix_outliers("Cap")).pack(
            fill="x", pady=3)
        ctk.CTkButton(bt, text="Remove outlier rows", fg_color=BAD, hover_color="#e11d48",
                      command=lambda: self.fix_outliers("Remove")).pack(fill="x", pady=3)
        right = self.card(self.body)
        right.grid(row=0, column=1, sticky="nsew")
        if not self.outl:
            return
        sel = ctk.StringVar(value=max(self.outl, key=lambda c: self.outl[c]["n"]))
        fb = ctk.CTkFrame(right, fg_color="transparent")
        fb.pack(fill="x", padx=14, pady=(12, 0))
        self.lbl(fb, "Column:", 12, color=MUTED).pack(side="left")
        area = ctk.CTkFrame(right, fg_color="transparent")

        def draw(*_):
            for w in area.winfo_children():
                w.destroy()
            c, d = sel.get(), self.outl[sel.get()]
            v = self.df[c].dropna()
            lo_z, hi_z = v.quantile([0.005, 0.995])
            vz = v[(v >= min(lo_z, d["lo"])) & (v <= max(hi_z, d["hi"]))]
            fig = mkfig(6, 5)
            a1, a2 = fig.subplots(2, 1).flat
            sty(a1)
            sty(a2)
            a1.boxplot(vz, vert=False, patch_artist=True, boxprops=dict(facecolor=ACCENT),
                       medianprops=dict(color="white"), flierprops=dict(markerfacecolor=BAD, markersize=3))
            a1.set_title(f"{c}: {d['n']:,} outliers ({d['pct']:.1f}%)")
            a2.hist(vz, bins=40, color=BLUE)
            for x in (d["lo"], d["hi"]):
                a2.axvline(x, color=BAD, ls="--", lw=1.2)
            a2.set_title("IQR limits in red (zoomed to 99% of data)")
            fig.tight_layout()
            self.embed(fig, area)

        ctk.CTkOptionMenu(fb, values=list(self.outl), variable=sel, command=draw, fg_color=CARD2,
                          button_color=ACCENT, width=220).pack(side="left", padx=8)
        area.pack(fill="both", expand=True)
        draw()

    def fix_outliers(self, mode):
        if not self.outl or not any(d["n"] for d in self.outl.values()):
            self.toast("No outliers to fix", WARN)
            return
        word = "cap" if mode == "Cap" else "REMOVE rows with"
        if not messagebox.askyesno("Confirm", f"Do you want to {word} IQR outliers in the cleaned data?"):
            return
        self.df = apply_outliers(self.df, mode, self.audit)
        self.prof_new, self.st_new, self.h_new = analyze(self.df)
        self.outl = outlier_stats(self.df)
        self.show("Outliers")
        self.toast("Outliers handled ✔ (see Change Log)")

    # ================================================================== EXPORT
    def pg_export(self):
        if not self.gate(True):
            return
        for icon, title, desc, cmd in [
            ("💾", "Cleaned data (CSV)", "The cleaned dataset as a .csv file", self.ex_csv),
            ("📗", "Cleaned data (Excel)", "Cleaned data plus the change log on a second sheet", self.ex_xlsx),
            ("📄", "Cleaning report (HTML)", "Auto-written report: before/after table, health score, every step", self.ex_report),
            ("🖼️", "Before / After images", "PNG tables and summary, ready to paste into Word", self.ex_images),
            ("📜", "Change log (CSV)", "Full audit trail of every change", self.ex_log),
        ]:
            c = self.card(self.body)
            c.pack(fill="x", pady=5)
            self.lbl(c, icon, 26).pack(side="left", padx=18, pady=14)
            t = ctk.CTkFrame(c, fg_color="transparent")
            t.pack(side="left")
            self.lbl(t, title, 14, True).pack(anchor="w")
            self.lbl(t, desc, 11, color=MUTED).pack(anchor="w")
            ctk.CTkButton(c, text="Save", width=100, fg_color=ACCENT, hover_color="#7c3aed",
                          command=cmd).pack(side="right", padx=18)

    def ex_csv(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="cleaned_data.csv")
        if p:
            self.df.to_csv(p, index=False)
            self.toast("Saved CSV ✔")

    def ex_xlsx(self):
        p = filedialog.asksaveasfilename(defaultextension=".xlsx", initialfile="cleaned_data.xlsx")
        if p:
            try:
                with pd.ExcelWriter(p) as w:
                    self.df.head(1_000_000).to_excel(w, "Cleaned data", index=False)
                    pd.DataFrame(self.audit).to_excel(w, "Change log", index=False)
                self.toast("Saved Excel ✔")
            except Exception as e:
                messagebox.showerror("Excel export failed", str(e))

    def ex_log(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="change_log.csv")
        if p:
            pd.DataFrame(self.audit).to_csv(p, index=False)
            self.toast("Saved change log ✔")

    def ex_report(self):
        p = filedialog.asksaveasfilename(defaultextension=".html", initialfile="cleaning_report.html")
        if not p:
            return
        e = htmllib.escape
        rows = "".join(f"<tr><td>{k}</td><td>{self.st_raw[k]:,}</td><td>{self.st_new[k]:,}</td></tr>" for k in self.st_raw)
        hs = "".join(f"<tr><td>{k}</td><td>{self.h_raw[k]:.0f}</td><td>{self.h_new[k]:.0f}</td></tr>" for k in self.h_raw)
        steps = "".join(f"<tr><td>{e(a['Step'])}</td><td>{e(str(a['Column']))}</td><td>{e(a['What changed'])}</td>"
                        f"<td>{a['Cells']:,}</td></tr>" for a in self.audit)
        css = ("body{font-family:Segoe UI,Arial;max-width:900px;margin:30px auto;color:#222}"
               "h1{color:#6d28d9}table{border-collapse:collapse;width:100%;margin:10px 0 24px}"
               "td,th{border:1px solid #ddd;padding:7px 10px;text-align:left}th{background:#f3e8ff}")
        doc = (f"<html><head><meta charset='utf-8'><title>Cleaning Report</title><style>{css}</style></head><body>"
               f"<h1>Data Cleaning Report</h1><p><b>File:</b> {e(self.file_name)}<br>"
               f"<b>Generated:</b> {datetime.now():%Y-%m-%d %H:%M} by TidyLab</p>"
               f"<h2>Dataset summary</h2><table><tr><th>Measure</th><th>Before</th><th>After</th></tr>{rows}</table>"
               f"<h2>Data health score (0-100)</h2><table><tr><th>Dimension</th><th>Before</th><th>After</th></tr>{hs}</table>"
               f"<h2>Cleaning steps performed</h2><table><tr><th>Step</th><th>Column</th><th>What changed</th>"
               f"<th>Cells</th></tr>{steps}</table></body></html>")
        with open(p, "w", encoding="utf-8") as f:
            f.write(doc)
        self.toast("Report saved ✔")

    def ex_images(self):
        folder = filedialog.askdirectory(title="Choose folder for images")
        if not folder:
            return

        def tbl(d, title, name):
            d = d.head(10).iloc[:, :9]
            fig = Figure(figsize=(max(8, 1.6 * d.shape[1]), 4.2))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.set_title(title, fontsize=13, fontweight="bold")
            t = ax.table(cellText=d.astype(str).apply(lambda c: c.str[:16]).values,
                         colLabels=[str(c)[:16] for c in d.columns], loc="center")
            t.auto_set_font_size(False)
            t.set_fontsize(8)
            t.scale(1, 1.4)
            fig.savefig(os.path.join(folder, name), bbox_inches="tight", dpi=150)

        tbl(self.raw, "Dataset BEFORE cleaning", "01_before_data.png")
        tbl(self.df, "Dataset AFTER cleaning", "02_after_data.png")
        rows = [[k, f"{self.st_raw[k]:,}", f"{self.st_new[k]:,}"] for k in self.st_raw]
        rows.append(["Health score", f"{self.h_raw['Overall']:.0f}", f"{self.h_new['Overall']:.0f}"])
        fig = Figure(figsize=(7, 3))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title("Before vs After Cleaning", fontweight="bold")
        ax.table(cellText=rows, colLabels=["Measure", "Before", "After"], loc="center").scale(1, 1.8)
        fig.savefig(os.path.join(folder, "03_before_after_summary.png"), bbox_inches="tight", dpi=150)
        self.toast("3 images saved ✔")


if __name__ == "__main__":
    App().mainloop()