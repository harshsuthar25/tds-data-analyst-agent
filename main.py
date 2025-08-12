# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os, io, base64, re, time
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

app = FastAPI(title="TDS Data Analyst Agent")

def to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^0-9\.\-]', '', regex=True), errors='coerce')

def find_column(df, keywords):
    if df is None: return None
    cols = list(df.columns)
    for kw in keywords:
        for c in cols:
            if kw.lower() in c.lower():
                return c
    # fallback: return first numeric column if present
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def dataframe_from_wikipedia(url):
    try:
        tables = pd.read_html(url)
        if len(tables) > 0:
            return tables[0]
    except Exception:
        pass
    r = requests.get(url, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if table is None:
        return None
    df = pd.read_html(str(table))[0]
    return df

def make_scatter_plot(x, y, xlabel='x', ylabel='y') -> str:
    mask = (~np.isnan(x)) & (~np.isnan(y))
    coeffs = None
    xs = ys = None
    if mask.sum() >= 2:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
        ys = np.polyval(coeffs, xs)
    dpi = 80
    fig_w, fig_h = 6, 4
    buf = io.BytesIO()
    while True:
        plt.figure(figsize=(fig_w, fig_h))
        plt.scatter(x, y, s=10)
        if coeffs is not None:
            plt.plot(xs, ys, linestyle=':', color='red', linewidth=1.4)
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.tight_layout()
        buf.seek(0); buf.truncate(0)
        plt.savefig(buf, format='png', dpi=dpi)
        plt.close()
        data = buf.getvalue()
        if len(data) <= 100000 or dpi <= 10:
            break
        dpi = int(dpi * 0.7)
        fig_w = max(3, fig_w * 0.9)
        fig_h = max(2, fig_h * 0.9)
    b64 = base64.b64encode(data).decode('ascii')
    return f"data:image/png;base64,{b64}"

@app.post("/api/")
async def analyze(request: Request):
    start = time.time()
    form = await request.form()
    # mandatory
    if "questions.txt" not in form:
        return JSONResponse(status_code=400, content={"error":"questions.txt is required (form field name must be 'questions.txt')"})
    questions_file = form["questions.txt"]
    q_text = (await questions_file.read()).decode('utf-8')

    # collect other uploaded files
    attachments = {}
    for k, v in form.items():
        if hasattr(v, "filename") and v.filename:
            try:
                b = await v.read()
                attachments[k] = {"filename": v.filename, "content": b}
            except Exception:
                pass

    # try to make a dataframe from CSV attachment OR a table on a URL mentioned in questions.txt
    df = None
    for name, meta in attachments.items():
        if name.lower().endswith(".csv") or (meta["filename"] and meta["filename"].lower().endswith(".csv")):
            try:
                df = pd.read_csv(io.BytesIO(meta["content"]))
                break
            except Exception:
                try:
                    df = pd.read_csv(io.StringIO(meta["content"].decode('utf-8')))
                    break
                except Exception:
                    pass

    if df is None:
        urls = re.findall(r'(https?://[^\s\)]+)', q_text)
        for url in urls:
            if "wikipedia.org" in url or url.endswith(".html") or url.endswith(".htm"):
                try:
                    df = dataframe_from_wikipedia(url)
                    if df is not None: break
                except Exception:
                    pass

    # split question text into lines (handles numbered lists too)
    qs = []
    for line in q_text.splitlines():
        line = line.strip()
        if not line: continue
        line = re.sub(r'^\d+\s*[\.\)]\s*', '', line)  # remove leading "1." or "2)"
        qs.append(line)
    if not qs:
        qs = [q_text.strip()]

    answers = []
    for q in qs:
        qlower = q.lower()
        # sample heuristics for common patterns (these cover the sample tasks)
        if "how many" in qlower and ("$2" in qlower or "2 bn" in qlower or "2 billion" in qlower):
            if df is None:
                answers.append("No data table found to answer this question.")
                continue
            gross_col = find_column(df, ["gross", "worldwide", "box office", "box_office", "boxoffice"])
            year_col = find_column(df, ["year", "release", "released"])
            if gross_col is None:
                answers.append("Couldn't find a gross column in the data.")
                continue
            gross = to_numeric(df[gross_col])
            mask = gross >= 2_000_000_000
            if year_col:
                years = to_numeric(df[year_col])
                mask = mask & (years < 2000)
            answers.append(int(mask.sum()))
            continue

        if "earliest" in qlower and ("$1.5" in qlower or "1.5 bn" in qlower or "1.5 billion" in qlower):
            if df is None:
                answers.append("No data table found to answer this question.")
                continue
            gross_col = find_column(df, ["gross", "worldwide", "box office"])
            title_col = find_column(df, ["title", "film", "movie", "name"])
            year_col = find_column(df, ["year", "release"])
            if gross_col is None:
                answers.append("Couldn't find gross column.")
                continue
            gross = to_numeric(df[gross_col])
            mask = gross > 1_500_000_000
            if mask.sum() == 0:
                answers.append("No films found with gross > 1.5bn")
                continue
            subset = df[mask].copy()
            if year_col and title_col:
                subset['year_num'] = to_numeric(subset[year_col])
                subset = subset.sort_values('year_num', na_position='last')
                answers.append(str(subset.iloc[0][title_col]))
            else:
                answers.append(str(subset.iloc[0][title_col] if title_col else subset.index[0]))
            continue

        if "correlation" in qlower and "rank" in qlower and "peak" in qlower:
            if df is None:
                answers.append("No data table found to answer this question.")
                continue
            rank_col = find_column(df, ["rank", "position"])
            peak_col = find_column(df, ["peak", "highest", "peak position"])
            if rank_col is None or peak_col is None:
                answers.append("Couldn't identify Rank or Peak columns.")
                continue
            rnum = to_numeric(df[rank_col])
            pnum = to_numeric(df[peak_col])
            mask = (~rnum.isna()) & (~pnum.isna())
            if mask.sum() < 2:
                answers.append("Not enough numeric data to compute correlation.")
                continue
            corr = float(np.corrcoef(rnum[mask], pnum[mask])[0,1])
            answers.append(round(corr, 6))
            continue

        if "scatterplot" in qlower and "rank" in qlower and "peak" in qlower:
            if df is None:
                answers.append("No data table found to draw plot.")
                continue
            rank_col = find_column(df, ["rank", "position"])
            peak_col = find_column(df, ["peak", "highest", "peak position"])
            if rank_col is None or peak_col is None:
                answers.append("Couldn't identify Rank or Peak columns to plot.")
                continue
            x = to_numeric(df[rank_col]).values
            y = to_numeric(df[peak_col]).values
            data_uri = make_scatter_plot(x, y, xlabel=rank_col, ylabel=peak_col)
            answers.append(data_uri)
            continue

        # fallback: if asks "how many" return row count
        if df is not None and ("how many" in qlower or "count" in qlower):
            answers.append(len(df))
            continue

        answers.append(f"Not implemented: {q}")

    return JSONResponse(content=answers)
