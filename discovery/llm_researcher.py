import json
import os
import requests
from core_kernel.db import get_recent_failures, get_today_quota, record_quota
from discovery.dsl import validate_spec
from discovery.quant_search import evaluate_and_register_spec

GEMINI_PRIORITY = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest", "gemini-flash-lite-latest"]
GROQ_PRIORITY = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]

def _pick_gemini_model(key: str) -> str | None:
    r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}", timeout=15)
    if r.status_code != 200:
        return None
    avail = {m["name"].split("/")[-1] for m in r.json().get("models", [])}
    for m in GEMINI_PRIORITY:
        if m in avail:
            return m
    return next((m for m in avail if "flash" in m and "tts" not in m and "image" not in m), None)

def _pick_groq_model(key: str) -> str | None:
    r = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=15)
    if r.status_code != 200:
        return None
    avail = {m["id"] for m in r.json().get("data", [])}
    for m in GROQ_PRIORITY:
        if m in avail:
            return m
    return next((m for m in avail if not any(x in m for x in ("whisper", "guard", "orpheus", "allam"))), None)

def _call_gemini_json(prompt: str) -> dict | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key or get_today_quota("gemini_calls") >= 6:
        return None
    model = _pick_gemini_model(key)
    if not model:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0.5}}
    r = requests.post(url, json=payload, timeout=30)
    record_quota("gemini_calls", calls=1, status=f"HTTP_{r.status_code}")
    if r.status_code != 200:
        return None
    return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])

def _call_groq_json(prompt: str) -> dict | None:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key or get_today_quota("groq_calls") >= 10:
        return None
    model = _pick_groq_model(key)
    if not model:
        return None
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "temperature": 0.3},
        timeout=30,
    )
    record_quota("groq_calls", calls=1, status=f"HTTP_{r.status_code}")
    if r.status_code != 200:
        return None
    return json.loads(r.json()["choices"][0]["message"]["content"])

def _call_openrouter_free_json(prompt: str) -> dict | None:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or get_today_quota("openrouter_calls") >= 10:
        return None
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": "openrouter/free", "messages": [{"role": "user", "content": prompt + "\nReturn valid JSON only."}], "temperature": 0.3},
        timeout=35,
    )
    record_quota("openrouter_calls", calls=1, status=f"HTTP_{r.status_code}")
    if r.status_code != 200:
        return None
    txt = r.json()["choices"][0]["message"]["content"].strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1].lstrip("json").strip()
    return json.loads(txt)

def _safe_llm_json(prompt: str, prefer: str = "gemini") -> dict | None:
    order = [_call_gemini_json, _call_groq_json, _call_openrouter_free_json] if prefer == "gemini" else [_call_groq_json, _call_gemini_json, _call_openrouter_free_json]
    for fn in order:
        try:
            res = fn(prompt)
            if isinstance(res, dict):
                return res
        except Exception:
            continue
    return None

def run_llm_research_cycle(ohlcv: dict, macro_df, sec_df, fred_latest: dict, regime_tag: str, n_trials_start: int) -> dict:
    closes, dollar_vol = ohlcv["close"], ohlcv["dollar_vol"]
    ret_20 = closes.pct_change(20, fill_method=None).iloc[-1].dropna()
    top_win = ret_20.nlargest(5).round(3).to_dict()
    top_lose = ret_20.nsmallest(5).round(3).to_dict()
    failures = get_recent_failures(8)

    research_prompt = f"""You are a ruthless quantitative alpha researcher maximizing out-of-sample geometric wealth across ALL regimes (including VIX > 30 selloffs).
Current regime: {regime_tag}. Macro: {json.dumps(fred_latest)}.
20d winners: {json.dumps(top_win)}. Losers: {json.dumps(top_lose)}.
Avoid repeating recent failures: {json.dumps(failures[:5])}.
IMPORTANT: Long-only equity momentum just lost -31% in the last 63 days. Propose 3 diverse hypotheses that can make positive money in BOTH bull markets AND VIX>30 selloffs (using macro_rotator with inverse/bond/gold/T-bill ETFs, pairs_reversion, or long/short ts_momentum).
Return JSON: {{"hypotheses": [{{"rationale": "short reason", "spec": <DSL_JSON>}}]}}
Allowed DSL kinds:
1) {{"kind":"cs_momentum","pool":"all|stocks|etfs","lookback":20..126,"skip":0..10,"top_pct":0.1..0.25,"allow_short":true,"max_pos_weight":0.2}}
2) {{"kind":"ts_momentum","pool":"etfs|all","lookback":15..90,"vol_window":15..45,"allow_short":true,"max_pos_weight":0.3}}
3) {{"kind":"mean_reversion","pool":"all|stocks","lookback":2..8,"trend_window":50..200,"top_pct":0.1..0.2,"allow_short":true,"max_pos_weight":0.2}}
4) {{"kind":"macro_rotator","vix_thresh":19..28,"curve_thresh":-0.2..0.2,"mom_window":15..63,"risk_on_assets":["QQQ","QLD","SPY","SMH","NVDA"],"risk_off_assets":["TLT","GLD","UUP","SHY","SH","PSQ","QID"],"max_pos_weight":0.5}}
5) {{"kind":"pairs_reversion","lookback":15..60,"z_entry":1.2..2.0,"pairs":[["SMH","QQQ"],["XLE","USO"],["GLD","SLV"],["HYG","LQD"]],"max_pos_weight":0.25}}"""

    prop = _safe_llm_json(research_prompt, prefer="gemini")
    if not prop or not prop.get("hypotheses"):
        return {"status": "SKIPPED_OR_QUOTA_GUARD", "evaluated": []}

    critic_prompt = f"""You are a quantitative critic. Only reject hypotheses that are exact duplicates of recent failures or structurally invalid. Do NOT reject macro_rotator or pairs_reversion just for listing ETFs—let the backtest kernel decide.
Recent failures: {json.dumps(failures[:5])}. Proposed: {json.dumps(prop)}.
Return JSON: {{"approved_indices": [0, 1, 2], "critique": "brief summary"}}"""
    critique = _safe_llm_json(critic_prompt, prefer="groq")
    approved = critique.get("approved_indices", [0, 1, 2]) if critique else [0, 1, 2]

    evaluated = []
    n_trials = n_trials_start
    for idx in approved:
        if not isinstance(idx, int) or idx < 0 or idx >= len(prop["hypotheses"]):
            continue
        spec = prop["hypotheses"][idx].get("spec", {})
        if not validate_spec(spec):
            continue
        n_trials += 1
        _, res = evaluate_and_register_spec(spec, "llm_ai", closes, dollar_vol, macro_df, sec_df, n_trials, regime_tag)
        res["rationale"] = prop["hypotheses"][idx].get("rationale", "")
        evaluated.append(res)
    return {"status": "OK", "critique": (critique or {}).get("critique", ""), "evaluated": evaluated}
