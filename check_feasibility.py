import os, json, requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv('data/calibration/llm_jury_full_1700_annotated.csv')

def check_fb(exp):
    if not isinstance(exp, str):
        return False
    low = exp.lower()
    return any(e in low for e in ['rate limit', '429', 'failed to evaluate', 'tpd', 'quota exceeded', 'read timed out', 'httpsconnectionpool', 'fallback'])

gem_fb = df['Gemini-3-Flash_explanation'].apply(check_fb)
gpt_fb = df['GPT-OSS-120B_explanation'].apply(check_fb)
qwen_fb = df['Qwen-3.8-27B_explanation'].apply(check_fb)

print(f"Gemini missing: {gem_fb.sum()} pairs")
print(f"Qwen missing: {qwen_fb.sum()} pairs")
print(f"GPT missing: {gpt_fb.sum()} pairs")

# Test Gemini API key
gem_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
print("\nTesting Gemini API...")
if gem_key:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview:generateContent?key={gem_key}"
    resp = requests.post(url, headers={"Content-Type": "application/json"}, json={"contents": [{"parts": [{"text": "ping"}]}], "generationConfig": {"maxOutputTokens": 5}})
    print("Gemini API Status:", resp.status_code)
    if resp.status_code != 200:
        # Try fallback model
        url2 = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent?key={gem_key}"
        resp2 = requests.post(url2, headers={"Content-Type": "application/json"}, json={"contents": [{"parts": [{"text": "ping"}]}], "generationConfig": {"maxOutputTokens": 5}})
        print("Gemini 3.1 flash lite Status:", resp2.status_code)
else:
    print("No Gemini API key found in env!")

# Test Groq Qwen
k1 = os.environ.get('GROQ_API_KEY')
k2 = os.environ.get('GROQ_API_KEY_BACKUP')
print("\nTesting Groq Qwen (qwen/qwen3.8-27b)...")
rq = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {k1}", "Content-Type": "application/json"}, json={"model": "qwen/qwen3.8-27b", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5})
print("Qwen Key 1 Status:", rq.status_code)
if rq.status_code != 200:
    print("  Qwen Key 1 Error:", rq.text[:200])

rq2 = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {k2}", "Content-Type": "application/json"}, json={"model": "qwen/qwen3.8-27b", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5})
print("Qwen Key 2 Status:", rq2.status_code)
if rq2.status_code != 200:
    print("  Qwen Key 2 Error:", rq2.text[:200])

# Test Groq GPT
print("\nTesting Groq GPT (openai/gpt-oss-120b)...")
rg = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {k1}", "Content-Type": "application/json"}, json={"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5})
print("GPT Key 1 Status:", rg.status_code)
if rg.status_code != 200:
    print("  GPT Key 1 Error:", rg.text[:200])

rg2 = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {k2}", "Content-Type": "application/json"}, json={"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5})
print("GPT Key 2 Status:", rg2.status_code)
if rg2.status_code != 200:
    print("  GPT Key 2 Error:", rg2.text[:200])
