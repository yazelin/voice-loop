#!/usr/bin/env python3
"""麥克風 → Whisper（本機 GPU）→ LLM → CosyVoice 講出來 → 播放。

預設用「你自己剛剛講的那句話」當 voice clone 的參考音，STT 的結果當參考文字，
所以不必事先錄樣本，AI 會用你自己的聲音回答你。
想固定成別的聲音就給 --voice。

跑法（必須用 CosyVoice 那個 venv）：
  ~/CosyVoice/.venv/bin/python voice_loop.py
  ~/CosyVoice/.venv/bin/python voice_loop.py --voice ~/cosy-narrator/assets/jinn-tiffy-10s.wav
  ~/CosyVoice/.venv/bin/python voice_loop.py --input some.wav   # 不用麥克風，跑一輪就結束
  python3 voice_loop.py --selfcheck                             # 不載模型的自我檢查
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# 路徑都可以用環境變數覆寫，換機器不必改 code
COSYVOICE = Path(os.environ.get("COSYVOICE_DIR", Path.home() / "CosyVoice"))
MODEL_DIR = Path(os.environ.get("COSYVOICE_MODEL", COSYVOICE / "pretrained_models/Fun-CosyVoice3-0.5B"))
WHISPER_CLI = Path(os.environ.get("WHISPER_CLI", Path.home() / ".mori/bin/whisper-cli"))
WHISPER_MODEL = Path(os.environ.get("WHISPER_MODEL", Path.home() / ".mori/models/ggml-small.bin"))
WORK = Path.home() / "voice-loop/tmp"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = {"llmshare": "deepseek-v4-flash:0731", "groq": "openai/gpt-oss-120b"}

PAREN_RE = re.compile(r"[(（\[][^)）\]]{0,6}[)）\]]")


def clean_stt(text):
    """whisper 常回（音樂）（掌聲）這種註記與前後空白，清掉。"""
    return PAREN_RE.sub("", text).strip()


def record(out_wav, device):
    """按 Enter 開始、再按 Enter 停止。回傳是否錄到東西。"""
    input("按 Enter 開始錄音…")
    cmd = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", str(out_wav)]
    if device:
        cmd[1:1] = ["-D", device]
    proc = subprocess.Popen(cmd)
    input("錄音中… 再按 Enter 停止。")
    proc.terminate()
    proc.wait()
    return out_wav.exists() and out_wav.stat().st_size > 16000  # 約 0.5 秒以上


def transcribe(wav):
    env = {**os.environ, "LD_LIBRARY_PATH": str(WHISPER_CLI.parent)}
    out = subprocess.run(
        [str(WHISPER_CLI), "-m", str(WHISPER_MODEL), "-l", "zh", "-nt", "-np",
         "--prompt", "以下是繁體中文的句子。", "-f", str(wav)],
        capture_output=True, text=True, env=env,
    ).stdout
    return clean_stt(out)


def build_prompt(question, max_chars):
    return (f"用正體中文口語回答，{max_chars} 個字以內，只回答問題本身，"
            f"不要開場白、不要條列、不要 emoji。問題：{question}")


def ask_llmshare(prompt, model):
    r = subprocess.run(["llmshare", "raw", model, prompt], capture_output=True, text=True)
    if r.returncode != 0:
        return f"抱歉，模型沒有回應。{r.stderr.strip()[:60]}"
    return r.stdout


def ask_groq(prompt, model):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return "沒有設 GROQ_API_KEY，問不到 Groq。"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 400,
        "reasoning_effort": "low",  # gpt-oss 會先想再答，想太久就失去用 Groq 的意義
    }).encode()
    req = urllib.request.Request(
        GROQ_URL, body,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
         # Cloudflare 會擋掉 urllib 的預設 User-Agent，回 403 error code 1010
         "User-Agent": "voice-loop/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        return f"抱歉，Groq 回了 {e.code}。{e.read()[:80].decode('utf-8', 'replace')}"
    except OSError as e:
        return f"抱歉，連不上 Groq。{e}"
    return data["choices"][0]["message"].get("content") or ""


def ask_llm(question, backend, model, max_chars):
    prompt = build_prompt(question, max_chars)
    raw = ask_groq(prompt, model) if backend == "groq" else ask_llmshare(prompt, model)
    return " ".join(raw.split())[:max_chars * 2] or "我不知道要怎麼回答這個。"


def selfcheck():
    assert "60 個字以內" in build_prompt("在嗎？", 60) and "在嗎？" in build_prompt("在嗎？", 60)
    assert clean_stt(" （音樂） 今天天氣如何？ ") == "今天天氣如何？"
    assert clean_stt("(掌聲)好的") == "好的"
    assert clean_stt("這是（一個很長很長很長的東西）保留") == "這是（一個很長很長很長的東西）保留"
    print("selfcheck ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llmshare", "groq"], default="llmshare",
                    help="回答用哪個 LLM 後端。groq 要 GROQ_API_KEY")
    ap.add_argument("--model", help="不給就用該後端的預設："
                                    f"llmshare={DEFAULT_MODEL['llmshare']}、groq={DEFAULT_MODEL['groq']}")
    ap.add_argument("--device", default="", help="arecord 裝置，如 plughw:1,0；留空用系統預設")
    ap.add_argument("--max-chars", type=int, default=60, help="回答字數上限")
    ap.add_argument("--voice", help="固定的 clone 參考音 wav；不給就用你每次講的那句")
    ap.add_argument("--voice-text", help="參考音的逐字稿；不給就用 whisper 轉一次")
    ap.add_argument("--input", help="拿現成 wav 代替麥克風，跑一輪就結束")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    model = args.model or DEFAULT_MODEL[args.backend]

    WORK.mkdir(parents=True, exist_ok=True)
    import logging
    # CosyVoice 的 INFO 與 tqdm 進度條會把畫面洗掉；「合成文字比參考文字短」那個
    # WARNING 在這個用法下必然會出現（回答就是比參考音短），一起壓掉
    logging.basicConfig(level=logging.ERROR, force=True)
    os.environ.setdefault("TQDM_DISABLE", "1")
    sys.path += [str(COSYVOICE / "third_party/Matcha-TTS"), str(COSYVOICE)]
    import onnxruntime
    # ponytail: 8G 顯存塞不下 onnx CUDA EP ＋ CosyVoice，把 onnx 逼回 CPU（它很小，慢不了多少）。
    # 換大顯存的機器可以刪掉這三行
    _orig = onnxruntime.InferenceSession
    onnxruntime.InferenceSession = lambda *a, **k: _orig(*a, **{**k, "providers": ["CPUExecutionProvider"]})
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel

    ref_wav = ref_text = None
    if args.voice:
        ref_wav = str(Path(args.voice).expanduser())
        ref_text = args.voice_text or transcribe(Path(ref_wav))
        print(f"參考聲音：{ref_wav}\n參考文字：{ref_text}")

    print("載入 CosyVoice…", flush=True)
    t0 = time.time()
    cv = AutoModel(model_dir=str(MODEL_DIR), fp16=True)
    print(f"好了（{time.time() - t0:.0f}s）。{args.backend} / {model}　Ctrl-C 離開\n", flush=True)

    wav = Path(args.input) if args.input else WORK / "in.wav"
    out = WORK / "out.wav"
    while True:
        if not args.input and not record(wav, args.device):
            print("沒錄到聲音，再試一次。\n")
            continue

        t0 = time.time()
        heard = transcribe(wav)
        print(f"你說：{heard}　（{time.time() - t0:.1f}s）", flush=True)
        if not heard:
            print("聽不出內容，再試一次。\n")
            if args.input:
                return
            continue

        t0 = time.time()
        answer = ask_llm(heard, args.backend, model, args.max_chars)
        print(f"回答：{answer}　（{time.time() - t0:.1f}s）", flush=True)

        # ponytail: 整段一次合成,不切句。回答本來就只有幾十個字,
        # 切得越碎離參考文字越遠,CosyVoice 的 clone 品質越差
        t0 = time.time()
        pieces = [
            j["tts_speech"]
            for j in cv.inference_zero_shot(
                answer,
                f"You are a helpful assistant.<|endofprompt|>{ref_text or heard}",
                ref_wav or str(wav),
                stream=False,
            )
        ]
        torchaudio.save(str(out), torch.cat(pieces, dim=1), cv.sample_rate)
        print(f"合成 {time.time() - t0:.1f}s，播放中…\n", flush=True)
        subprocess.run(["paplay", str(out)])
        if args.input:
            return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
