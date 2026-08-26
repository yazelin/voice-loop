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
# mori 的共享 whisper 服務（契約 §6 / §11）：descriptor 固定路徑，supervisor 負責起停
WHISPER_DESCRIPTOR = Path.home() / ".mori/whisper-server.json"
WHISPER_SUPERVISOR = Path.home() / ".mori/bin/mori-whisper-serve"
# groq 與 local 都是 OpenAI 相容端點，差在網址跟要不要金鑰
LLM_URL = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "local": "http://127.0.0.1:8080/v1/chat/completions",
}
DEFAULT_MODEL = {
    "llmshare": "deepseek-v4-flash:0731",
    "groq": "openai/gpt-oss-120b",
    "local": "qwen3.5-4b",  # llama-server 只載一個模型，這個名字只是標籤
}

PAREN_RE = re.compile(r"[(（\[][^)）\]]{0,6}[)）\]]")
# 餵給 whisper 的 initial prompt。錄到靜音時它會把這句原樣吐回來，要當成沒聽到
STT_HINT = "以下是繁體中文的句子。"


def clean_stt(text):
    """whisper 常回（音樂）（掌聲）這種註記與前後空白，清掉；prompt 回音當成沒聽到。"""
    text = PAREN_RE.sub("", text).strip()
    return "" if text and text in STT_HINT else text


def record(out_wav, device):
    """開始錄，按 Enter 停止。回傳是否錄到東西。"""
    cmd = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", str(out_wav)]
    if device:
        cmd[1:1] = ["-D", device]
    proc = subprocess.Popen(cmd)
    input("錄音中… 再按 Enter 停止。")
    proc.terminate()
    proc.wait()
    return out_wav.exists() and out_wav.stat().st_size > 16000  # 約 0.5 秒以上


def find_whisper_server():
    """讀 mori 的 descriptor（契約 §6）。模型已經在顯存裡，借用它我們就不必再載一份。
    descriptor 可能是舊的，所以順便確認 pid 還活著。"""
    try:
        d = json.loads(WHISPER_DESCRIPTOR.read_text(encoding="utf-8"))
        os.kill(d["pid"], 0)
    except (OSError, ValueError, KeyError):
        return None
    return f"http://{d['host']}:{d['port']}{d.get('inference_path', '/inference')}"


def ensure_whisper_server(timeout=20):
    """沒在跑就叫 mori 的 supervisor 起一份。它閒置 10 分鐘會自關（DEFAULT_IDLE_SECS=600），
    所以聊天空檔久一點就得再喚醒一次。--ensure 是冪等的，重複呼叫沒關係。"""
    url = find_whisper_server()
    if url or not WHISPER_SUPERVISOR.is_file():
        return url
    try:
        subprocess.run([str(WHISPER_SUPERVISOR), "--ensure"],
                       capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    deadline = time.time() + timeout
    while time.time() < deadline:      # 冷啟動載 small 模型大約 6 秒
        url = find_whisper_server()
        if url:
            return url
        time.sleep(0.5)
    return None


def _via_server(wav, server):
    """(文字, 成功與否)。空字串但成功 = 真的沒聽到內容。"""
    r = subprocess.run(
        ["curl", "-s", "--max-time", "60", "-F", f"file=@{wav}", "-F", "language=zh",
         "-F", "response_format=json", "-F", f"prompt={STT_HINT}", server],
        capture_output=True, text=True,
    )
    try:
        return clean_stt(" ".join(json.loads(r.stdout)["text"].split())), True
    except (ValueError, KeyError):
        return "", False


def transcribe(wav, stt=None):
    """stt 是可變的 {"url": ...}；server 中途被關掉就自己重找，找不到退回 whisper-cli。"""
    if stt and stt.get("url"):
        text, ok = _via_server(wav, stt["url"])
        if ok:
            return text
        fresh = ensure_whisper_server()        # 多半是閒置 10 分鐘自關了，喚醒它
        if fresh:
            text, ok = _via_server(wav, fresh)
            if ok:
                stt["url"] = fresh
                print(f"whisper-server 換到 {fresh}", flush=True)
                return text
        stt["url"] = None
        print("叫不動 mori 的 whisper-server，改用本機 whisper-cli。"
              "它要多吃約 900 MiB 顯存。", flush=True)

    env = {**os.environ, "LD_LIBRARY_PATH": str(WHISPER_CLI.parent)}
    r = subprocess.run(
        [str(WHISPER_CLI), "-m", str(WHISPER_MODEL), "-l", "zh", "-nt", "-np",
         "--prompt", STT_HINT, "-f", str(wav)],
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        # 顯存不夠時 whisper 是 abort，而且會把 GDB backtrace 印到 stdout。
        # 只看 exit code 卻照用 stdout，那串 backtrace 就會被當成使用者講的話送進 LLM。
        print(f"whisper 失敗（exit {r.returncode}）：{r.stderr.strip()[-200:]}", flush=True)
        return ""
    return clean_stt(r.stdout)


def gpu_free_mib():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.split()[0])
    except (ValueError, IndexError):
        return None


def vram_check(stt_is_shared):
    """CosyVoice 載入要 3490 MiB、推論尖峰再多約 900。whisper 自己跑的話還要約 900。
    不夠就先講，免得等 20 秒載完才在合成那一步炸掉。"""
    free = gpu_free_mib()
    if free is None:
        return
    need = 4400 if stt_is_shared else 5300
    print(f"顯存剩 {free} MiB（這一輪大約需要 {need}）")
    if free < need:
        print("  不夠。省顯存的順序：關 rustdesk（約 240）、關幾個瀏覽器分頁、\n"
              "  llama-server 的 -ngl 調小、少接一台外接螢幕。詳見 README「地端 LLM」。")


COMMANDS = {
    ":voice": "換參考聲音，例：:voice assets/jinn-tiffy-10s.wav",
    ":record": "重錄一段當參考聲音",
    ":backend": "換 LLM 後端：:backend groq / local / llmshare",
    ":len": "回答字數上限，例：:len 30",
    ":clear": "清掉對話歷史，重新開始一個話題",
    ":history": "看目前記著哪幾輪",
    ":say": "不錄音，直接打字問，例：:say 今天天氣如何",
    ":help": "看這份清單",
    ":q": "離開",
}


def parse_command(line):
    """回傳 (指令, 參數)；不是指令就回 (None, 原字串)。"""
    line = line.strip()
    if not line.startswith(":"):
        return None, line
    name, _, arg = line.partition(" ")
    return name, arg.strip()


# 上限不是顯存（-c 從 1024 開到 8192 只多 76 MiB），是小模型的長 context 表現。
# 10 輪約 420 tokens，加上輸出上限 400 仍在 -c 1024 之內，所以兩種設定都安全。
HISTORY_TURNS = 10


def build_prompt(question, max_chars, history=()):
    rule = (f"用正體中文口語回答，{max_chars} 個字以內，只回答問題本身，"
            f"不要開場白、不要條列、不要 emoji。")
    if not history:
        return rule + f"問題：{question}"
    past = "\n".join(f"我：{q}\n你：{a}" for q, a in history[-HISTORY_TURNS:])
    return (f"{rule}下面是我們剛才的對話，接著回答最後那個問題，"
            f"可以延續前面的話題。\n\n{past}\n我：{question}\n你：")


def ask_llmshare(prompt, model):
    r = subprocess.run(["llmshare", "raw", model, prompt], capture_output=True, text=True)
    if r.returncode != 0:
        return f"抱歉，模型沒有回應。{r.stderr.strip()[:60]}"
    return r.stdout


def ask_openai_compatible(prompt, model, url, key=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 400,
        # gpt-oss / Qwen3 都會先想再答，想太久就失去意義。兩邊各認一個欄位，
        # 不認得的那個會被忽略
        "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json",
               # Cloudflare 會擋掉 urllib 的預設 User-Agent，回 403 error code 1010
               "User-Agent": "voice-loop/1.0"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, json.dumps(payload).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        return f"抱歉，{url} 回了 {e.code}。{e.read()[:80].decode('utf-8', 'replace')}"
    except OSError as e:
        return f"抱歉，連不上 {url}。{e}"
    return data["choices"][0]["message"].get("content") or ""


def ask_llm(question, backend, model, max_chars, url=None, history=()):
    prompt = build_prompt(question, max_chars, history)
    if backend == "llmshare":
        raw = ask_llmshare(prompt, model)
    elif backend == "groq":
        key = os.environ.get("GROQ_API_KEY")
        raw = ask_openai_compatible(prompt, model, url or LLM_URL["groq"], key) if key \
            else "沒有設 GROQ_API_KEY，問不到 Groq。"
    else:
        raw = ask_openai_compatible(prompt, model, url or LLM_URL["local"])
    return " ".join(raw.split())[:max_chars * 2] or "我不知道要怎麼回答這個。"


def selfcheck():
    assert "60 個字以內" in build_prompt("在嗎？", 60) and "在嗎？" in build_prompt("在嗎？", 60)
    assert clean_stt(" （音樂） 今天天氣如何？ ") == "今天天氣如何？"
    assert clean_stt("(掌聲)好的") == "好的"
    assert clean_stt("這是（一個很長很長很長的東西）保留") == "這是（一個很長很長很長的東西）保留"
    assert clean_stt("是繁體中文的句子。") == ""      # 錄到靜音時 whisper 的 prompt 回音
    assert clean_stt(STT_HINT) == ""
    assert clean_stt("以下是繁體中文的句子。真的嗎？") == "以下是繁體中文的句子。真的嗎？"
    p0 = build_prompt("在嗎？", 60)
    assert "問題：在嗎？" in p0 and "剛才的對話" not in p0
    p1 = build_prompt("那再說一次", 60, [("天空為什麼藍", "因為散射"), ("那海呢", "反射天空")])
    assert "我：天空為什麼藍" in p1 and "你：因為散射" in p1 and p1.endswith("我：那再說一次\n你：")
    long_hist = [(f"問{i}", f"答{i}") for i in range(20)]
    kept = build_prompt("x", 60, long_hist)          # 只帶最後 HISTORY_TURNS 輪
    assert f"問{20 - HISTORY_TURNS}" in kept and f"問{20 - HISTORY_TURNS - 1}" not in kept
    assert parse_command("  ") == (None, "")
    assert parse_command("你好") == (None, "你好")
    assert parse_command(":len 30") == (":len", "30")
    assert parse_command(":record") == (":record", "")
    assert parse_command(":say 今天 天氣 如何") == (":say", "今天 天氣 如何")
    print("selfcheck ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["llmshare", "groq", "local"], default="llmshare",
                    help="回答用哪個 LLM 後端。groq 要 GROQ_API_KEY；local 要自己先跑 llama-server")
    ap.add_argument("--llm-url", help=f"覆寫 OpenAI 相容端點，預設 local 是 {LLM_URL['local']}")
    ap.add_argument("--model", help="不給就用該後端的預設："
                                    f"llmshare={DEFAULT_MODEL['llmshare']}、groq={DEFAULT_MODEL['groq']}")
    ap.add_argument("--device", default="", help="arecord 裝置，如 plughw:1,0；留空用系統預設")
    ap.add_argument("--max-chars", type=int, default=60, help="回答字數上限")
    ap.add_argument("--voice", help="固定的 clone 參考音 wav；不給就用你每次講的那句")
    ap.add_argument("--record-voice", nargs="?", const="", metavar="WAV",
                    help="開場先錄一段當這場的參考音，之後每輪都用它。"
                         "給路徑就順便存成檔（連同 .txt 逐字稿），下次直接 --voice 那個檔")
    ap.add_argument("--voice-text", help="參考音的逐字稿；不給就讀旁邊的同名 .txt，"
                                         "沒有的話用 whisper 轉一次")
    ap.add_argument("--input", help="拿現成 wav 代替麥克風，跑一輪就結束")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    model = args.model or DEFAULT_MODEL[args.backend]
    if args.voice and args.record_voice is not None:
        ap.error("--voice 和 --record-voice 只能挑一個")
    if args.record_voice is not None and args.input:
        ap.error("--record-voice 要用麥克風，不能跟 --input 一起用")


    WORK.mkdir(parents=True, exist_ok=True)

    stt = {"url": ensure_whisper_server()}
    if stt["url"]:
        print(f"用 mori 的共享 whisper-server（{stt['url']}），不另外佔顯存。")
    vram_check(bool(stt["url"]))

    # 一定要在 import torch 之前設。顯存剩一千多卻配置不到一百 MiB 就是碎片化，
    # expandable_segments 讓配置器可以擴張既有區段，不必找連續空間。
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    if args.record_voice is not None:
        ref_wav = str(Path(args.record_voice).expanduser()) if args.record_voice else str(WORK / "voice.wav")
        print("先錄一段當這場的參考音：講十到三十秒，自然講話不要唸稿。")
        input("按 Enter 開始錄音…")
        if not record(Path(ref_wav), args.device):
            sys.exit("沒錄到聲音，重跑一次。")
        ref_text = transcribe(Path(ref_wav), stt)
        if not ref_text:
            sys.exit("聽不出參考音的內容，重錄一次（安靜一點、講完整的句子）。")
        print(f"參考文字：{ref_text}")
        if args.record_voice:
            Path(ref_wav).with_suffix(".txt").write_text(ref_text + "\n", encoding="utf-8")
            print(f"已存成 {ref_wav}，下次可以直接 --voice {ref_wav}")
    elif args.voice:
        ref_wav = str(Path(args.voice).expanduser())
        sidecar = Path(ref_wav).with_suffix(".txt")  # 參考音旁邊有同名 .txt 就直接用
        ref_text = args.voice_text or (
            sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else transcribe(Path(ref_wav), stt)
        )
        print(f"參考聲音：{ref_wav}\n參考文字：{ref_text}")


    print("載入 CosyVoice…", flush=True)
    t0 = time.time()
    cv = AutoModel(model_dir=str(MODEL_DIR), fp16=True)
    # 參考音固定的話，它的前處理只要算一次就好（10 秒參考音每輪要 1.4 秒）。
    # 預設模式每輪的參考音都不一樣（就是你剛講那句），沒得快取。
    # 參考音固定的話聲紋只算一次；換聲音時重算一次覆蓋掉，省得重開整支程式
    history = []   # [(問, 答), ...]，只帶最近 HISTORY_TURNS 輪進 prompt
    state = {"wav": ref_wav, "text": ref_text, "spk": "", "backend": args.backend,
             "model": model, "len": args.max_chars}

    def use_reference(wav_path, text):
        state["wav"], state["text"], state["spk"] = wav_path, text, "fixed"
        cv.add_zero_shot_spk(f"You are a helpful assistant.<|endofprompt|>{text}", wav_path, "fixed")

    if ref_wav:
        use_reference(ref_wav, ref_text)
    print(f"好了（{time.time() - t0:.0f}s）。{state['backend']} / {state['model']}"
          f"　輸入 :help 看指令，:q 離開\n", flush=True)

    wav = Path(args.input) if args.input else WORK / "in.wav"
    out = WORK / "out.wav"
    while True:
        typed = ""
        if not args.input:
            name, arg = parse_command(input("按 Enter 錄音，或輸入指令（:help）…"))
            if name == ":q":
                return
            if name == ":help":
                for k, v in COMMANDS.items():
                    print(f"  {k:9s} {v}")
                print()
                continue
            if name == ":clear":
                history.clear()
                print("對話歷史清掉了\n")
                continue
            if name == ":history":
                if not history:
                    print("目前沒有歷史\n")
                for q, a in history[-HISTORY_TURNS:]:
                    print(f"  我：{q}\n  你：{a}")
                print()
                continue
            if name == ":len":
                if arg.isdigit() and int(arg) > 0:
                    state["len"] = int(arg)
                    print(f"回答字數上限改成 {state['len']}\n")
                else:
                    print("要給正整數，例：:len 30\n")
                continue
            if name == ":backend":
                if arg in DEFAULT_MODEL:
                    state["backend"], state["model"] = arg, DEFAULT_MODEL[arg]
                    print(f"後端改成 {arg} / {state['model']}\n")
                else:
                    print(f"只能是 {' / '.join(DEFAULT_MODEL)}\n")
                continue
            if name == ":voice":
                path = Path(arg).expanduser() if arg else None
                if not path or not path.exists():
                    print("找不到那個檔案\n")
                    continue
                sidecar = path.with_suffix(".txt")
                text = sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else transcribe(path, stt)
                if not text:
                    print("聽不出參考音的內容，換一個檔案\n")
                    continue
                use_reference(str(path), text)
                print(f"參考聲音改成 {path}\n參考文字：{text}\n")
                continue
            if name == ":record":
                ref = WORK / "voice.wav"
                print("講十到三十秒，自然講話不要唸稿。")
                input("按 Enter 開始錄音…")
                if not record(ref, args.device):
                    print("沒錄到聲音\n")
                    continue
                text = transcribe(ref, stt)
                if not text:
                    print("聽不出內容，再錄一次\n")
                    continue
                use_reference(str(ref), text)
                print(f"參考聲音換成剛剛那段\n參考文字：{text}\n")
                continue
            if name == ":say":
                if not arg:
                    print("要給問題，例：:say 今天天氣如何\n")
                    continue
                if not state["wav"]:
                    print("打字模式沒有當下的錄音可以當聲音樣本，先 :record 或 :voice\n")
                    continue
                typed = arg
            elif name is not None:
                print(f"沒有 {name} 這個指令，:help 看清單\n")
                continue
            elif not record(wav, args.device):
                print("沒錄到聲音，再試一次。\n")
                continue

        turn_start = time.time()  # 從送進 whisper 算到播放前，就是使用者感覺到的等待
        if typed:
            heard = typed
            stt_secs = 0.0
            print(f"你問：{heard}", flush=True)
        else:
            t0 = time.time()
            heard = transcribe(wav, stt)
            stt_secs = time.time() - t0
            print(f"你說：{heard}　（{stt_secs:.1f}s）", flush=True)
        if not heard:
            print("聽不出內容，再試一次。\n")
            if args.input:
                return
            continue

        t0 = time.time()
        answer = ask_llm(heard, state["backend"], state["model"], state["len"],
                         args.llm_url, history)
        llm = time.time() - t0
        print(f"回答：{answer}　（{llm:.1f}s）", flush=True)

        # ponytail: 整段一次合成,不切句。回答本來就只有幾十個字,
        # 切得越碎離參考文字越遠,CosyVoice 的 clone 品質越差
        t0 = time.time()
        try:
            pieces = [
                j["tts_speech"]
                for j in cv.inference_zero_shot(
                    answer,
                    f"You are a helpful assistant.<|endofprompt|>{state['text'] or heard}",
                    state["wav"] or str(wav),
                    zero_shot_spk_id=state["spk"],
                    stream=False,
                )
            ]
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # 顯存不夠是最常見的失敗，別讓一次失敗把整場對話打掉
            torch.cuda.empty_cache()
            print(f"合成失敗：{str(e).splitlines()[0][:120]}")
            print(f"  顯存剩 {gpu_free_mib()} MiB。關掉 rustdesk 或幾個瀏覽器分頁再試，"
                  f"或 :backend groq 把 llama-server 的顯存讓出來。\n")
            continue
        history.append((heard, answer))
        audio = torch.cat(pieces, dim=1)
        torchaudio.save(str(out), audio, cv.sample_rate)
        tts = time.time() - t0
        print(f"合計 {time.time() - turn_start:.1f}s"
              f"（STT {stt_secs:.1f} ＋ LLM {llm:.1f} ＋ 合成 {tts:.1f}）"
              f"　顯存剩 {gpu_free_mib()}"
              f"　語音長 {audio.shape[1] / cv.sample_rate:.1f}s，播放中…\n", flush=True)
        subprocess.run(["paplay", str(out)])
        if args.input:
            return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
