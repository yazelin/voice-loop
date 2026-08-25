# voice-loop

對著麥克風講話，AI 用**你自己的聲音**回答你。

一輪約 4 到 9 秒（STT 約 0.6s ＋ LLM 0.4 到 2s ＋ 合成 2 到 6s）：

```
麥克風 ──► whisper.cpp（本機 GPU）──► llmshare（雲端 LLM）──► CosyVoice3（本機 GPU）──► paplay
  arecord        語音轉文字              簡短回答一句          用你剛剛那句話當聲音樣本
```

voice clone 的參考音就是**你剛剛講的那句話**，參考文字就是 STT 的結果，所以不用事先錄樣本。
想固定成別的聲音，給 `--voice`。

## 依賴

先跑檢查，缺什麼它會講清楚怎麼補：

```bash
bash setup.sh
```

| 元件 | 做什麼 | 在哪 | 備註 |
|---|---|---|---|
| **CosyVoice3 0.5B** | TTS ＋ zero-shot voice clone | 本機 `~/CosyVoice`，走 **GPU** | 約 12 GB（程式 ＋ 5.1 GB 模型）。載入吃約 3.2 GiB 顯存，RTF 約 0.7 |
| **whisper.cpp** | 語音轉文字 | 本機 `~/.mori/bin/whisper-cli` ＋ `ggml-small.bin`，走 **GPU** | 要編 CUDA 版（`-DGGML_CUDA=ON`），不然會掉回 CPU |
| **llmshare** 或 **Groq** | 產生回答 | **打雲端 API**，二選一 | 見下面「換 LLM 後端」 |
| arecord / paplay | 錄音、播放 | `alsa-utils`、`pulseaudio-utils` | |
| ffmpeg | 音檔處理 | | |

整條線裡**只有產生回答那一段會連外網**，STT 與 TTS 都在本機 GPU。

### 換 LLM 後端

`--backend` 二選一，兩邊都不給 `--model` 就用各自的預設：

| `--backend` | 預設模型 | 要什麼 | 怎麼裝 |
|---|---|---|---|
| `llmshare`（預設） | `deepseek-v4-flash:0731` | `llmshare` CLI ＋ 環境變數 `LLMSHARE_API_KEY` | <https://github.com/yazelin/duotify-ollama-cloud-setup>；`llmshare models` 看全部模型 |
| `groq` | `openai/gpt-oss-120b` | 環境變數 `GROQ_API_KEY` | <https://console.groq.com/keys>，不必裝任何套件（走 stdlib 的 urllib） |

```bash
export GROQ_API_KEY=gsk_...
~/CosyVoice/.venv/bin/python voice_loop.py --backend groq
```

Groq 那條走 OpenAI 相容的 `/openai/v1/chat/completions`，`reasoning_effort` 設成 `low`
（gpt-oss 會先想再答，想太久就失去用 Groq 的意義）。

**實測（2026-08-26，同樣三個問題各問一次）**：

| 後端 | 平均 | 最慢 | 答起來的樣子 |
|---|---|---|---|
| Groq `openai/gpt-oss-120b` | 0.37s | 0.44s | 準確但乾，像查資料 |
| llmshare `deepseek-v4-flash:0731` | 1.74s | 2.01s | 慢五倍，但口語、有溫度，比較像在聊天 |

要反應快選 Groq，要講話像人選 llmshare。整輪的另外兩段（STT 約 0.6 秒、合成 2 到 6 秒）
兩邊一樣，所以換 Groq 大概省一秒多。

### 路徑覆寫

不想照預設路徑放，用環境變數：

```bash
export COSYVOICE_DIR=/somewhere/CosyVoice
export COSYVOICE_MODEL=/somewhere/Fun-CosyVoice3-0.5B
export WHISPER_CLI=/somewhere/whisper-cli
export WHISPER_MODEL=/somewhere/ggml-small.bin
```

## 跑

一定要用 CosyVoice 那個 venv 的 python（依賴都裝在裡面）：

```bash
~/CosyVoice/.venv/bin/python voice_loop.py
```

按 Enter 開始錄音，再按 Enter 停止，然後等它回答。Ctrl-C 離開。

```bash
# 換 LLM 後端與模型
~/CosyVoice/.venv/bin/python voice_loop.py --backend groq
~/CosyVoice/.venv/bin/python voice_loop.py --model glm-5.2

# 固定用附的 Tiffy 聲音回答（逐字稿自動讀同名 .txt）
~/CosyVoice/.venv/bin/python voice_loop.py --voice assets/jinn-tiffy-10s.wav

# 指定麥克風（arecord -l 查裝置編號）
~/CosyVoice/.venv/bin/python voice_loop.py --device plughw:1,0

# 不用麥克風，拿現成 wav 跑一輪就結束（測試用）
~/CosyVoice/.venv/bin/python voice_loop.py --input some.wav

# 純文字邏輯自測，不載模型、不用 GPU
python3 voice_loop.py --selfcheck
```

其他選項：`--max-chars`（回答字數上限，預設 60）。

## 合成要更快

合成是整輪裡最花時間的一段。實測拆解（10 秒參考音、約 6 秒的回答）：
參考音前處理 1.4 秒 ＋ 模型生成 2.8 到 3.7 秒。

量過、有用的：

| 手段 | 效果 |
|---|---|
| **固定參考音時快取聲紋**（給了 `--voice` 就自動做） | 每輪 4.4 秒 → 2.8 秒。省下的就是那 1.4 秒的前處理 |
| **回答短一點**（`--max-chars 30`） | 模型段的 RTF 約 0.5，時間跟音長成正比，字數砍半時間就砍半 |

量過、沒用或不划算的：

| 手段 | 為什麼不用 |
|---|---|
| `stream=True` 邊生邊播 | 首段確實早約 1 秒出來，但總時間反而多約 1 秒（RTF 0.70 → 0.90），而且只切成 2 到 3 段，效果不穩 |
| `speed=1.15` | 沒有量到差別 |
| onnxruntime 擺回 GPU | 前處理 1.5 秒 → 1.1 秒，只省 0.35 秒，然後第二輪就 OOM |
| TensorRT / JIT | 模型目錄裡沒有現成的 `.plan` / `.zip`，要自己建引擎。8 GB 顯存做這個不划算 |

預設模式（用你自己的聲音）吃不到聲紋快取，因為每輪的參考音都不一樣。不過那時候的參考音
就是你剛講的一兩句，只有兩三秒，前處理本來就只要 0.3 到 0.7 秒。

## 附的範例聲音

repo 內附了 `assets/jinn-tiffy-10s.wav`，一段十秒的台灣中文旁白（暱稱 Tiffy）。用它回答：

```bash
cd ~/voice-loop
~/CosyVoice/.venv/bin/python voice_loop.py --voice assets/jinn-tiffy-10s.wav
```

逐字稿不必自己給。`--voice` 會先找**同名的 `.txt`**（這裡是 `assets/jinn-tiffy-10s.txt`），
找不到才用 whisper 轉一次。要換成自己的聲音，就把 wav 和同名 txt 一起放進去：

```bash
cp 我的聲音.wav 我的聲音.txt assets/
~/CosyVoice/.venv/bin/python voice_loop.py --voice assets/我的聲音.wav
```

給了 `--voice` 就會自動快取聲紋，每輪合成少 1.5 秒（見上面「合成要更快」）。

**授權注意**：這段音是 2026 年用 ElevenLabs 的商用聲線「Tiffy - Taiwanese Bilingual
Narrator」產生的，放在這裡只供**跑通流程的示範**。拿它 clone 出來的語音要對外發佈
（影片、廣告、商品）之前，請自己確認 ElevenLabs 的授權條款。要做對外內容，換成你
自己錄的聲音，或乾脆不給 `--voice`，直接用你講話的那句。

## 踩過的坑

- **8 GB 顯存塞不下 onnxruntime 的 CUDA EP ＋ CosyVoice**，會在 `cublasCreate` 爆
  `CUBLAS_STATUS_ALLOC_FAILED`。程式裡把 onnxruntime 逼回 CPU（那兩個 onnx 很小，慢不了多少），
  大顯存的機器可以刪掉那三行。
- 桌面本身就吃掉不少顯存（實測 Xorg 961 MiB ＋ 瀏覽器分頁數百 MiB）。跑之前先看 `nvidia-smi`，
  剩不到 4.2 GiB 就先關東西。
- whisper 對中文預設吐簡體，靠 `--prompt "以下是繁體中文的句子。"` 壓回正體，不是百分之百。
- 錄太短（少於一秒）clone 出來的聲音會不穩，講完整一句再放開。
- **Groq 會擋 urllib 的預設 User-Agent**，回 `403 error code 1010`（那是 Cloudflare 不是 Groq）。
  程式送了自己的 `User-Agent`，別把那行拿掉。
- 啟動時 wetext 從 modelscope 抓檔案會 403、印「no frontend is avaliable」，不影響合成。
- 參考音的**語氣神態也會被 clone**。拿 TTS 合成音當參考，輸出就是機器人唸稿腔。
- **回答比參考音短的時候 clone 品質會掉**，CosyVoice 會印
  `too short than prompt text, this may lead to bad performance`。所以程式整段一次合成、
  不切句（切得越碎離參考文字越遠）。這個警告在這種用法下必然出現，已經連同 tqdm 進度條
  一起壓掉；想看回來就把 `logging.basicConfig` 那行改回 `WARNING`、拿掉 `TQDM_DISABLE`。
  真的很在意品質，就講長一點的句子當樣本。

## 授權 License

MIT © 2026 林亞澤 (Yaze Lin)

---

由 **林亞澤 Yaze Lin** 開發。覺得有用，歡迎分享，或請我喝杯咖啡。

- 原始碼 GitHub：<https://github.com/yazelin/voice-loop>
- Facebook：<https://www.facebook.com/yaze.lin.gm>
- Buy Me a Coffee：<https://buymeacoffee.com/yazelin>
