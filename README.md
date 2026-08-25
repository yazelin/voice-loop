# voice-loop

對著麥克風講話，AI 用**你自己的聲音**回答你。

一輪約 4 到 9 秒（STT 約 0.6s ＋ LLM 0.4 到 2s ＋ 合成 2 到 6s）：

```
麥克風 ──► whisper.cpp（本機 GPU）──► llmshare（雲端 LLM）──► CosyVoice3（本機 GPU）──► paplay
  arecord        語音轉文字              簡短回答一句          用你剛剛那句話當聲音樣本
```

聲音怎麼來，三種模式：

| 怎麼跑 | 參考音是什麼 | 適合 |
|---|---|---|
| 不給參數 | **你剛剛講的那句話**，每輪換 | 隨手玩，不用事先準備 |
| `--record-voice` | 開場先錄一段，整場都用它 | 想要聲音穩定又不想先準備檔案 |
| `--voice 檔案.wav` | 指定的檔案 | 已經有錄好的樣本 |

參考文字都不用自己打：不給參數時用 STT 的結果，其餘兩種讀同名 `.txt` 或自動轉一次。

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
| `local` | `qwen3.5-2b` | 自己先跑 `llama-server` | 見下面「地端 LLM」。完全離線，資料不出這台機器 |

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

# 開場錄一段當這場的參考音（存起來，下次可以直接 --voice 它）
~/CosyVoice/.venv/bin/python voice_loop.py --record-voice assets/我的聲音.wav
~/CosyVoice/.venv/bin/python voice_loop.py --record-voice   # 不存檔，只用這一場

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

### 跑起來之後可以下的指令

提示字元那裡直接按 Enter 就是錄音，也可以打指令。這些都是為了**不用重開**——
重開一次要付 CosyVoice 那 20 秒的載入。

| 指令 | 做什麼 |
|---|---|
| `:voice <檔案>` | 中途換參考聲音，聲紋當場重新快取 |
| `:record` | 中途重錄一段當參考聲音 |
| `:backend groq` | 換 LLM 後端（`llmshare` / `groq` / `local`） |
| `:len 30` | 改回答字數上限 |
| `:say 今天天氣如何` | 不錄音，直接打字問 |
| `:help` / `:q` | 清單／離開 |

`:say` 需要有固定的參考聲音（`--voice`、`--record-voice` 或 `:record`），因為打字模式
沒有「你剛講的那句」可以當聲音樣本。

每輪結束會印出時間拆解，方便看瓶頸在哪：

```
合計 2.7s（STT 0.0 ＋ LLM 0.3 ＋ 合成 2.4）　語音長 4.2s，播放中…
```

計時從送進 whisper 開始、到音檔寫完為止，也就是你講完話之後真正在等的那段。

## 地端 LLM（`--backend local`）

整條線唯一連外網的就是產生回答那一段。想連這段也離線，就在本機跑 `llama-server`，
它給的是 OpenAI 相容端點，程式那邊跟 Groq 共用同一段程式碼，只差網址。

### 一次性建置

llama.cpp **沒有 Linux 的 CUDA 預編版**（只有 Windows 的），所以要自己編。這台的
根碟長期只剩幾 GB，所以整包放在第二顆分割區：

```bash
P=/media/ct/57465421-bf2a-4daf-9133-eab6179e456f/home/ct/llama

git clone --depth 1 https://github.com/ggml-org/llama.cpp "$P/src"
cmake -S "$P/src" -B "$P/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-12 \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$P/build" --target llama-server -j$(nproc)
cp -a "$P/build/bin/." "$P/bin/" && rm -rf "$P/build"   # 只留 209 MB 的執行檔與函式庫

curl -L -o "$P/models/Qwen3.5-2B-Q4_K_M.gguf" \
  https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf
```

兩個一定要給的 cmake 參數：`CMAKE_CUDA_ARCHITECTURES=89` 是 RTX 4060 的 compute
capability，不指定會為所有架構編一遍，慢好幾倍；`CMAKE_CUDA_HOST_COMPILER=gcc-12`
是因為系統預設 gcc-13，CUDA 12.0 不吃。

### 每次啟動

```bash
P=/media/ct/57465421-bf2a-4daf-9133-eab6179e456f/home/ct/llama
LD_LIBRARY_PATH="$P/bin" "$P/bin/llama-server" \
  -m "$P/models/Qwen3.5-2B-Q4_K_M.gguf" \
  --host 127.0.0.1 --port 8080 \
  -ngl 16 -c 1024 -np 1 -fa on --no-webui

# 另一個終端機
~/CosyVoice/.venv/bin/python voice_loop.py --backend local
```

`LD_LIBRARY_PATH` 不能省，把 `build/bin` 搬走之後執行檔找不到自己的 `.so`。

### 為什麼是 `-ngl 16`

這是整件事最反直覺的地方：**顯存不夠，`-ngl` 開太大反而是 whisper 掛掉**。

8 GB 顯存的實際分帳（2026-08-26 實測）：

| 項目 | 佔用 |
|---|---|
| CosyVoice 載入（權重＋CUDA context） | 3490 MiB |
| CosyVoice 推論尖峰 | 4373 MiB |
| whisper-cli ＋ ggml-small | 約 900 MiB |
| Xorg（三螢幕） | 971 MiB |
| Chromium／Electron 分頁 | 約 480 MiB |
| rustdesk | 257 MiB |
| gnome-shell 等雜項 | 約 105 MiB |

扣掉桌面那堆之後，CosyVoice 的尖峰就把剩下的吃掉大半，**只剩約 1900 MiB 給 LLM 和 STT 分**。
LLM 拿太多，whisper 就會直接 SIGSEGV。

`-ngl` 決定幾層放 GPU、幾層留 CPU，實測：

| `-ngl` | llama 顯存 | LLM 平均延遲 |
|---|---|---|
| 99（全上） | 1400 MiB | 0.32s |
| 24 | 1370 MiB | 0.33s |
| 20 | 1238 MiB | 0.38s |
| **16** | **1096 MiB** | **0.58s** |
| 12 | 960 MiB | 1.45s |

16 是甜蜜點：省下的 304 MiB 剛好讓 whisper 塞得進去，速度只從 0.32 掉到 0.58 秒。
掉到 12 速度就崩到 1.45 秒，比 Groq 還慢，沒有意義。

**顯存多出來的話**（關掉 rustdesk 省 257 MiB、關幾個瀏覽器分頁、少接一台外接螢幕），
可以把 `-ngl` 往上調回 20 或 24。三個螢幕合計 968 萬像素，筆電內建面板只佔 24%，
Xorg 那 971 MiB 大部分是兩台外接螢幕的 framebuffer——不過拔線前後的差值我沒實測過。

`-c 1024 -np 1 -fa on` 是把 context 調小、只開一個 slot、開 flash attention。
llama-server 預設會開 4 個平行 slot，KV cache 跟著乘四。這組只省 100 MiB，
比不上 `-ngl`，但不花錢。

### 模型怎麼挑的

顯存先卡死尺寸，剩下的才輪到品質。同樣六個問題實測：

| 模型 | 顯存 | 平均 | 品質 |
|---|---|---|---|
| Qwen3-1.7B Q4_K_M | 約 1100 MiB | 0.09s | **淘汰**。六題有四題只是把問題原樣複誦一遍 |
| **Qwen3.5-2B Q4_K_M** | 1400 MiB（-ngl 99） | 0.32s | 六題都正常回答。偶爾有事實瑕疵，偶爾夾雜簡體字 |
| Qwen3.5-0.8B Q4_K_M | 682 MiB | 0.11s | **淘汰**。句子讀起來很順，但事實是錯的：「太陽反射雲層所以天空是藍的」、「電動車沒有二氧化碳排放」，還會把問題當成在問自己（「是的，我說話速度很快」）。而且串進整條線之後回答語無倫次 |

小尺寸模型目前最新的世代是 **Qwen3.5**（2026-02）。Qwen3.6 和 Qwen3.8 都只放 27B 以上，
所以 2B 這一階沒有更新的可選。官方 `Qwen/Qwen3.5-2B-GGUF` 不存在，量化版要找
`unsloth/Qwen3.5-2B-GGUF`。

**別再往下試更小的**。1.7B 和 0.8B 兩顆都試過、都刪了：1.7B 會複誦問題，0.8B 會流暢地
講錯事實——後者更危險，因為聽起來完全正常。2B 是這個任務跑得動的下限，真的顯存不夠，
寧可把 `-ngl` 調小讓 2B 分層跑，也不要換更小的模型。

### 值不值得

| 後端 | LLM 延遲 | 備註 |
|---|---|---|
| Groq `openai/gpt-oss-120b` | 0.37s | 答得準但乾 |
| **local Qwen3.5-2B（-ngl 16）** | **0.58s** | 完全離線。品質是 2B 的水準，別期待太高 |
| llmshare `deepseek-v4-flash:0731` | 1.74s | 最口語、最有溫度 |

地端**不會比 Groq 快**，換過來的理由是離線與資料不出門。而且整輪的瓶頸從來不是 LLM，
是合成那 2.8 秒。

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
找不到才用 whisper 轉一次。

Tiffy 只是其中一個選項，不會被蓋掉。`assets/` 想放幾支就放幾支，一支 wav 配一支同名 txt，
每次跑再用 `--voice` 挑要哪一支：

```bash
cp 我的聲音.wav 我的聲音.txt assets/     # 多一個選項，Tiffy 還在

~/CosyVoice/.venv/bin/python voice_loop.py --voice assets/jinn-tiffy-10s.wav   # Tiffy
~/CosyVoice/.venv/bin/python voice_loop.py --voice assets/我的聲音.wav          # 你自己
~/CosyVoice/.venv/bin/python voice_loop.py                                     # 不挑,用你當下講的那句
```

`--voice` 和 `--record-voice` 都會自動快取聲紋，每輪合成少 1.5 秒（見上面「合成要更快」）。

`--record-voice` 給了路徑就會把 wav 和轉好的 `.txt` 一起存下來，下次直接
`--voice 那個檔` 就好，不必再錄一次。不給路徑就只在這一場有效。

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
- **錄到靜音時 whisper 會把那句 prompt 原樣吐回來**，看起來就像你真的講了那句話。程式把
  等於 prompt 的轉寫結果當成沒聽到（`STT_HINT`），換了 prompt 的話那個守衛要跟著換。
- 錄太短（少於一秒）clone 出來的聲音會不穩，講完整一句再放開。
- **Groq 會擋 urllib 的預設 User-Agent**，回 `403 error code 1010`（那是 Cloudflare 不是 Groq）。
  程式送了自己的 `User-Agent`，別把那行拿掉。
- **whisper 被擠掉的時候會 SIGSEGV**（`exit -11`，有時是 `exit 10`），畫面上不會出現 OOM 字樣。
  更糟的是它 abort 前會把 **GDB 的 backtrace 印到 stdout**，所以只看 exit code 卻照用 stdout，
  那整串 backtrace 會被當成你講的話送進 LLM（真的發生過，模型認真回答了那段 backtrace）。
  現在失敗一律回空字串，並把 exit code 與 stderr 印出來。
- **mori 的 whisper-server 若在跑就借用它**（`ps` 裡撈 `--port`）。它的模型已經在顯存裡，
  借用等於白賺：省下自己那份約 900 MiB，STT 也從 0.82 秒變 0.37 秒（不必每次重載模型）。
  它沒在跑才退回 `whisper-cli`。
- **開始載入前會先看顯存夠不夠**，不夠就直接講，免得等 20 秒載完才在合成那步炸掉。
  真的炸了程式會自己處理掉，不會把整場對話打斷。
- **`pkill -f "llama-server..."` 會把自己殺掉**：執行這行的 shell，它自己的 cmdline 就含有
  `llama-server` 這串字，`pkill -f` 比對得到。要用連接埠找 PID：
  `ss -ltnp | awk '/127.0.0.1:8080/ {print $NF}' | grep -o 'pid=[0-9]*' | cut -d= -f2`。
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
