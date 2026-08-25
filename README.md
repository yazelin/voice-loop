# voice-loop

對著麥克風講話，AI 用**你自己的聲音**回答你。

一輪約 9 秒（實測 2.4s ＋ 3.5s ＋ 3.3s）：

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
| **llmshare** | 產生回答 | CLI，**打雲端 API** | 預設模型 `deepseek-v4-flash:0731`。金鑰放環境變數 `LLMSHARE_API_KEY` |
| arecord / paplay | 錄音、播放 | `alsa-utils`、`pulseaudio-utils` | |
| ffmpeg | 音檔處理 | | |

整條線裡**只有 llmshare 會連外網**，STT 與 TTS 都在本機 GPU。
llmshare 是共享閘道 CLI，安裝與金鑰見 <https://github.com/yazelin/duotify-ollama-cloud-setup>；
`llmshare models` 可以看全部模型，換模型用 `--model`。

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
# 換 LLM
~/CosyVoice/.venv/bin/python voice_loop.py --model glm-5.2

# 固定用別的聲音回答（不給 --voice-text 就自動轉一次逐字稿）
~/CosyVoice/.venv/bin/python voice_loop.py --voice assets/jinn-tiffy-10s.wav

# 指定麥克風（arecord -l 查裝置編號）
~/CosyVoice/.venv/bin/python voice_loop.py --device plughw:1,0

# 不用麥克風，拿現成 wav 跑一輪就結束（測試用）
~/CosyVoice/.venv/bin/python voice_loop.py --input some.wav

# 純文字邏輯自測，不載模型、不用 GPU
python3 voice_loop.py --selfcheck
```

其他選項：`--max-chars`（回答字數上限，預設 60）。

## 附的範例聲音

`assets/jinn-tiffy-10s.wav`（＋ 同名 `.txt` 逐字稿）是一段十秒的台灣中文旁白，
用來示範 `--voice`：

```bash
~/CosyVoice/.venv/bin/python voice_loop.py \
  --voice assets/jinn-tiffy-10s.wav \
  --voice-text "$(cat assets/jinn-tiffy-10s.txt)"
```

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
