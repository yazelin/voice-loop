#!/usr/bin/env bash
# 檢查（缺的就裝）voice-loop 需要的四樣東西：CosyVoice3、whisper.cpp、llmshare、錄放音。
# 只做必要的事，不裝已經有的。跑法：bash setup.sh
set -uo pipefail
ok=0; miss=0
say() { printf '%-28s %s\n' "$1" "$2"; }
need() { say "$1" "缺：$2"; miss=1; }
have() { say "$1" "OK  $2"; }

echo "== voice-loop 依賴檢查 =="

# 1. GPU
if command -v nvidia-smi >/dev/null; then
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  have "NVIDIA GPU" "剩餘顯存 ${free} MiB"
  [ "$free" -lt 4200 ] && echo "   警告：CosyVoice 載入要約 3.2 GiB，現在只剩 ${free} MiB，先關掉吃顯存的程式。"
else
  need "NVIDIA GPU" "沒有 nvidia-smi。純 CPU 跑 CosyVoice 的 RTF 約 6～8，這個迴圈會慢到不能用。"
fi

# 2. 錄放音
for c in arecord paplay ffmpeg; do
  command -v $c >/dev/null && have "$c" "$(command -v $c)" || need "$c" "sudo apt install alsa-utils pulseaudio-utils ffmpeg"
done

# 3. llmshare（回答用的 LLM 閘道）
if command -v llmshare >/dev/null; then
  have "llmshare" "$(command -v llmshare)"
  [ -n "${LLMSHARE_API_KEY:-}" ] && have "LLMSHARE_API_KEY" "已設" \
    || need "LLMSHARE_API_KEY" "export LLMSHARE_API_KEY=... （沒有這把金鑰就問不到模型）"
else
  need "llmshare" "見 https://github.com/yazelin/duotify-ollama-cloud-setup 的 README"
fi

# 4. whisper.cpp（STT，要 CUDA 版才會用到 GPU）
WHISPER_CLI="${WHISPER_CLI:-$HOME/.mori/bin/whisper-cli}"
WHISPER_MODEL="${WHISPER_MODEL:-$HOME/.mori/models/ggml-small.bin}"
[ -x "$WHISPER_CLI" ] && have "whisper-cli" "$WHISPER_CLI" \
  || need "whisper-cli" "自己編 whisper.cpp（cmake -DGGML_CUDA=ON）後 export WHISPER_CLI=<路徑>"
[ -f "$WHISPER_MODEL" ] && have "whisper 模型" "$WHISPER_MODEL" \
  || need "whisper 模型" "下載 ggml-small.bin 後 export WHISPER_MODEL=<路徑>
   https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"

# 5. CosyVoice3（TTS ＋ voice clone）
CV="${COSYVOICE_DIR:-$HOME/CosyVoice}"
CVMODEL="${COSYVOICE_MODEL:-$CV/pretrained_models/Fun-CosyVoice3-0.5B}"
if [ -x "$CV/.venv/bin/python" ] && [ -d "$CVMODEL" ]; then
  have "CosyVoice3" "$CV"
else
  need "CosyVoice3" "要裝約 12 GB（程式碼 ＋ 5.1 GB 模型）。磁碟剩 $(df -h "$HOME" | awk 'NR==2{print $4}')"
  cat <<'TIP'

   安裝（會下載約 12 GB，跑一次就好）：
     git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git ~/CosyVoice
     cd ~/CosyVoice && uv venv --python 3.10 .venv
     uv pip install -p .venv setuptools -r requirements.txt \
       --index-strategy unsafe-best-match \
       --extra-index-url https://download.pytorch.org/whl/cu121
     .venv/bin/python -c "from huggingface_hub import snapshot_download; \
       snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', \
       local_dir='pretrained_models/Fun-CosyVoice3-0.5B', \
       ignore_patterns=['llm.rl.pt','*.batch.onnx','flow.decoder.estimator.fp32.onnx'])"
TIP
fi

echo
if [ "$miss" = 0 ]; then
  echo "全部齊了。開跑："
  echo "  \"\${COSYVOICE_DIR:-\$HOME/CosyVoice}\"/.venv/bin/python voice_loop.py"
else
  echo "上面標「缺」的補完再跑。純文字邏輯可以先自測：python3 voice_loop.py --selfcheck"
  exit 1
fi
