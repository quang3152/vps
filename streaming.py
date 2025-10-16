import asyncio
import json
import logging
from dataclasses import dataclass

import os
import numpy as np
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
# ---- ViStreamASR
# pip install torch torchaudio fastapi uvicorn
from ViStreamASR.streaming import \
    StreamingASR  # hoặc: from streaming import StreamingASR

# --- Cấu hình “tương đương” ---

ASR_SR = int(os.getenv("ASR_SR", 16000))                 # Hz
CHUNK_MS = int(os.getenv("CHUNK_MS", 640))               # ms
AUTO_FINALIZE_AFTER = float(os.getenv("AUTO_FINALIZE_AFTER", 15.0))  # s

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 9241))
# ASR_SR = 16000
# CHUNK_MS = 640  # giống default ViStreamASR để độ trễ/độ mượt cân bằng
CHUNK_SAMPLES = int(ASR_SR * CHUNK_MS / 1000.0)
# AUTO_FINALIZE_AFTER = 15.0
# Khởi tạo engine một lần
asr = StreamingASR(chunk_size_ms=CHUNK_MS, auto_finalize_after=AUTO_FINALIZE_AFTER, debug=True)
asr._ensure_engine_initialized()


@dataclass
class StreamState:
    buf: torch.Tensor
    text: str
    resampler: torchaudio.transforms.Resample


def make_state(orig_sr: int) -> StreamState:
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=ASR_SR)
    return StreamState(
        buf=torch.zeros(0, dtype=torch.float32),
        text="",
        resampler=resampler,
    )


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.websocket("/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    state: StreamState | None = None
    client_sr = ASR_SR

    try:
        while True:
            msg = await ws.receive()

            # client ngắt
            if msg.get("type") == "websocket.disconnect":
                break

            # 1) KHUNG BYTES: audio
            if msg.get("bytes") is not None:
                if state is None:
                    # Chưa "start" thì bỏ qua (giống server cũ)
                    continue

                # Browser gửi Float32 mono → cần copy để tránh "non-writable tensor"
                raw = np.frombuffer(msg["bytes"], dtype=np.float32).copy()
                y = torch.from_numpy(raw).float().view(-1)

                # chuẩn hóa tránh clipping
                if y.numel():
                    peak = torch.max(torch.abs(y)).item()
                    if peak > 1e-6:
                        y = y / max(1.0, peak)

                # resample → 16 kHz
                y = state.resampler(y)

                # tích lũy theo CHUNK_SAMPLES (giữ “cảm giác” như FRAME_SIZE trước đây)
                state.buf = torch.cat([state.buf, y])
                while state.buf.numel() >= CHUNK_SAMPLES:
                    chunk = state.buf[:CHUNK_SAMPLES].numpy()  # numpy float32 16k mono
                    state.buf = state.buf[CHUNK_SAMPLES:]

                    # ... sau khi tạo `chunk` và gọi engine:
                    result = asr.engine.process_audio(chunk, is_last=False)

                    final_text = result.get("new_final_text")
                    if final_text:
                        # 1) Gửi final ngay khi engine finalize (giữa chừng)
                        await ws.send_text(json.dumps({"type": "final", "text": final_text}))
                        # 2) Reset text theo ý bạn (để segment mới không “thừa” text cũ)
                        state.text = ""
                        # QUAN TRỌNG: không gửi partial rỗng ngay sau khi reset
                        continue

                    new_partial = result.get("current_transcription") or ""
                    # 3) Chỉ gửi partial nếu KHÔNG rỗng và có thay đổi
                    if new_partial and new_partial != state.text:
                        state.text = new_partial
                        await ws.send_text(json.dumps({"type": "partial", "text": state.text}))

                continue

            # 2) KHUNG TEXT: control JSON ("start"/"stop"/"ping")
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"type": "error", "message": "Non-JSON text frame"}))
                    continue

                mtype = data.get("type")

                if mtype == "start":
                    client_sr = int(data.get("sampleRate", ASR_SR))
                    state = make_state(client_sr)
                    # reset state trong engine để bắt phiên mới
                    asr.engine.reset_state()
                    await ws.send_text(json.dumps({"type": "ready", "sampleRate": ASR_SR}))
                    continue

                if mtype == "stop":
                    # ... trong nhánh mtype == "stop":
                    if state is not None:
                        if state.buf.numel() > 0:
                            result = asr.engine.process_audio(state.buf.numpy(), is_last=True)
                        else:
                            result = asr.engine.process_audio(np.array([], dtype=np.float32), is_last=True)

                        final_text = result.get("new_final_text") or state.text
                        await ws.send_text(json.dumps({"type": "final", "text": final_text}))
                        state = None
                    else:
                        await ws.send_text(json.dumps({"type": "final", "text": ""}))
                    continue

                if mtype == "finalize_segment":
                    # Xử lý finalize segment từ silence detection
                    if state is not None:
                        # Gửi buffer còn lại với is_last=True để force finalize
                        if state.buf.numel() > 0:
                            result = asr.engine.process_audio(state.buf.numpy(), is_last=True)
                            state.buf = torch.zeros(0, dtype=torch.float32)  # Clear buffer sau khi finalize
                        else:
                            # Gửi empty array với is_last=True để trigger finalization
                            result = asr.engine.process_audio(np.array([], dtype=np.float32), is_last=True)

                        final_text = result.get("new_final_text") or state.text
                        if final_text and final_text.strip():
                            await ws.send_text(json.dumps({
                                "type": "final",
                                "text": final_text,
                                "reason": data.get("reason", "silence_detected")
                            }))
                            state.text = ""  # Reset state text

                        # Reset engine state để chuẩn bị cho segment mới
                        asr.engine.reset_state()
                    continue

                if mtype == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                    continue

                await ws.send_text(json.dumps({"type": "error", "message": f"Unknown message type: {mtype}"}))
                continue

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.exception("ASR error")
        if ws.client_state == WebSocketState.CONNECTED:
            try:
                await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass
    finally:
        if ws.client_state == WebSocketState.CONNECTED:
            try:
                await ws.close()
            except Exception:
                pass


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
