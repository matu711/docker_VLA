# 推論

## 最新 checkpoint 確認

### pi0-fast

```bash
find checkpoints/pi0_fast_picture_env_lora/test_picture_env_len512 \
  -maxdepth 1 -mindepth 1 -type d | sort
```

### pi0

```bash
find checkpoints/pi0_picture_env_lora/pi0_picture_env_lora_test \
  -maxdepth 1 -mindepth 1 -type d | sort
```

---

## pi0-fast サーバ起動

```bash
CKPT=$(find checkpoints/pi0_fast_picture_env_lora/test_picture_env_len512 \
  -maxdepth 1 -mindepth 1 -type d -printf "%f\n" | sort -n | tail -1)

uv run python scripts/serve_policy.py \
  --default-prompt "pick up the object and place it at the goal" \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi0_fast_picture_env_lora \
  --policy.dir ./checkpoints/pi0_fast_picture_env_lora/test_picture_env_len512/$CKPT
```

---

## pi0 サーバ起動

```bash
CKPT=$(find checkpoints/pi0_picture_env_lora/pi0_picture_env_lora_test \
  -maxdepth 1 -mindepth 1 -type d -printf "%f\n" | sort -n | tail -1)

uv run python scripts/serve_policy.py \
  --default-prompt "pick up the object and place it at the goal" \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi0_picture_env_lora \
  --policy.dir ./checkpoints/pi0_picture_env_lora/pi0_picture_env_lora_test/$CKPT
```

サーバ起動成功時

```text
INFO:websockets.server:server listening on 0.0.0.0:8000
```

が表示される．

---
Clientライブラリ準備

初回のみ実行

cd ~/workspace/picture_env

pip install -e ~/workspace/openpi/packages/openpi-client

確認
---

## Client

```bash
cd ~/workspace/picture_env

python3 scripts/openpi_test.py
```

---

## サーバ停止

```bash
Ctrl + C
```
