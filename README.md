OpenPI + PictureEnv Fine-Tuning Memo
概要
PictureEnv 用に OpenPI (pi0-fast) を LoRA Fine-Tuning するための構築手順および実験メモ．
1. リポジトリ取得
OpenPI
cd ~/workspace
git clone https://github.com/Physical-Intelligence/openpi.git
PictureEnv
cd ~/workspace
git clone <picture_env_repository>
2. Docker 起動
使用 Dockerfile
docker/Dockerfile
ビルドaux
docker build -t openpi .
起動
docker run \
    --gpus all \
    -it \
    --rm \
    -v ~/workspace:/home/rllab/workspace \
    openpi
※ workspace を volume mount しているため，
~/workspace/openpi
を修正すると Docker 内にも反映される．
Docker 再ビルド不要．
3. OpenPI 環境構築
cd ~/workspace/openpi
uv sync
確認
uv run python -c "import openpi"
4. OpenPI 修正箇所
以下は OpenPI 本体に追加した内容．
src/openpi/training/config.py
src/openpi/training/data_loader.py
src/openpi/policies/policy_config.py
src/openpi/models/tokenizer.py
src/openpi/models/pi0_fast.py
5. PictureEnv 用 Config 追加
追加 Config
pi0_fast_picture_env_lora
主な設定
action_dim=4
action_horizon=10
max_token_len=512
batch_size=1
num_workers=0
num_train_steps=10000
LoRA 使用
paligemma_variant="gemma_2b_lora"
6. Dataset
使用データ
/home/rllab/workspace/picture_env/log/dataset.npz
確認
import numpy as np
data = np.load("dataset.npz")
print(data.files)
print(data["actions"].shape)
7. 学習
実行
cd ~/workspace/openpi
uv run scripts/train.py \
    pi0_fast_picture_env_lora \
    --exp-name=test_picture_env_len512 \
    --overwrite
出力
checkpoints/
└── pi0_fast_picture_env_lora
    └── test_picture_env_len512
8. Policy Server
起動
cd ~/workspace/openpi
uv run python scripts/serve_policy.py \
    --default-prompt "pick up the object and place it at the goal" \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi0_fast_picture_env_lora \
    --policy.dir ./checkpoints/pi0_fast_picture_env_lora/test_picture_env_len512/10000
正常時
INFO:websockets.server:server listening on 0.0.0.0:8000
9. Client
インストール
pip install -e ~/workspace/openpi/packages/openpi-client
接続
from openpi_client import websocket_client_policy
URL
ws://localhost:8000
10. Tokenizer デバッグ
学習時 token
Action:
<loc0751><loc0675><loc0729><loc0602>
<loc0757><loc0713><loc0737>
<loc0493><loc0493><loc0078>|
推論時 token 例
Action:
<loc0602><loc0757><loc0701>...
確認場所
src/openpi/models/tokenizer.py
11. 現在の課題
推論時
Error decoding tokens:
Decoded DCT coefficients ...
が発生．
学習 token と推論 token の比較を継続中．
候補
学習不足
Action token 生成長
FAST decode 周辺
推論 token 分布の崩れ
12. よく使うコマンド
学習
uv run scripts/train.py ...
推論サーバ
uv run python scripts/serve_policy.py ...
クライアント
python3 scripts/openpi_test.py
サーバ停止
Ctrl + C