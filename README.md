# OpenPI + PictureEnv Fine-Tuning Memo

PictureEnv 用に OpenPI の π0 を LoRA Fine-Tuning するための構築手順および実験メモです。

## 概要

本リポジトリでは，PictureEnv のデータセットを用いて OpenPI の π0 を Fine-Tuning し，Policy Server 経由で推論を行います。

主な内容は以下の通りです。

* OpenPI の環境構築
* PictureEnv 用 Dataset の利用
* π0 の LoRA Fine-Tuning
* Policy Server の起動
* Client からの推論実行
* 学習済みモデルの動作確認

## 1. リポジトリ取得

### OpenPI

```bash
cd ~/docker_VLA/workspace
git clone https://github.com/Physical-Intelligence/openpi.git
```
## 2. Docker 起動前のユーザ名変更

Docker をビルドする前に，Dockerfile, runfile 内のユーザ名を確認します。

```bash
cd ~/docker_VLA
```

以下のように，ユーザ名が `shunsuke-m` になっている場合があります。

```dockerfile
ARG USR_NAME=shunsuke-m
```

本実験では Docker 内の作業パスを

```text
/home/{$USER}/docker_VLA/workspace
```

に統一するため，以下のように変更します。

```dockerfile
ARG USR_NAME=$USER
```

変更後，保存してから Docker イメージをビルドします。

---

## 3. Docker 起動

Docker のビルドおよび起動は，リポジトリ内のシェルスクリプトを用いて行います。

### Docker イメージのビルド

```bash
cd ~/docker_VLA
sh build.sh
```

### Docker コンテナの起動

```bash
cd ~/docker_VLA
sh run.sh
```

## 4. OpenPI 内の Dataset パス変更

PictureEnv の dataset を読み込むために，OpenPI 側の Config に書かれている `data_path` を変更します。

対象ファイルは以下です。

```text
/home/${USER}/docker_VLA/workspace/openpi/src/openpi/training/config.py
```

`config.py` 内の `PictureEnvDataConfig` を確認します。

変更前の例：

```python
data=PictureEnvDataConfig(
    repo_id="picture_env_pi0",
    data_path="/home/rlsv/proj-matsumoto/docker_VLA/workspace/picture_env/log/dataset.npz",
),
```

本環境では `docker_VLA/workspace` にパスを統一するため，以下のように変更します。

```python
data=PictureEnvDataConfig(
    repo_id="picture_env_pi0",
    data_path="/home/${USER}/docker_VLA/workspace/picture_env/log/dataset.npz",
),
```

例えば Docker 内のユーザ名が `rllab` の場合は，以下のようにします。

```python
data=PictureEnvDataConfig(
    repo_id="picture_env_pi0",
    data_path="/home/rllab/docker_VLA/workspace/picture_env/log/dataset.npz",
),
```

## 5. Gymnasium-Robotics のインストール

PictureEnv では Gymnasium-Robotics を使用するため，Docker 内で `gymnasium_robotics` を editable install します。

```bash
cd /home/${USER}/docker_VLA/workspace/gymnasium_robotics
pip install -e .
```

## 6. デモデータの収集

PictureEnv で π0 Fine-Tuning 用のデモデータを収集します。

Docker 内で以下を実行します。

```bash
cd /home/${USER}/docker_VLA/workspace/picture_env
python scripts/collect_demo.py
```

収集したデータは，以下に保存されます。

```text
/home/${USER}/docker_VLA/workspace/picture_env/log/dataset.npz
```
## 7. π0 の Fine-Tuning

収集した PictureEnv のデモデータを用いて，OpenPI の π0 を LoRA Fine-Tuning します。

### 使用する Config

Fine-Tuning には，OpenPI に追加した以下の Config を使用します。

```text
pi0_picture_env_lora
```

この Config .pyでは，PictureEnv 用に以下のような設定を行います。

```python
TrainConfig(
    name="pi0_picture_env_lora",

    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=32,
        action_horizon=10,
    ),

    data=PictureEnvDataConfig(
        repo_id="picture_env_pi0",
        data_path="/home/${USER}/docker_VLA/workspace/picture_env/log/dataset.npz",
    ),

    num_train_steps=10000,
    batch_size=1,
    num_workers=0,
)
```

例：

```python
data_path="/home/rllab/docker_VLA/workspace/picture_env/log/dataset.npz"
```

### 主な設定

| 項目                                        | 内容                               |
| ----------------------------------------- | -------------------------------- |
| `paligemma_variant="gemma_2b_lora"`       | PaliGemma 側を LoRA で学習する          |
| `action_expert_variant="gemma_300m_lora"` | Action Expert 側を LoRA で学習する      |
| `action_dim=32`                           | π0 が扱う action 次元                 |
| `action_horizon=10`                       | 一度に予測する action 系列の長さ             |
| `batch_size=1`                            | GPU メモリ削減のため 1 に設定               |
| `num_workers=0`                           | Dataset 読み込みを single process にする |
| `num_train_steps=10000`                   | 学習ステップ数                          |

---

### 学習コマンド

Docker 内で OpenPI ディレクトリに移動します。

```bash
cd /home/${USER}/docker_VLA/workspace/openpi
```

以下のコマンドで Fine-Tuning を実行します。

```bash
uv run python scripts/train.py \
    pi0_picture_env_lora \
    --exp-name=test_picture_env \
    --overwrite
```

### オプションの意味

| オプション                         | 内容                  |
| ----------------------------- | ------------------- |
| `pi0_picture_env_lora`        | 使用する TrainConfig 名  |
| `--exp-name=test_picture_env` | 実験名                 |
| `--overwrite`                 | 同じ実験名の出力がある場合に上書きする |

---

### 出力先

学習結果は以下に保存されます。

```text
/home/${USER}/docker_VLA/workspace/openpi/checkpoints/
└── pi0_picture_env_lora
    └── test_picture_env
        └── 10000
```

`10000` は学習ステップ数を表します。
`num_train_steps=10000` の場合，最終チェックポイントは `10000` に保存されます。

---

## 8. Policy Server の起動

Fine-Tuning 後のチェックポイントを用いて，OpenPI の Policy Server を起動します。

Policy Server を起動することで，別の Python スクリプトや PictureEnv 側の client から，学習済み π0 policy に action を問い合わせることができます。

### openpi-client のインストール

Policy Server に client 側から接続するために，`openpi-client` を editable install します。


```bash
pip install -e ~/workspace/openpi/packages/openpi-client
```

### Policy Server 起動コマンド

Docker 内で OpenPI ディレクトリに移動します。

```bash
cd /home/${USER}/docker_VLA/workspace/openpi
```

以下のコマンドで Policy Server を起動します。

```bash
uv run python scripts/serve_policy.py \
    --default-prompt "pick up the object and place it at the goal" \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi0_picture_env_lora \
    --policy.dir ./checkpoints/pi0_picture_env_lora/test_picture_env/10000
```

### 正常起動時の表示

正常に起動すると，以下のような表示が出ます。

```text
INFO:websockets.server:server listening on 0.0.0.0:8000
```

この表示が出ていれば，Policy Server は起動しています。

### サーバ停止

Policy Server を停止する場合は，起動しているターミナルで以下を押します。

```text
Ctrl + C
```
