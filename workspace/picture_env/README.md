# BC Project Refactor

FetchPickAndPlace のデータ収集，BC学習，評価を分離したプロジェクトです．
基本の流れは，

```text
1. データ収集
2. 学習
3. 評価
```

です．
モデルは `config.yaml` の `model.name` を変えるだけで切り替えます．

---

## 0. 準備

プロジェクトのディレクトリに移動します．

```bash
cd bc_project_refactor
```

依存関係を入れます．

```bash
pip install -r requirements.txt
```

`yaml` がないと言われた場合は，追加で以下を実行します．

```bash
pip install pyyaml
```

---

## 1. データ収集

まず最初にこれを実行します．

```bash
python scripts/collect_data.py --config config.yaml
```

成功すると，以下が作成されます．

```text
log/dataset.npz
```

この `dataset.npz` が，学習に使うデモデータです．

### データ収集で見る設定

`config.yaml` のここを主に見ます．

```yaml
collect:
  n_demo: 100
  base_object_pos: [1.30, 0.50, 0.42]
  goal_pos: [1.30, 1.00, 0.42]
  x_min: 1.10
  x_max: 1.50
  sample_y: false
  y_min: 0.45
  y_max: 0.55
  show: false
  action_noise_std: 0.03
```

現在は，object の x 座標を `x_min〜x_max` の範囲からランダムに選びます．
等間隔ではありません．

```text
x = uniform(x_min, x_max)
```

`sample_y: true` にすると，y 座標も `y_min〜y_max` の範囲からランダムに選びます．

```yaml
collect:
  sample_y: true
```

### 動きを遅くしたい場合

`scripted_policy` を変更します．

```yaml
scripted_policy:
  kp: 5.0
  max_action_xyz: 0.5
  grasp_hold_steps: 15
  place_hold_steps: 10
```

さらにゆっくり動かすなら，例えばこうします．

```yaml
scripted_policy:
  kp: 3.0
  max_action_xyz: 0.3
```

掴みを安定させたい場合は，`grasp_hold_steps` を増やします．

```yaml
scripted_policy:
  grasp_hold_steps: 25
```

---

## 2. 学習

データ収集が終わったら，学習します．

```bash
python scripts/train_bc.py --config config.yaml
```

成功すると，以下が保存されます．

```text
log/bc_model/best_bc_model.pt
```

---

## 3. 評価

学習済みモデルを使って動作確認します．

```bash
python scripts/test_bc.py --config config.yaml
```

評価では，seen位置とunseen位置で動かします．
設定はここです．

```yaml
test:
  n_seen_pos: 5
  sleep_sec: 0.02
  run_seen: true
  run_unseen: true
```

---

## 4. モデルの切り替え

`config.yaml` のここだけを変えます．

```yaml
model:
  name: cnn_bc
```

使えるモデル名は以下です．

```text
cnn_bc
transformer_bc
act_bc
diffusion_policy
cvae_bc
```

### CNN BC

一番基本のモデルです．

```yaml
model:
  name: cnn_bc
```

実行例です．

```bash
python scripts/train_bc.py --config configs/cnn_bc.yaml
python scripts/test_bc.py --config configs/cnn_bc.yaml
```

### Transformer BC

画像特徴とロボット状態を使って action を出す Transformer 版です．

```yaml
model:
  name: transformer_bc
```

実行例です．

```bash
python scripts/train_bc.py --config configs/transformer_bc.yaml
python scripts/test_bc.py --config configs/transformer_bc.yaml
```

### ACT BC

action を1ステップではなく，複数ステップの chunk として出すモデルです．

```yaml
model:
  name: act_bc
  action_horizon: 16
```

データ側も同じ horizon にします．

```yaml
data:
  action_horizon: 16
```

実行例です．

```bash
python scripts/train_bc.py --config configs/act_bc.yaml
python scripts/test_bc.py --config configs/act_bc.yaml
```

### Diffusion Policy

Diffusion Policy の最小版です．
現在はまず使いやすさ優先で，1 step action を生成する形です．
本格的にする場合は，action chunk を denoise する形へ拡張します．

```yaml
model:
  name: diffusion_policy
  diffusion_steps: 50
```

実行例です．

```bash
python scripts/train_bc.py --config configs/diffusion_policy.yaml
python scripts/test_bc.py --config configs/diffusion_policy.yaml
```

### CVAE BC

同じ観測に対して複数の行動候補を持たせたい場合に使うモデルです．

```yaml
model:
  name: cvae_bc
  latent_dim: 32
  kl_weight: 0.01
```

実行例です．

```bash
python scripts/train_bc.py --config configs/cvae_bc.yaml
python scripts/test_bc.py --config configs/cvae_bc.yaml
```

---

## 5. よく使う実行順

最初に CNN で確認します．

```bash
python scripts/collect_data.py --config config.yaml
python scripts/train_bc.py --config configs/cnn_bc.yaml
python scripts/test_bc.py --config configs/cnn_bc.yaml
```

次に Transformer を試します．

```bash
python scripts/train_bc.py --config configs/transformer_bc.yaml
python scripts/test_bc.py --config configs/transformer_bc.yaml
```

ACT を試します．

```bash
python scripts/train_bc.py --config configs/act_bc.yaml
python scripts/test_bc.py --config configs/act_bc.yaml
```

Diffusion Policy を試します．

```bash
python scripts/train_bc.py --config configs/diffusion_policy.yaml
python scripts/test_bc.py --config configs/diffusion_policy.yaml
```

CVAE を試します．

```bash
python scripts/train_bc.py --config configs/cvae_bc.yaml
python scripts/test_bc.py --config configs/cvae_bc.yaml
```

---

## 6. ファイル構成

```text
bc_project_refactor/
  config.yaml
  configs/
    cnn_bc.yaml
    transformer_bc.yaml
    act_bc.yaml
    diffusion_policy.yaml
    cvae_bc.yaml

  scripts/
    collect_data.py
    train_bc.py
    test_bc.py

  bc_project/
    config.py
    dataset.py
    preprocess.py
    utils.py

    envs/
      fetch_pick_place.py
      scripted_policy.py

    models/
      registry.py
      common.py
      cnn_bc.py
      transformer_bc.py
      act_bc.py
      diffusion_policy.py
      cvae_bc.py
```

---

## 7. 変更した主な点

### 変更済み

- データ収集を等間隔から範囲ランダムサンプリングに変更．
- `config.yaml` からデータ収集，学習，モデル，評価を設定できるように変更．
- `CNN / Transformer / ACT / Diffusion Policy / CVAE` を `model.name` で切り替え可能に変更．
- 掴み保持のために `grasp_hold_steps` を追加．
- 置く前の保持のために `place_hold_steps` を追加．
- 動きを遅くするために `kp` と `max_action_xyz` を設定化．

### そのまま残している点

- 環境は `FetchPickAndPlaceDense-v4` のまま．
- 保存形式は `dataset.npz` のまま．
- 基本の観測は `image + gripper_pos` のまま．
- action は `[dx, dy, dz, gripper]` の4次元のまま．

---

## 8. エラー対応

### `ModuleNotFoundError: No module named 'yaml'`

```bash
pip install pyyaml
```

### camera が見つからない場合

データ収集時に available cameras が表示されます．
`config.yaml` のここを，表示された camera 名に合わせます．

```yaml
env:
  camera_name: external_camera_0
```

### dataset がないと言われる場合

先にデータ収集を実行します．

```bash
python scripts/collect_data.py --config config.yaml
```

### model がないと言われる場合

先に学習を実行します．

```bash
python scripts/train_bc.py --config config.yaml
```
