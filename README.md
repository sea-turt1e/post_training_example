# post_training_example

QLoRA と DPO を組み合わせたポストトレーニング実装サンプルです。

## 概要

このリポジトリでは、以下の技術を組み合わせた言語モデルのファインチューニング手法を実装しています。

| 技術 | 内容 |
|------|------|
| **QLoRA** | 4-bit 量子化 + LoRA アダプタにより、少ない VRAM でファインチューニング |
| **DPO** | 人間の選好データから直接ポリシーを最適化（強化学習不要の RLHF 代替手法）|

データは HuggingFace Hub の [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf) を自動ダウンロードして使用するため、ローカル環境でそのまま実行できます。

## ファイル構成

```
.
├── train_qlora_dpo.py   # メイン学習スクリプト
├── requirements.txt     # 依存ライブラリ一覧
└── README.md
```

## 環境構築

Python 3.10 以上を推奨します。

```bash
pip install -r requirements.txt
```

> **CPU のみの環境でも動作します。** その場合は `train_qlora_dpo.py` 内の
> `ScriptConfig.load_in_4bit` を `False` に変更してください。
> GPU (VRAM 8GB 以上) がある場合は QLoRA の恩恵を最大限に受けられます。

## 使い方

```bash
python train_qlora_dpo.py
```

デフォルト設定で学習が始まります。カスタマイズしたい場合は `ScriptConfig` の各フィールドを変更してください。

### 主な設定項目

| パラメータ | デフォルト値 | 説明 |
|------------|-------------|------|
| `model_name` | `facebook/opt-125m` | ベースモデル (HuggingFace ID) |
| `dataset_name` | `Anthropic/hh-rlhf` | 選好データセット |
| `max_samples` | `1000` | 使用サンプル数上限 |
| `max_eval_samples` | `200` | 評価に使用するサンプル数上限 |
| `load_in_4bit` | `True` | QLoRA 用 4-bit 量子化 |
| `lora_r` | `16` | LoRA のランク |
| `lora_alpha` | `32` | LoRA のスケーリング係数 |
| `beta` | `0.1` | DPO の温度パラメータ β |
| `num_train_epochs` | `1` | 学習エポック数 |
| `output_dir` | `./output` | モデル保存先 |

## 技術解説

### QLoRA (Quantized Low-Rank Adaptation)

```
ベースモデル (4-bit 量子化で読み込み)
    └── LoRA アダプタ (float16 で学習)
            ├── q_proj
            └── v_proj
```

- `BitsAndBytesConfig` で NF4 量子化 + Double Quantization を有効化
- `prepare_model_for_kbit_training` で kbit 学習向けの前処理を実施
- `LoraConfig` で学習対象の重みを Attention 層に限定

### DPO (Direct Preference Optimization)

```
データ形式:
  prompt   : "Human: <質問>"
  chosen   : "<良い応答>"   (人間が選好する応答)
  rejected : "<悪い応答>"   (人間が選好しない応答)
```

DPO の損失関数:

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_\text{ref}) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$

- β が大きいほど参照モデルからの乖離にペナルティ
- 強化学習（PPO など）が不要で実装がシンプル

## 参考文献

- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [TRL ライブラリ (HuggingFace)](https://github.com/huggingface/trl)
- [PEFT ライブラリ (HuggingFace)](https://github.com/huggingface/peft)
