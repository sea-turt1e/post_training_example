"""
QLoRA + DPO 組み合わせ学習スクリプト

概要:
    - QLoRA (4-bit 量子化 + LoRA) を使ってベースモデルをメモリ効率よく読み込み
    - DPO (Direct Preference Optimization) で選好データから強化学習なしに RLHF 相当の学習を行う
    - データは HuggingFace Hub から自動ダウンロード (Anthropic/hh-rlhf)
    - ローカル CPU / GPU どちらでも動作するよう設計

必要なライブラリのインストール:
    pip install -r requirements.txt

使い方:
    python train_qlora_dpo.py

ローカル実行のヒント:
    - GPU (VRAM 8GB 以上推奨) があると学習が高速化されます。
    - CPU のみの場合は load_in_4bit=False, use_cpu=True に変更してください。
    - モデルサイズは facebook/opt-125m (125M パラメータ) で手軽に試せます。
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from trl import DPOConfig, DPOTrainer

# ---------------------------------------------------------------------------
# ハイパーパラメータ設定
# ---------------------------------------------------------------------------

@dataclass
class ScriptConfig:
    # --- モデル ---
    model_name: str = field(
        default="facebook/opt-125m",
        metadata={"help": "ベースモデルの HuggingFace モデル ID"},
    )

    # --- データ ---
    dataset_name: str = field(
        default="Anthropic/hh-rlhf",
        metadata={"help": "HuggingFace Hub 上の選好データセット名"},
    )
    max_samples: Optional[int] = field(
        default=1000,
        metadata={"help": "使用するサンプル数の上限 (None で全件使用)"},
    )
    max_eval_samples: Optional[int] = field(
        default=200,
        metadata={"help": "評価に使用するサンプル数の上限 (None で全件使用)"},
    )
    max_length: int = field(
        default=512,
        metadata={"help": "トークン列の最大長"},
    )

    # --- QLoRA ---
    load_in_4bit: bool = field(
        default=True,
        metadata={"help": "4-bit 量子化を有効にする (GPU 必要)"},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA のランク"},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA のスケーリング係数"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA の Dropout 率"},
    )

    # --- 学習 ---
    output_dir: str = field(
        default="./output",
        metadata={"help": "モデルの保存先ディレクトリ"},
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "学習エポック数"},
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "デバイスごとのバッチサイズ"},
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "勾配累積ステップ数"},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "学習率"},
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "DPO の温度パラメータ β"},
    )


# ---------------------------------------------------------------------------
# データ前処理
# ---------------------------------------------------------------------------

def extract_prompt(text: str) -> str:
    """Anthropic/hh-rlhf の chosen/rejected テキストから Human のプロンプト部分を抜き出す。"""
    # フォーマット: "Human: <prompt>\n\nAssistant: <response>"
    if "\n\nAssistant:" in text:
        return text.split("\n\nAssistant:")[0]
    return text


def extract_response(text: str) -> str:
    """Anthropic/hh-rlhf の chosen/rejected テキストから Assistant の応答部分を抜き出す。"""
    if "\n\nAssistant:" in text:
        return text.split("\n\nAssistant:", 1)[1].strip()
    return text


def preprocess_dataset(dataset, max_samples: Optional[int] = None):
    """
    DPOTrainer が期待する {"prompt", "chosen", "rejected"} 形式に変換する。

    Anthropic/hh-rlhf の各行:
        - chosen:   "Human: ...\n\nAssistant: <良い応答>"
        - rejected: "Human: ...\n\nAssistant: <悪い応答>"
    """
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def _map(example):
        prompt = extract_prompt(example["chosen"])
        chosen_response = extract_response(example["chosen"])
        rejected_response = extract_response(example["rejected"])
        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }

    return dataset.map(_map, remove_columns=dataset.column_names)


# ---------------------------------------------------------------------------
# モデル・トークナイザの準備
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(config: ScriptConfig):
    """QLoRA 設定でモデルとトークナイザを読み込む。"""
    use_cuda = torch.cuda.is_available()
    # Macを Mシリーズ使用する場合
    if not use_cuda:
        use_cuda = torch.backends.mps.is_available()
    print(f"使用デバイス: {'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'}")

    # 4-bit 量子化の設定 (QLoRA の核心部分)
    bnb_config = None
    if config.load_in_4bit and use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # NF4 量子化 (QLoRA 論文推奨)
            bnb_4bit_compute_dtype=torch.float16,  # 計算は float16 で実行
            bnb_4bit_use_double_quant=True,        # Double Quantization でさらに節約
        )
    elif config.load_in_4bit and not use_cuda:
        print("GPU が見つかりません。CPU モードで float32 読み込みに切り替えます。")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
    )

    # kbit 学習向けの前処理 (勾配チェックポイント有効化、LayerNorm を float32 に固定など)
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # LoRA アダプタの設定
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # OPT モデルの Attention 投影層を対象にする。Loraの元々の設定。
        target_modules=["q_proj", "v_proj"], 
        # 他に以下がある。
        # 入力
        # └── Self-Attention ──── q_proj, k_proj, v_proj, o_proj
        # └── FFN (MLP) ────────── gate_proj, up_proj, down_proj
        # └── 出力ヘッド ──────── lm_head (embeddingに近い)

    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # トークナイザ
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    config = ScriptConfig()

    print("=" * 60)
    print("QLoRA + DPO 学習スクリプト")
    print("=" * 60)
    print(f"モデル      : {config.model_name}")
    print(f"データセット: {config.dataset_name}")
    print(f"4-bit 量子化: {config.load_in_4bit}")
    print(f"LoRA ランク : {config.lora_r}")
    print(f"β (DPO)     : {config.beta}")
    print("=" * 60)

    # 1. データセットの読み込みと前処理
    print("\n[1/4] データセットを読み込み中...")
    raw = load_dataset(config.dataset_name, split="train")
    train_dataset = preprocess_dataset(raw, max_samples=config.max_samples)
    print(f"  学習サンプル数: {len(train_dataset)}")
    print(f"  カラム       : {train_dataset.column_names}")

    # 評価データ (小規模)
    raw_eval = load_dataset(config.dataset_name, split="test")
    eval_dataset = preprocess_dataset(raw_eval, max_samples=config.max_eval_samples)

    # 2. モデルとトークナイザの準備
    print("\n[2/4] モデルを読み込み中 (QLoRA 設定)...")
    model, tokenizer = load_model_and_tokenizer(config)

    # 3. DPO 学習設定
    print("\n[3/4] DPO 学習を設定中...")
    use_cuda = torch.cuda.is_available()

    dpo_config = DPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        beta=config.beta,
        max_length=config.max_length,
        max_prompt_length=config.max_length // 2,
        fp16=use_cuda,                      # GPU 時は float16 混合精度
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",                   # wandb 等への送信を無効化
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # 4. 学習開始
    print("\n[4/4] 学習を開始します...")
    trainer.train()

    # 5. モデルの保存
    print(f"\n学習完了。モデルを {config.output_dir} に保存します...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("保存完了！")

    # 6. 簡単な動作確認推論
    print("\n--- 簡単な推論テスト ---")
    model.eval()
    # 学習データと同じ "Human: ...\n\nAssistant:" フォーマットで推論
    prompt = "Human: What is the capital of France?\n\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt")
    if use_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"入力 : {prompt}")
    print(f"出力 : {generated}")


if __name__ == "__main__":
    main()
