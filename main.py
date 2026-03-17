import json
import os
import re

import chardet
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ==================== A. 通用工具 ====================
def read_csv_auto_encoding(file_path: str) -> pd.DataFrame:
    """自动检测编码并读取 CSV，避免编码导致的读文件失败。"""
    with open(file_path, "rb") as f:
        raw = f.read(100000)
        result = chardet.detect(raw)

    encoding = result.get("encoding") or "utf-8"
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except Exception:
        return pd.read_csv(file_path, encoding="utf-8", errors="ignore")


# ==================== B. 核心类：统一实验框架 ====================
class RestaurantReviewPredictor:


    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", output_dir="./finetuned_model_lora"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[Init] device={self.device}, model={model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    # ---------- 1) 推理相关 ----------
    def create_prompt(self, review_text: str, few_shot_examples=None) -> str:
        """统一提示模板：保证 Baseline/N-shot 可公平比较。"""
        prompt = """You are a restaurant review rating predictor. Analyze the review and predict the rating from 1 to 5 stars.

Rating Scale:
1 star - Terrible experience, major problems
2 stars - Poor, significant issues
3 stars - Average, mixed feelings
4 stars - Good, mostly positive
5 stars - Excellent, highly recommended

Output ONLY the rating in format: \"X star\" (where X is 1-5).
"""

        if few_shot_examples:
            prompt += "\nExamples:\n"
            for review, rating in few_shot_examples:
                prompt += f"Review: {review}\nRating: {rating}\n\n"

        prompt += f"\nReview: {review_text}\nRating:"
        return prompt

    def extract_rating(self, text: str) -> str:
        """统一评分解析：把自由生成归一到 1~5 star。"""
        text = text.lower().strip()

        for i in range(1, 6):
            if f"{i} star" in text:
                return f"{i} star"

        match = re.search(r"stars?:?\s*([1-5])", text)
        if match:
            return f"{match.group(1)} star"

        for i, word in enumerate(["one", "two", "three", "four", "five"], start=1):
            if re.search(rf"\b{word}\b", text) or re.search(rf"\b{i}\b", text):
                return f"{i} star"

        return "3 star"

    def predict_one(self, review_text: str, few_shot_examples=None) -> str:
        prompt = self.create_prompt(review_text, few_shot_examples)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                temperature=0.1,
                do_sample=True,
                top_p=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip().split("\n")[0].strip()
        return self.extract_rating(response)

    def select_few_shot_examples(self, train_df: pd.DataFrame, n_shots=5):
        """按类别分层 + 中等长度优先，提升few-shot示例代表性。"""
        examples = []
        for rating in range(1, 6):
            rating_str = f"{rating} star"
            candidates = train_df[train_df["Rating"] == rating_str].copy()
            if len(candidates) == 0:
                continue

            candidates["review_length"] = candidates["Review"].str.len()
            moderate = candidates[(candidates["review_length"] >= 100) & (candidates["review_length"] <= 400)]
            if len(moderate) > 0:
                candidates = moderate

            median_len = candidates["review_length"].median()
            best_idx = (candidates["review_length"] - median_len).abs().idxmin()
            selected = candidates.loc[best_idx]
            examples.append((selected["Review"], rating_str))

        return examples[:n_shots]

    def run_inference(self, test_df: pd.DataFrame, few_shot_examples=None) -> pd.DataFrame:
        """统一推理入口：Baseline 传 None，N-shot 传示例。"""
        records = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
            pred = self.predict_one(row["Review"], few_shot_examples=few_shot_examples)
            records.append({"Review_id": row["Review_id"], "Predicted_Rating": pred})
        return pd.DataFrame(records)

    # ---------- 2) LoRA 微调相关 ----------
    def prepare_train_dataset(self, train_csv: str, max_length=512) -> Dataset:
        """把训练数据转成指令格式并tokenize，供CausalLM训练。"""
        df = read_csv_auto_encoding(train_csv)

        texts = []
        for _, row in df.iterrows():
            text = f"""You are a restaurant review rating predictor. Analyze the review and predict the rating from 1 to 5 stars.

Rating Scale:
1 star - Terrible experience, major problems
2 stars - Poor, significant issues
3 stars - Average, mixed feelings
4 stars - Good, mostly positive
5 stars - Excellent, highly recommended

Output ONLY the rating in format: \"X star\" (where X is 1-5).

Review: {row['Review']}
Rating: {row['Rating']}"""
            texts.append(text)

        dataset = Dataset.from_dict({"text": texts})

        def tokenize_fn(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    def setup_lora(self, r=16, alpha=32, dropout=0.05):
        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_cfg)

    def train_lora(self, tokenized_dataset: Dataset, num_epochs=3, batch_size=4, learning_rate=2e-4):
        split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        val_ds = split["test"]

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
        )
        trainer.train()

        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

    def load_lora_adapter(self):
        """加载已经训练好的LoRA适配器。"""
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base_model, self.output_dir)
        self.model.eval()

    # ---------- 3) 统一评估 ----------
    @staticmethod
    def evaluate(pred_df: pd.DataFrame, answer_df: pd.DataFrame, verbose=True):
        merged = pred_df.merge(answer_df, on="Review_id")
        y_true = merged["Rating"].tolist()
        y_pred = merged["Predicted_Rating"].tolist()

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, labels=["1 star", "2 star", "3 star", "4 star", "5 star"], average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, labels=["1 star", "2 star", "3 star", "4 star", "5 star"], average="weighted"),
        }

        if verbose:
            print("\n[Metrics]")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1-Macro: {metrics['f1_macro']:.4f}")
            print(f"F1-Weighted: {metrics['f1_weighted']:.4f}")
            print("\n[Classification Report]")
            print(classification_report(y_true, y_pred, labels=["1 star", "2 star", "3 star", "4 star", "5 star"]))

        return metrics


# ==================== C. 两个运行模式 ====================
def run_quick_compare(train_csv, test_csv, answer_csv):
    """快速对比：Baseline vs 5-shot（不训练）。"""
    predictor = RestaurantReviewPredictor()

    train_df = read_csv_auto_encoding(train_csv)
    test_df = read_csv_auto_encoding(test_csv)
    answer_df = read_csv_auto_encoding(answer_csv)

    print("\n===== Baseline =====")
    baseline_pred = predictor.run_inference(test_df, few_shot_examples=None)
    baseline_metrics = predictor.evaluate(baseline_pred, answer_df)

    print("\n===== 5-shot =====")
    examples = predictor.select_few_shot_examples(train_df, n_shots=5)
    nshot_pred = predictor.run_inference(test_df, few_shot_examples=examples)
    nshot_metrics = predictor.evaluate(nshot_pred, answer_df)

    result = {"baseline": baseline_metrics, "nshot": nshot_metrics}
    with open("quick_compare_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n结果已保存: quick_compare_results.json")


def run_full_compare(train_csv, test_csv, answer_csv, do_train=False):
    """完整对比：Baseline + 5-shot + LoRA（训练或加载）。"""
    predictor = RestaurantReviewPredictor(output_dir="./finetuned_model_lora")

    train_df = read_csv_auto_encoding(train_csv)
    test_df = read_csv_auto_encoding(test_csv)
    answer_df = read_csv_auto_encoding(answer_csv)

    print("\n===== Baseline =====")
    baseline_pred = predictor.run_inference(test_df)
    baseline_metrics = predictor.evaluate(baseline_pred, answer_df)

    print("\n===== 5-shot =====")
    examples = predictor.select_few_shot_examples(train_df, n_shots=5)
    nshot_pred = predictor.run_inference(test_df, few_shot_examples=examples)
    nshot_metrics = predictor.evaluate(nshot_pred, answer_df)

    print("\n===== LoRA =====")
    if do_train:
        tokenized = predictor.prepare_train_dataset(train_csv)
        predictor.setup_lora(r=16, alpha=32, dropout=0.05)
        predictor.train_lora(tokenized)
    else:
        if not os.path.exists(predictor.output_dir):
            print("未找到已训练LoRA目录，请先用 --full-train 训练。")
            return
        predictor.load_lora_adapter()

    lora_pred = predictor.run_inference(test_df)
    lora_metrics = predictor.evaluate(lora_pred, answer_df)

    result = {"baseline": baseline_metrics, "nshot": nshot_metrics, "lora": lora_metrics}
    with open("full_compare_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n结果已保存: full_compare_results.json")


if __name__ == "__main__":
    import sys

    TRAIN_CSV = "review_train.csv"
    TEST_CSV = "review_test.csv"
    ANSWER_CSV = "test_answer.csv"

    if len(sys.argv) < 2:
        print("Usage: python main.py [--quick | --full-load | --full-train]")
    elif sys.argv[1] == "--quick":
        run_quick_compare(TRAIN_CSV, TEST_CSV, ANSWER_CSV)
    elif sys.argv[1] == "--full-load":
        run_full_compare(TRAIN_CSV, TEST_CSV, ANSWER_CSV, do_train=False)
    elif sys.argv[1] == "--full-train":
        run_full_compare(TRAIN_CSV, TEST_CSV, ANSWER_CSV, do_train=True)
    else:
        print("Usage: python main.py [--quick | --full-load | --full-train]")
