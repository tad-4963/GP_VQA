import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MedicalCXRVQADataset(Dataset):
    def __init__(self, csv_file_path, tokenizer_name="albert-base-v2", max_length=128):
        # Map 6 danh mục ý định từ dataset
        self.intent_map = {
            "presence": 0,
            "abnormality": 1,
            "location": 2,
            "view": 3,   
            "type": 4,   
            "level": 5   
        }
        
        # Đọc file CSV thật
        self.df = pd.read_csv(csv_file_path)
        if "question" not in self.df.columns:
            raise ValueError("CSV khong co cot 'question'")

        # Lam sach cot text/label de train on dinh hon
        self.df["question"] = self.df["question"].fillna("").astype(str).str.strip()
        self.df["question_type"] = (
            self.df.get("question_type", "presence")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Lấy từng dòng trong dataframe
        row = self.df.iloc[idx]
        
        # Đọc câu hỏi
        question = row["question"]
        if not question or question.lower() == "nan":
            question = "unknown"
        
        # Lấy nhãn câu hỏi (dùng .get() với giá trị default là 0 để tránh lỗi nếu format file bị lệch nhẹ)
        q_type = str(row.get("question_type", "presence")).strip().lower()
        intent_label = torch.tensor(self.intent_map.get(q_type, 0), dtype=torch.long)
        
        # Tokenize
        encoding = self.tokenizer(
            question,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'intent_label': intent_label,
            'dicom_id': str(row.get('dicom_id', 'unknown'))
        }

    def get_label_counts(self):
        return self.df["question_type"].value_counts()

if __name__ == "__main__":
    csv_path = "data/medical_cxr/medical-cxr-vqa-questions.csv" 
    
    print("Đang đọc file CSV khổng lồ, đợi chút nhé...")
    try:
        dataset = MedicalCXRVQADataset(csv_path, max_length=32)
        print(f"Thành công! Tổng số câu hỏi trong dataset: {len(dataset)} câu.")
        
        print("\nCác cột trong file CSV:", list(dataset.df.columns))
        
        print("\nDòng dữ liệu đầu tiên:")
        print(dataset.df.iloc[0])
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại {csv_path}. Kiểm tra lại đường dẫn nhé!")