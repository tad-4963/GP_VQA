import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskALBERTBiLSTM(nn.Module):
    def __init__(self, num_intents=6, num_ner_tags=10, hidden_dim=256):
        super(MultiTaskALBERTBiLSTM, self).__init__()
        
        # 1. Bộ mã hóa ngôn ngữ ALBERT (kế thừa từ Bước 1)
        self.albert = AutoModel.from_pretrained("albert-base-v2")
        albert_out_dim = self.albert.config.hidden_size
        
        # 2. Lớp BiLSTM mô hình hóa sự phụ thuộc không gian của câu hỏi
        self.bilstm = nn.LSTM(
            input_size=albert_out_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Hai nhánh phân loại độc lập (Multi-Task Heads)
        bilstm_out_dim = hidden_dim * 2 # Nhân 2 vì là Bidirectional (2 chiều)
        
        # Nhánh 1: Dự đoán 1 trong 6 Intent (Sự hiện diện, Bất thường, So sánh...)
        self.intent_classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_intents)
        )
        
        # Nhánh 2: Dự đoán nhãn NER cho từng chữ trong câu (Anatomy, Observation...)
        self.ner_classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_ner_tags)
        )

    def forward(self, input_ids, attention_mask, return_ner=True):
        # Đi qua ALBERT
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Đi qua BiLSTM
        lstm_output, _ = self.bilstm(sequence_output)
        
        # Phân nhánh 1: Intent Classification (Dùng token [CLS] ở vị trí đầu tiên đại diện cho cả câu)
        cls_output = lstm_output[:, 0, :]
        intent_logits = self.intent_classifier(cls_output)

        if not return_ner:
            return intent_logits
        
        # Phân nhánh 2: NER (Dự đoán nhãn cho toàn bộ các token trong câu)
        ner_logits = self.ner_classifier(lstm_output)
        
        return intent_logits, ner_logits
if __name__ == "__main__":
    import sys
    import os
    # Thêm thư mục gốc vào path để import được Dataloader
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from src.datasets.cxr_vqa_dataset import MedicalCXRVQADataset
    from torch.utils.data import DataLoader

    # 1. Khởi tạo mô hình
    print("Đang khởi tạo Multi-Task Model...")
    model = MultiTaskALBERTBiLSTM(num_intents=6, num_ner_tags=10) # 6 ý định, giả sử có 10 loại thực thể
    
    # 2. Load 1 batch dữ liệu nhỏ từ file CSV (đường dẫn của bạn)
    csv_path = "data/medical_cxr/medical-cxr-vqa-questions.csv"
    dataset = MedicalCXRVQADataset(csv_path, max_length=32)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # 3. Lấy 1 batch và chạy qua mô hình
    batch = next(iter(dataloader))
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    
    print("\n--- Đưa dữ liệu qua mô hình ---")
    intent_logits, ner_logits = model(input_ids, attention_mask)
    
    print(f"Kích thước Input IDs: {input_ids.shape} -> [batch_size, max_length]")
    print(f"Kích thước Intent Logits: {intent_logits.shape} -> [batch_size, num_intents]")
    print(f"Kích thước NER Logits: {ner_logits.shape} -> [batch_size, max_length, num_ner_tags]")
    print("Thành công! Dữ liệu đã chảy qua não bộ ngôn ngữ.")