import torch  # 딥러닝 모델 구성과 텐서 계산을 위해 사용합니다.
import torch.nn as nn  # 신경망 레이어와 손실함수를 사용하기 위해 불러옵니다.
from torch.utils.data import DataLoader, TensorDataset  # 데이터를 배치 단위로 나눠 학습하기 위한 도구입니다.
from pathlib import Path  # 현재 스크립트 폴더 기준으로 파일을 읽고 쓰기 위해 사용합니다.

BASE_DIR = Path(__file__).resolve().parent  # 이 스크립트가 있는 폴더의 실제 경로를 구합니다.

# 1. 저장된 데이터 로딩
print("데이터를 불러옵니다...", flush=True)  # 데이터 로딩이 시작되었음을 즉시 출력합니다.
dataset = torch.load(BASE_DIR / 'processed_dataset.pt', weights_only=False)  # 전처리 단계에서 저장한 데이터셋 파일을 읽습니다.
X_train, Y_train = dataset['X_train'], dataset['Y_train']  # 학습용 입력과 정답 시퀀스를 꺼냅니다.
X_valid_0, Y_valid_0 = dataset['X_valid_0'], dataset['Y_valid_0']  # 정상 데이터만으로 만든 검증셋을 꺼냅니다.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU가 가능하면 GPU, 아니면 CPU에서 학습하도록 설정합니다.

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)  # 학습 데이터를 32개씩 섞어서 공급하는 로더를 만듭니다.
valid_loader = DataLoader(TensorDataset(X_valid_0, Y_valid_0), batch_size=32, shuffle=False)  # 검증 데이터는 순서를 유지한 채 32개씩 공급합니다.


# 2. 모델 구조 정의 (가이드북 구조 반영)
class LSTM_AE(nn.Module):  # LSTM 기반 오토인코더 모델 클래스를 정의합니다.
    def __init__(self, n_features, seq_len):
        super(LSTM_AE, self).__init__()  # 부모 클래스 초기화를 수행합니다.
        self.seq_len = seq_len  # 나중에 디코더 입력을 반복 생성할 때 사용할 시퀀스 길이를 저장합니다.
        # Encoder
        self.enc1 = nn.LSTM(input_size=n_features, hidden_size=64, batch_first=True)  # 첫 번째 인코더 LSTM은 입력 5개 특징을 64차원 은닉상태로 변환합니다.
        self.enc2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)  # 두 번째 인코더 LSTM은 64차원을 32차원 잠재표현으로 압축합니다.
        # Decoder
        self.dec1 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)  # 첫 번째 디코더 LSTM은 잠재표현을 복원용 시퀀스로 펼칩니다.
        self.dec2 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)  # 두 번째 디코더 LSTM은 표현 차원을 다시 64로 늘립니다.
        self.out = nn.Linear(64, n_features)  # 마지막 선형층은 각 시점 출력을 원래 특징 수 5개로 바꿉니다.

    def forward(self, x):
        x, _ = self.enc1(x)  # 입력 시퀀스를 첫 번째 인코더에 통과시켜 중간 표현을 얻습니다.
        x, (h_n, _) = self.enc2(x)  # 두 번째 인코더를 거쳐 최종 은닉상태 h_n을 얻습니다.

        # RepeatVector 역할 (마지막 Hidden State를 시퀀스 길이만큼 복제)
        x = h_n.transpose(0, 1).repeat(1, self.seq_len, 1)  # 마지막 은닉상태 하나를 시퀀스 길이만큼 복제해 디코더 입력으로 사용합니다.

        x, _ = self.dec1(x)  # 복제된 잠재표현을 첫 번째 디코더에 통과시킵니다.
        x, _ = self.dec2(x)  # 이어서 두 번째 디코더에 통과시킵니다.
        x = self.out(x)  # 각 시점의 출력을 최종 특징 차원으로 변환합니다.
        return x  # 복원 혹은 예측된 시퀀스를 반환합니다.


model = LSTM_AE(n_features=5, seq_len=20).to(device)  # 특징 5개, 시퀀스 길이 20인 모델을 만들고 지정 장치로 보냅니다.
criterion = nn.MSELoss()  # 예측 시퀀스와 정답 시퀀스 차이를 평균제곱오차로 계산합니다.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 가중치 업데이트는 Adam 최적화 알고리즘을 사용합니다.

# 학습률 감소(ReduceLROnPlateau) 스케줄러
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8)  # 검증 손실이 좋아지지 않으면 학습률을 줄여 안정적으로 학습합니다.

# 3. 모델 학습 루프
print(f"[{device}] 환경에서 학습을 시작합니다...", flush=True)  # 실제 학습이 어느 장치에서 돌아가는지 출력합니다.
epochs = 50  # 전체 학습 반복 횟수를 50번으로 설정합니다.

train_losses, valid_losses = [], []  # 에폭별 학습/검증 손실을 기록할 리스트입니다.
best_val_loss = float('inf')  # 가장 좋은 검증 손실의 초기값을 매우 큰 수로 둡니다.
patience_counter = 0  # Early Stopping을 위한 참을 횟수 카운터입니다.
early_stop_patience = 17  # 검증 성능이 좋아지지 않아도 최대 17번까지는 기다립니다.

for epoch in range(epochs):  # 전체 에폭 수만큼 학습을 반복합니다.
    model.train()  # 모델을 학습 모드로 전환합니다.
    train_loss = 0  # 현재 에폭의 누적 학습 손실을 0으로 시작합니다.
    for batch_x, batch_y in train_loader:  # 학습 데이터를 배치 단위로 하나씩 가져옵니다.
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 배치 데이터를 CPU 또는 GPU 장치로 보냅니다.

        optimizer.zero_grad()  # 이전 배치에서 계산된 기울기를 초기화합니다.
        output = model(batch_x)  # 현재 배치 입력을 모델에 넣어 예측 시퀀스를 얻습니다.
        loss = criterion(output, batch_y)  # 예측 결과와 실제 정답 사이 손실을 계산합니다.
        loss.backward()  # 손실을 기준으로 역전파하여 기울기를 계산합니다.
        optimizer.step()  # 계산된 기울기로 모델 가중치를 업데이트합니다.
        train_loss += loss.item()  # 현재 배치 손실값을 누적합니다.

    # 검증 단계
    model.eval()  # 모델을 평가 모드로 전환합니다.
    valid_loss = 0  # 현재 에폭의 누적 검증 손실을 0으로 시작합니다.
    with torch.no_grad():  # 검증 중에는 기울기를 계산하지 않아 속도와 메모리를 절약합니다.
        for batch_x, batch_y in valid_loader:  # 검증 데이터도 배치 단위로 가져옵니다.
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 검증 배치도 같은 장치로 보냅니다.
            output = model(batch_x)  # 검증 입력에 대한 예측을 수행합니다.
            loss = criterion(output, batch_y)  # 검증 배치 손실을 계산합니다.
            valid_loss += loss.item()  # 검증 손실을 누적합니다.

    avg_train_loss = train_loss / len(train_loader)  # 에폭 전체 평균 학습 손실을 구합니다.
    avg_valid_loss = valid_loss / len(valid_loader)  # 에폭 전체 평균 검증 손실을 구합니다.
    train_losses.append(avg_train_loss)  # 평균 학습 손실을 기록합니다.
    valid_losses.append(avg_valid_loss)  # 평균 검증 손실을 기록합니다.

    scheduler.step(avg_valid_loss)  # 현재 검증 손실을 보고 학습률 조절 여부를 결정합니다.

    print(  # 각 에폭의 학습 상태를 보기 쉽게 출력합니다.
        f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.5f} | Valid Loss: {avg_valid_loss:.5f}",
        flush=True
    )

    # Early Stopping 로직
    if avg_valid_loss < best_val_loss - 0.00001:  # 현재 검증 손실이 이전 최고 기록보다 충분히 좋아졌는지 확인합니다.
        best_val_loss = avg_valid_loss  # 가장 좋은 검증 손실 값을 갱신합니다.
        patience_counter = 0  # 성능이 좋아졌으므로 참을 횟수 카운터를 다시 0으로 리셋합니다.
        torch.save(model.state_dict(), BASE_DIR / 'best_lstm_ae.pth')  # 현재까지 가장 좋은 모델 가중치를 파일로 저장합니다.
    else:
        patience_counter += 1  # 성능 향상이 없으면 기다린 횟수를 1 증가시킵니다.
        if patience_counter >= early_stop_patience:  # 더 이상 기다릴 한계를 넘었는지 확인합니다.
            print("Early Stopping 트리거 발동! 학습을 조기 종료합니다.", flush=True)  # 학습 중단 사실을 출력합니다.
            break  # 학습 루프를 조기 종료합니다.

print("학습 완료! (best_lstm_ae.pth 저장됨)", flush=True)  # 전체 학습 종료 메시지를 출력합니다.
