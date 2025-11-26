import torch
from torch import nn

# CNN(합성곱 신경망) 모델 정의
# 손글씨 숫자를 분류하는 목적(0~9)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 이미지 특징을 추출하는 파트(Convolution)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),  # 입력 1채널, 출력 32채널, 필터 3x3, stride=1
            nn.ReLU(),               # 활성화 함수
            nn.MaxPool2d(2),         # 다운샘플링 → 크기 줄이기

            nn.Conv2d(32, 64, 3, 1), # 또 한 번 특징 추출
            nn.ReLU(),
            nn.MaxPool2d(2),         # 또 다운샘플링
        )

        # 특징을 기반으로 숫자 예측하는 Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Flatten(),             # 2D 이미지 → 1D 벡터
            nn.Linear(64 * 5 * 5, 128), # 입력 1600 → 128
            nn.ReLU(),
            nn.Linear(128, 10),       # 출력: 0~9 → 총 10개 클래스
        )

    def forward(self, x):
        # forward() = 모델 실행(추론 pipeline)
        x = self.conv(x)
        x = self.fc(x)
        return x


# 훈련된 모델을 불러오는 함수
def load_trained_model(weight_path: str, device=None) -> nn.Module:
    if device is None:
        # GPU 있으면 GPU / 없으면 CPU 사용
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)

    # 저장된 가중치 파일을 불러옴 (예: models/mnist_cnn.pth)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()  # 모델을 inference(추론)용 모드로 전환
    return model
