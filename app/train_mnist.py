import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

from .model import CNN   # 위에서 만든 CNN 모델 불러오기

def train_mnist(
        batch_size: int = 128,      # 한 번에 학습시키는 이미지 개수
        epochs: int = 5,            # 전체 데이터 5번 학습
        lr: float = 1e-3,           # learning rate (학습 속도)
        save_path: str = "models/mnist_cnn.pth",
):
    # GPU 또는 CPU 선택
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 데이터셋 이미지 변환 설정
    transform = transforms.Compose([
        transforms.ToTensor(),                    # Tensor 변환
        transforms.Normalize((0.1307,), (0.3081,)),  # 정규화
    ])

    # MNIST 학습용 데이터 다운받기
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    # 데이터셋을 batch 단위로 불러오는 객체
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 모델 생성
    model = CNN().to(device)
    # 손실 함수 (CrossEntropy: 분류문제용)
    criterion = nn.CrossEntropyLoss()
    # 옵티마이저(Adam): 학습하는 알고리즘
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 학습 반복(epoch)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            # 입력 이미지/정답을 장치(device)로 이동
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()         # 이전 배치의 gradient 초기화
            outputs = model(imgs)         # 모델 실행 → 결과 출력
            loss = criterion(outputs, labels)  # 결과와 정답 비교
            loss.backward()               # 오차 역전파
            optimizer.step()              # 가중치 업데이트

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}/{epochs}] loss = {avg_loss:.4f}")

    Path("models").mkdir(parents=True, exist_ok=True)
    # 학습 완료된 모델 weight 저장
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_mnist()
