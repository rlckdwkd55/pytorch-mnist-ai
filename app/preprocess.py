import cv2
import numpy as np
import torch
from torchvision import transforms

# MNIST 정규화 변환
to_tensor = transforms.Compose([
    transforms.ToTensor(),                    # Tensor 변환
    transforms.Normalize((0.1307,), (0.3081,)),
])

def preprocess_for_mnist(image_bytes: bytes) -> torch.Tensor:
    """
    이미지 데이터를 받아 숫자만 남기고 MNIST 스타일로 변환 후
    모델 입력 준비까지 해서 반환하는 함수
    """

    # 바이트 데이터를 numpy 배열로 변환
    nparr = np.frombuffer(image_bytes, np.uint8)
    # numpy 배열을 OpenCV 이미지로 decode
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("이미지 디코딩 실패")

    # 1) grayscale 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) GaussianBlur + Otsu 임계값 → 숫자 부분만 밝게 함
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 3) 가장 큰 윤곽(contour)을 찾아 숫자만 crop
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        # 가장 큰(= 숫자일 확률 높음) contour 선택
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        digit = thresh[y:y+h, x:x+w]
    else:
        # 숫자 감지 실패하면 그냥 threshold 전체 사용
        digit = thresh

    # 4) 이미지가 세로로 길거나 가로로 길면 정사각형으로 맞춤
    h, w = digit.shape
    if h > w:
        pad = (h - w) // 2
        digit = cv2.copyMakeBorder(digit, 0, 0, pad, h - w - pad, cv2.BORDER_CONSTANT, value=0)
    elif w > h:
        pad = (w - h) // 2
        digit = cv2.copyMakeBorder(digit, pad, w - h - pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # 5) MNIST 사이즈(28x28)로 맞추기
    resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

    # 6) MNIST는 숫자가 흰색, 배경이 검정 → 반전
    resized = 255 - resized

    pil_like = resized.astype(np.uint8)
    tensor = to_tensor(pil_like)   # (1,28,28)
    tensor = tensor.unsqueeze(0)   # (1,1,28,28)
    return tensor
