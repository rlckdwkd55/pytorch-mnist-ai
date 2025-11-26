import torch
from flask import Flask, request, jsonify, render_template
from .model import load_trained_model  # 모델 불러오기 함수
from .preprocess import preprocess_for_mnist  # 이미지 전처리 함수

def create_app():
    # Flask 웹 서버 생성
    # static 폴더와 templates 폴더 경로 지정
    app = Flask(__name__, static_folder="../static", template_folder="../templates")

    # CPU로 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Flask using device:", device)

    # 훈련된 모델 불러오기
    model = load_trained_model("models/mnist_cnn.pth", device=device)

    # 웹 브라우저에서 접속하는 기본 화면
    @app.route("/")
    def index():
        return render_template("index.html")

    # 숫자 인식 API
    @app.route("/api/predict", methods=["POST"])
    def predict():
        # POST form 안에 file 이 없으면 에러
        if "file" not in request.files:
            return jsonify({"error": "no_file"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "empty_filename"}), 400

        # 이미지 파일 읽기 (바이트)
        image_bytes = file.read()

        # 이미지 전처리 → tensor 변환
        tensor = preprocess_for_mnist(image_bytes).to(device)

        # 모델 실행(추론)
        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)  # 각 숫자 확률 계산
            pred = int(prob.argmax(dim=1).item())  # 가장 높은 확률 인덱스 = 예측 숫자
            conf = float(prob[0, pred].item())     # 해당 확률

        # JSON 반환
        return jsonify({"digit": pred, "confidence": conf})

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
