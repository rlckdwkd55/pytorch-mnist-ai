const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit-btn");
const resultDiv = document.getElementById("result");

// 선택한 이미지를 미리보기로 보여주기
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.style.display = "block";
});

// 서버에 이미지 전송 → 숫자 예측하기
submitBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) {
        alert("이미지를 먼저 선택해주세요!");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    resultDiv.textContent = "인식 중...";

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });

        const data = await res.json();
        resultDiv.textContent = `예측 숫자: ${data.digit}, 확률: ${(data.confidence * 100).toFixed(2)}%`;
    } catch (err) {
        console.error(err);
        resultDiv.textContent = "서버 오류 발생";
    }
});