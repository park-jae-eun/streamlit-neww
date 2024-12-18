from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# Flask 애플리케이션 생성
app = Flask(__name__)

# KoBART 모델 및 토크나이저 로드
model_name = 'gogamza/kobart-summarization'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 요약 함수
def summarize_article(text):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=130, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 기본 경로
@app.route('/')
def home():
    return render_template('index.html')

# 요약 요청 처리
@app.route('/summarize', methods=['POST'])
def summarize():
    # 사용자가 입력한 텍스트 가져오기
    article = request.form['article']
    # 요약 실행
    summary = summarize_article(article)
    # 결과를 웹페이지로 반환
    return render_template('index.html', summary=summary, article=article)

if __name__ == '__main__':
    app.run(debug=True)
