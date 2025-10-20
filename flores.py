# FLORES-200 예시: 영어/한국어/일본어/중국어(간체) 버전 출력
# 설치: pip install datasets

from datasets import load_dataset

# 1. FLORES-200 전체 데이터 불러오기
dataset = load_dataset("facebook/flores", "all")

# 2. dev set에서 한 문장 샘플 선택 (인덱스 0 사용)
sample = dataset["dev"][0]

# 3. 4개 언어 선택
langs = {
    "English": "eng_Latn",
    "Korean": "kor_Hang",
    "Japanese": "jpn_Jpan",
    "Chinese (Simplified)": "zho_Hans",
}

# 4. 결과 출력
print("🌍 Sample from FLORES-200 (multilingual parallel sentence)\n")
for name, code in langs.items():
    print(f"[{name}]")
    print(sample[code])
    print()
