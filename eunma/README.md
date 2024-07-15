# 새로운 모델 llama-3 사용 제안

## 모델 소개 
https://huggingface.co/Bllossom/llama-3-Korean-Bllossom-70B
한국어 영어 이중 언어모델

## 모델 테스트(코렙 튜토리얼)
https://colab.research.google.com/drive/1fBOzUVZ6NRKk_ugeoTbAOokWKqSN47IG?usp=sharing

## 모의 테스트 캡쳐

![test](https://github.com/user-attachments/assets/d13deb3d-97e5-4a07-adb0-39dfb54491d6)


user : 오늘의 날씨 알려줘

봇 : 죄송하지만 저는 실시간 데이터를 접근할 수 없기 때문에 현재 날씨 정보를 제공할 수 없습니다. 대신 여러분이 살고 있는 지역의 날씨를 확인하고 싶다면, 인터넷 검색 엔진이나 모바일 애플리케이션을 통해 최신 정보를 얻으실 수 있습니다.

답변이 gpt 3.5에 전혀 밀리지 않은 언어생성능력을 보여줌

## 논의할 내용
우리가 추구하는 챗봇모델치고는 성능이 좋음. 성능이 좋은건 문제가 안되나, 그걸 위해서 141gb의 대형모델을 사용해야함.(3.5api사용과 차이점은 api를 불러오면 작업이 끝남). 다만 금액 걱정은 하지 않아도 됨
이 방법을 채택하지 않는다면, 
1. 필요한 데이터만 학습시켜서 정해진 질문 - 대답 의 형식의 챗봇을 만들거나(이 경우 웹에서의 챗봇성능이 초라해보일 수 있음,,?)
2. 살짝의 요금을 지불하더라도 3.5api가져와서 사용(튜닝시 추가적인 비용 발생)
3. 또 다시 적당한 모델을 탐색한다

## 문제 및 해결방안
로컬에서 작업하기 어려운 141gb의 용량

### AWS
1. EC2 인스턴스 생성
  - AWS Mangement Console에서 작업 시작
  - EC2 대시보드의 Lunch Instance
  - AMI선택 후 Deep learning AMI(Ubuntu) 버전
  - 인스턴스 유형을 GPU사용을 위한 p2/p3/g4계열 선택

2. EC2 인스턴스 접속
   - SSH로 인스턴스 접속
```bash
ssh -i /path/to/your-key.pem ubuntu@your-ec2-public-dns
```

3. 환경설정 및 라이브러리 설치
```bash
sudo apt update
sudo apt install -y python3-pip
pip3 install torch transformers accelerate
```
4. 모델 로드 및 실행
아래 파이썬 코드를 EC2 인스턴스에 넣고 실행
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'Bllossom/llama-3-Korean-Bllossom-70B'

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 모델을 평가 모드로 설정
model.eval()

# 초기 시스템 프롬프트 설정
PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''

# 대화 이력 저장을 위한 리스트 초기화
conversation_history = [
    {"role": "system", "content": f"{PROMPT}"}
]

def generate_response(conversation_history):
    # 대화 이력을 텍스트 형식으로 변환
    chat_history = ""
    for message in conversation_history:
        if message["role"] == "system":
            chat_history += f"System: {message['content']}\n"
        elif message["role"] == "user":
            chat_history += f"User: {message['content']}\n"
        elif message["role"] == "assistant":
            chat_history += f"Assistant: {message['content']}\n"

    input_ids = tokenizer.encode(chat_history, return_tensors="pt").to(model.device)

    # 모델을 사용하여 응답 생성
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.1
    )

    # 생성된 응답 디코딩
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

print("챗봇을 시작합니다. '종료'라고 입력하면 대화가 종료됩니다.")
user_input = ""

while user_input.lower() != "종료":
    # 사용자로부터 입력 받기
    user_input = input("사용자: ")
    if user_input.lower() == "종료":
        break
    
    # 대화 이력에 사용자 입력 추가
    conversation_history.append({"role": "user", "content": f"{user_input}"})
    
    # 사용자 입력에 대한 챗봇 응답 생성
    chatbot_response = generate_response(conversation_history)
    
    # 대화 이력에 챗봇 응답 추가
    conversation_history.append({"role": "assistant", "content": f"{chatbot_response}"})
    
    # 챗봇 응답 출력
    print(f"챗봇: {chatbot_response}")

print("대화를 종료합니다. 감사합니다!")
```

5. 파인튜닝할 데이터 준비
   - 파인튜닝할 데이터를 S3버킷에 업로드 후 EC2 인스턴스에서 불러옴


7. 파인튜닝
   - 트랜스포머 라이브러리의 Trainer또는 Accelerate를 사용하여 파인튜닝 스크립트를 작성(학습률 배치크기 등의 하이퍼파라미터 설정)
  
### colab
1. 새로운 작업환경에서 런타입 유형을 GPU로 설정
2. 허깅페이스 라이브러리 설치
```python
!pip install transformers accelerate
```
3. 모델 로드 및 실행 예제코드
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'Bllossom/llama-3-Korean-Bllossom-70B'

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 모델을 평가 모드로 설정
model.eval()

# 초기 시스템 프롬프트 설정
PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''

# 대화 이력 저장을 위한 리스트 초기화
conversation_history = [
    {"role": "system", "content": f"{PROMPT}"}
]

def generate_response(conversation_history):
    # 대화 이력을 텍스트 형식으로 변환
    chat_history = ""
    for message in conversation_history:
        if message["role"] == "system":
            chat_history += f"System: {message['content']}\n"
        elif message["role"] == "user":
            chat_history += f"User: {message['content']}\n"
        elif message["role"] == "assistant":
            chat_history += f"Assistant: {message['content']}\n"

    input_ids = tokenizer.encode(chat_history, return_tensors="pt").to(model.device)

    # 모델을 사용하여 응답 생성
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.1
    )

    # 생성된 응답 디코딩
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

print("챗봇을 시작합니다. '종료'라고 입력하면 대화가 종료됩니다.")
user_input = ""

while user_input.lower() != "종료":
    # 사용자로부터 입력 받기
    user_input = input("사용자: ")
    if user_input.lower() == "종료":
        break
    
    # 대화 이력에 사용자 입력 추가
    conversation_history.append({"role": "user", "content": f"{user_input}"})
    
    # 사용자 입력에 대한 챗봇 응답 생성
    chatbot_response = generate_response(conversation_history)
    
    # 대화 이력에 챗봇 응답 추가
    conversation_history.append({"role": "assistant", "content": f"{chatbot_response}"})
    
    # 챗봇 응답 출력
    print(f"챗봇: {chatbot_response}")

print("대화를 종료합니다. 감사합니다!")
```
4. 파인튜닝에 필요한 라이브러리 예시
```python
!pip install transformers datasets accelerate
```
5. 파인튜닝에 쓸 데이터 구글드라이브 업로드 후 불러오기
```python
from google.colab import drive
drive.mount('/content/drive')
```
6. 파인튜닝 스크립트 및 하이퍼 파라미터 설정
```python
from transformers import Trainer, TrainingArguments

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir='./results',          # 학습 결과(모델, 로그 등)를 저장할 디렉토리
    num_train_epochs=3,              # 학습을 반복할 에포크 수 (전체 데이터셋을 몇 번 반복할 것인지)
    per_device_train_batch_size=4,   # 학습 시 사용되는 배치 크기 (GPU 하나당 4개의 샘플)
    per_device_eval_batch_size=4,    # 평가 시 사용되는 배치 크기 (GPU 하나당 4개의 샘플)
    warmup_steps=500,                # 학습 초기에 학습률을 천천히 증가시키는 단계 수 (워밍업 스텝)
    weight_decay=0.01,               # 가중치 감쇠 (L2 정규화) 비율, 과적합 방지를 위해 사용
    logging_dir='./logs',            # 학습 로그를 저장할 디렉토리
)

# Trainer 객체 생성
trainer = Trainer(
    model=model,                         # 학습할 모델
    args=training_args,                  # 학습 인자
    train_dataset=train_dataset,         # 학습 데이터셋
    eval_dataset=eval_dataset            # 평가 데이터셋
)

# 모델 학습 시작
trainer.train()
```
7. 파인튜닝 실행 및 모델저장
```python
trainer.save_model("/content/drive/MyDrive/path_to_save_model")
```




