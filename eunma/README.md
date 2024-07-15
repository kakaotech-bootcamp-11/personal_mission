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

## 문제 및 해결방안
### 문제 
용량 약 141GB

### 해결 방안
* AWS 
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
4. 
