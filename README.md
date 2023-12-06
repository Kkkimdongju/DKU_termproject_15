# Helmet detection(YOLOv8)
## 1. 개발환경
 ### 1) 사용한 OS
 #### - MacOS, Window 10
 ### 2) 파이썬 버전
 #### - 3.10.12
 ### 3) 설치한 라이브러리
 #### - PyYAML, ultralytics
 ## 2. 실행방법
 ### 1) 리포지토리에 업로드 한 Yolo.ipynb파일을 구글 colab에서 열기
 ### 2) 커스텀 데이터를 colab으로 다운로드
  ```python
 !wget -O Helmet_data.zip https://app.roboflow.com/ds/1Rj6BECRju?key=exdDCuJd4R
 ```
```python
import zipfile

with zipfile.ZipFile('/content/Helmet_data.zip') as target_file:
    target_file.extractall('/content/Helmet_Data')
```
### 3) 헬멧 데이터에 맞는 YAML파일 생성
```python
!pip install PyYAML
```
```python
import yaml

data = { 'train' : '/content/Helmet_Data/train/images/',
        'val' : '/content/Helmet_Data/valid/images/',
         'test' : '/content/Helmet_Data/test/images',
         'names' : ['Helmet', 'NoHelmet'],
         'nc' : 2}

with open('/content/Helmet_Data/Helmet_Data.yaml', 'w') as f: 
    yaml.dump(data,f)

with open('/content/Helmet_Data/Helmet_Data.yaml', 'r') as f: 
    helmet_yaml = yaml.safe_load(f)
    display(helmet_yaml)
```
### 4) YOLOv8 설치
```python
!pip install ultralytics
```
```python
import ultralytics

ultralytics.checks()
```
### 5) 전처리 모델 로드
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
```
### 6) YOLOv8 커스텀 데이터 학습하기
```python
model.train(data = '/content/Helmet_Data/Helmet_Data.yaml', epochs=100, patience=25, batch=32, imgsz=320)
```
### 7) 학습된 YOLOv8 이용해서 테스트 이미지 예측
```python
results = model.predict(source='/content/Helmet_Data/test/images/', save=True)
```
