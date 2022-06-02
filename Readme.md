# Petmily Flask 서버
[2022 캡스톤 프로젝트](https://github.com/bagoonichanger/Petmliy_android_app)
## 개요
Petmily 앱의 감정 분석, 종 분류, 개고양이 탐지와 같은 이미지 처리를 담당하는 서버이다. Flask와 PyTorch를 이용하여 구성했다.

`server.py` REST 방식으로 통신이 가능한 Flask 서버이다. 모델을 로드하고 메모리에 적재하여 요청된 이미지를 처리한다.

`findspecies.ipynb` 학습에 사용된 코드이다.

## 사용된 파이썬 라이브러리
`Pytorch, PIL, Numpy, MMClassification, MMDetection, Flask, Flask_restx, werkzeug`

## 사용한 모델
### [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
![swin](https://github.com/wolfdate25/Petmily_flask_server/blob/main/imgs/swin.png)
이미지를 여러 윈도우로 분할하고 연산 중 레이어에 윈도우를 하나씩 합치는 구조이다.  
다양한 크기의 엔티티를 처리할 수 있고, 계산량을 크게 늘리지 않는 ViT 기반 모델이다.

### [CoAtNet](https://github.com/chinhsuanwu/coatnet-pytorch)
![CoAtNet](https://github.com/wolfdate25/Petmily_flask_server/blob/main/imgs/coatnet.png?raw=true)
CNN과 Transformer의 장점을 합친 모델이다.  
적은 양의 데이터 셋과 작은 모델 크기에도 불구하고 뛰어난 성능을 보여준다. 

## 사용한 데이터셋
1. 객체 감지 - COCO Dataset(https://cocodataset.org/#home)
2. 견종 분류 - Stanford Dogs(http://vision.stanford.edu/aditya86/ImageNetDogs/)
3. 묘종 분류 - 자체 구축
4. 감정 분석 - 자체 구축

## 모델
[GDrive](https://drive.google.com/drive/folders/16yLD9X_-eXEqMnPpPdVCu97rNOeRsAwL?usp=sharing)
