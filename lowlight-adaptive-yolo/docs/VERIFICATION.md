# 배포 전 작동 점검

이 파일은 실행 코드의 점검 범위를 기록합니다. 실제 ExDark/COCO 탐지 성능 결과는 포함하지 않습니다.

## 실행해서 확인한 내용

- 데이터 변환·클래스 대응·평가·기준값 선택 단위 테스트 14개 통과.
- 사전학습 YOLO11n의 실제 CPU 추론.
- 공식 Zero-DCE++ 가중치 로딩과 실제 보정 추론.
- 6개 보정 정책과 3개 입력 크기 조합 18개 확인.
- adaptive 기준값 0은 무보정, 256은 항상 보정과 동일한 출력.
- 리사이즈된 예측 박스를 원본 좌표로 복원하는 코드 확인.
- 가상 학습 이미지 6장·검증 이미지 2장에서 D/E/M 각각 2 epoch 학습, best.pt·last.pt 저장.
- 작은 가상 데이터에서 검증 기준값 선택, 프로토콜 고정, 최종 평가, 온라인 처리 시간 측정, 그래프 4개, 요약, 결과 ZIP, 데모 생성.
- 완료된 학습 건너뛰기와 테스트 이후 기준값 재선택 차단.
- 공식 imageclasslist.txt의 7,363개 항목과 3000/1800/2563 분할 확인.
- START_HERE.ipynb의 40개 셀·19개 코드 셀 구문 및 비워진 실행 출력 확인.

가상 데이터에서 얻은 수치와 출력 이미지는 배포 ZIP에 넣지 않았습니다.

## 아직 확인하지 않은 범위

- 전체 ExDark/COCO 이미지 다운로드 및 모든 실물 이미지의 라벨.
- 전체 데이터셋을 사용한 학습 결과와 GPU 처리 속도.
- 사용자의 GitHub 저장소, Google Drive 인증, 실제 Colab UI에서의 로그인·연결.
- 학습 도중 강제 종료된 Colab 세션의 실제 복구. 코드에는 Ultralytics의 last.pt 기반 resume을 적용했습니다.

## 점검 환경

Python 3.12.14, torch 2.15.0+cpu, torchvision 0.29.2+cpu,
ultralytics 8.3.221, OpenCV 4.14.0, Pillow 12.3.0,
numpy 2.3.5, pandas 2.2.3, matplotlib 3.10.8,
pycocotools 2.0.11, gdown 5.2.2.

Colab GPU의 실제 torch/torchvision 버전은 다를 수 있습니다. 실행 때 environment.json에 환경을 기록합니다.

## 다시 점검하기

설치한 실행 환경에서 저장소 최상위에서 실행합니다.

```bash
python -m unittest discover -s tests -v
```

Zero-DCE++ 소스는 저자 저장소의 커밋
`09f202b690f82da939b8e6ec8535960ae97ad8bd`를 사용합니다.
모델과 가중치를 내려받을 때 SHA256을 provenance.json에 기록합니다.
