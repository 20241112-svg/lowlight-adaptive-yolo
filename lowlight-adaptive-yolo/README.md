# 선택적 저조도 보정 기반 객체 탐지 실험

**YOLO11n + Zero-DCE++를 이용해, 영상 보정을 언제 적용하는 것이 유리한지 비교하는 학부 프로젝트입니다.**

실행은 Google Colab에서 진행합니다. 이 저장소에는 코드와 안내서만 있으며, 데이터·사전학습 가중치·실험 결과는 포함되어 있지 않습니다. 결과는 실제 실행 후 생성됩니다.

## 처음 시작하기

1. GitHub에 `lowlight-adaptive-yolo`라는 **Public** 저장소를 만듭니다.
2. 받은 ZIP을 압축 해제하고, 폴더 **안의 내용물**을 GitHub에 업로드합니다.
3. [Google Colab](https://colab.research.google.com/)에서 **파일 → 노트북 열기 → GitHub**를 선택합니다.
4. 자신의 저장소 주소를 입력하고 `notebooks/START_HERE.ipynb`를 엽니다.
5. **파일 → Drive에 사본 저장**을 누릅니다.
6. **런타임 → 런타임 유형 변경 → GPU**를 선택합니다.
7. 첫 설정 셀의 `GITHUB_USER`를 자신의 GitHub 아이디로 바꿉니다.
8. 처음에는 `MODE = "smoke"`로 0~3단계만 실행합니다.

버튼 위치, 정상 출력, 이어서 학습하는 방법은 **[처음부터 따라 하는 실행 가이드](docs/RUN_GUIDE_KO.md)**에 설명되어 있습니다.

공개 저장소에서 코드를 읽어 오는 방식이므로 GitHub 토큰이나 Git 설치가 필요하지 않습니다. Google Drive 연결은 결과를 자신의 계정에 저장하기 위한 과정입니다.

## 연구 질문

> 동일한 탐지 모델에서 필요한 입력만 보정하면 탐지 성능을 유지하면서 처리 시간을 줄일 수 있는가?

선택적 보정이 항상 좋은 결과를 낸다고 가정하지 않습니다. 검증 과정에서 보정 없음 또는 항상 보정이 선택될 수도 있으며, 그 결과를 그대로 기록합니다.

## 비교할 실험

| 이름 | 학습 입력 | 추론 입력 처리 | 목적 |
|---|---|---|---|
| A | COCO 사전학습 모델 사용 | 보정 없음 | 추가 학습 전 기준 |
| B-gamma / B-CLAHE | 같은 사전학습 모델 | 감마 / CLAHE | 전통적 보정과 비교 |
| C | 같은 사전학습 모델 | 항상 Zero-DCE++ | 학습 없이 보정만 적용 |
| D | ExDark 원본 | 보정 없음 | 원본 파인튜닝 기준 |
| E | ExDark 보정본 | 항상 Zero-DCE++ | 보정본에 맞춘 파인튜닝 |
| M-원본 | ExDark 원본·보정본 1:1 혼합 | 보정 없음 | 혼합 학습 모델의 기준 |
| M-항상 | 위와 같은 M | 항상 Zero-DCE++ | 같은 가중치의 비교군 |
| M-선택 | 위와 같은 M | 검증셋에서 고른 기준값 적용 | 제안하는 입력 처리 정책 |

학습하는 탐지 모델은 **D·E·M 세 개**입니다. M의 세 가지 추론 정책은 같은 가중치를 사용합니다. Zero-DCE++는 사전학습 가중치를 고정합니다.

M은 각 원본 이미지에 대해 원본 또는 보정본 **한 가지 버전만** 사용합니다. 고정된 seed로 학습·검증 각 분할의 절반을 보정본으로 선택합니다. 원본과 보정본을 모두 넣어 학습량을 두 배로 늘리지 않습니다.

## 데이터와 평가 규칙

- **ExDark 공식 분할:** 학습 3,000장 / 검증 1,800장 / 테스트 2,563장.
- **밝은 이미지:** COCO **val2017**에서 원본의 평균 밝기 90 이상이며 대상 클래스의 일반 객체가 있는 이미지를 선택합니다. 기준값 선택용 300장과 최종 테스트용 500장은 서로 겹치지 않습니다.
- **기준값 선택:** ExDark 검증 300장 + 밝은 COCO 검증 300장. 후보는 `0, 30, 60, 90, 120, 256`입니다.
- **선택 기준:** M-항상 대비 전체·저조도·밝은 그룹의 mAP50 감소가 각각 0.01 이하인 후보 중 측정한 평균 처리 시간이 가장 짧은 값을 선택합니다. 0.01은 **1%p**입니다.
- **최종 평가:** ExDark 테스트 전체와 저조도 500장 + 밝은 500장의 혼합 테스트를 보고합니다.
- **속도:** 같은 입력 300장, batch=1, 예열 후 3회 반복. 실제 밝기 판단·보정·탐지·후처리 시간을 포함합니다. 디스크 파일 읽기·이미지 디코딩·일회성 모델 로딩은 제외합니다.
- COCO의 군중 영역 정답은 `iscrowd=1`로 유지합니다. COCO 모델의 80개 클래스와 ExDark의 12개 클래스는 공통 이름으로 대응시킵니다.

밝은 COCO 이미지를 선별할 때는 원본 이미지 밝기를 사용합니다. 실제 보정 선택은 모든 정책이 공통으로 적용하는 긴 변 최대 640픽셀 축소 뒤, 여백을 붙이기 전 이미지의 밝기로 판단합니다. 두 정의를 보고서에 함께 기록합니다.

`Table`과 `dining table` 등 클래스 의미 범위의 차이, 두 데이터셋의 장면 차이, 한 GPU 환경에서의 속도 측정은 한계로 밝혀야 합니다. “밝은 이미지”는 실제 촬영 시각이 낮이라는 뜻이 아닙니다.

## 저장되는 결과

Google Drive의 `내 드라이브/lowlight_lab/experiments/실행이름/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `config.json`, `environment.json` | 실험 설정, 패키지·GPU·모델 버전 |
| `runs/D`, `runs/E`, `runs/M` | 학습 기록, `best.pt`, `last.pt` |
| `protocol.json` | 검증셋에서 확정한 기준값과 모델 식별 정보 |
| `results/threshold_validation.csv` | 기준값 후보별 검증 성능 |
| `results/test_metrics.csv` | 최종 평가 성능 |
| `results/latency.csv` | 실제 전체 처리 시간과 보정 적용 비율 |
| `results/figures/` | 발표용 집계 그래프 |
| `results_for_github.zip` | 코드에서 선별한 공개용 집계 결과 |

GitHub에 결과를 올릴 때는 `results_for_github.zip`을 풀고 그 안의 `results` 폴더를 업로드합니다. 이 내보내기에는 원본·보정 이미지, 정답 라벨, 모델 가중치, 개별 이미지 예측 파일이 들어가지 않습니다.

## 실행 확인 범위

배포된 코드는 작은 합성 자료를 이용한 데이터 변환·평가 로직 검사와 제한된 CPU 동작 검사를 위한 테스트를 제공합니다. **전체 ExDark·COCO 실험 및 Colab GPU에서의 최종 성능은 사용자가 실행해 확인해야 합니다.** 저장소에 예상 성능을 실제 결과처럼 채워 넣지 않았습니다.

실제로 수행한 점검과 확인하지 않은 범위는 [배포 전 작동 점검](docs/VERIFICATION.md)에 기록했습니다.

```bash
python -m unittest discover -s tests -v
```

Colab의 GPU 배정과 실행 가능 시간은 고정되어 있지 않습니다. 연결이 끊겼을 때는 같은 설정으로 환경·데이터·학습 입력을 복구하고 해당 학습 셀을 다시 실행합니다. 이어서 학습하는 상세 순서는 실행 가이드를 참고하세요.

## 공식 자료

- [GitHub — 브라우저에서 파일 업로드](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics 학습 및 이어서 학습](https://docs.ultralytics.com/modes/train/)
- [ExDark 공식 데이터셋](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
- [ExDark 라벨·공식 분할 설명](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Groundtruth/README.md)
- [Zero-DCE++ 공식 코드](https://github.com/Li-Chongyi/Zero-DCE_extension)
- [Zero-DCE++ 저자 프로젝트](https://li-chongyi.github.io/Proj_Zero-DCE%2B%2B.html)
- [COCO 공식 다운로드](https://cocodataset.org/#download)

원자료·모델의 이용 조건은 각 공식 배포처에 따릅니다. 이 저장소는 해당 원자료와 가중치를 재배포하지 않습니다.
