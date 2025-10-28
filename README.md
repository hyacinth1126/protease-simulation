# Protease kinetics and FRET dequenching simulation

Kgp protease에 의한 peptide substrate 분해 반응을 모델링하고 시뮬레이션하는 Python 애플리케이션입니다.

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 애플리케이션 실행
```bash
streamlit run app.py
```

## 📊 기능

### 키네틱 분석
- **미카엘리스-멘텐 모델**: Vmax, Km 파라미터 자동 추정
- **Lineweaver-Burk 플롯**: 선형화된 키네틱 분석
- **통계적 분석**: R² 값과 파라미터 불확실성 제공

### 🆕 FRET 형광 분석
- **FRET 키네틱**: 형광 기반 실시간 protease 활성 측정
- **형광 시뮬레이션**: Förster 공명 에너지 전달 모델링
- **소광 해제**: 기질 절단에 따른 형광 회복 분석
- **실험 데이터 비교**: 시뮬레이션 vs 실험 데이터 피팅

### 🆕 Hydrogel FRET Timeseries 분석
- **글로벌 피팅 모델**: X(t) = 1 - exp(-(k_eff*E + k0)*t)
- **다중 효소 농도**: 여러 효소 농도에서 동시 피팅
- **효율성 상수 (k_eff)**: kcat/KM과 유사한 속도 상수 추정
- **배경 반응 (k0)**: 효소 비의존적 반응 속도 추정
- **선형성 확인**: k_obs vs [E] 선형 관계 검증

### 시뮬레이션
- **시간 경과 시뮬레이션**: 기질과 생성물 농도 변화 추적
- **FRET 형광 시뮬레이션**: 시간에 따른 형광 강도 변화
- **다양한 초기 조건**: 사용자 정의 기질 농도 및 시뮬레이션 시간

### 데이터 처리
- **CSV 파일 업로드**: 일반 키네틱 및 FRET 실험 데이터 입력
- **데이터 검증**: 자동 데이터 유효성 검사
- **결과 다운로드**: 분석 결과 CSV 다운로드

## 📁 CSV 파일 형식

### 일반 키네틱 분석 데이터

| 컬럼명 | 설명 | 단위 |
|--------|------|------|
| `substrate_concentration` | 기질 농도 | μM |
| `reaction_rate` | 반응 속도 | μM/min |
| `time` (선택사항) | 측정 시간 | min |
| `experiment_id` (선택사항) | 실험 ID | - |

### 🆕 FRET 형광 분석 데이터

| 컬럼명 | 설명 | 단위 |
|--------|------|------|
| `time` | 측정 시간 | min |
| `fluorescence_intensity` | 형광 강도 | RFU |
| `substrate_concentration` (선택사항) | 기질 농도 | μM |
| `experiment_id` (선택사항) | 실험 ID | - |

### 🆕 Hydrogel FRET Timeseries 데이터

| 컬럼명 | 설명 | 단위 |
|--------|------|------|
| `time_s` | 측정 시간 | second |
| `E_nM` | 효소 농도 | nM |
| `fluor` | 형광 강도 | RFU (또는 임의 단위) |
| `group_id` (선택사항) | 반복 측정 그룹 | - |

### 예시 데이터

**일반 키네틱 데이터:**
```csv
substrate_concentration,reaction_rate,time,experiment_id
1,0.0323,0,exp_1
2,0.0612,1,exp_1
5,0.1389,2,exp_1
10,0.2424,3,exp_1
20,0.4000,4,exp_1
```

**FRET 형광 데이터:**
```csv
time,fluorescence_intensity,substrate_concentration,experiment_id
0.0,58.2,100.0,fret_exp_1
5.0,164.8,78.2,fret_exp_1
10.0,263.9,60.8,fret_exp_1
15.0,355.8,46.3,fret_exp_1
20.0,441.5,34.6,fret_exp_1
```

**Hydrogel FRET Timeseries 데이터:**
```csv
time_s,E_nM,fluor
0.0,10,52.3
60.0,10,145.8
120.0,10,238.5
0.0,25,50.1
60.0,25,189.2
120.0,25,327.3
```

## 🔬 Kgp Protease 정보

Kgp (Lysine-specific gingipain)는 *Porphyromonas gingivalis*에서 분비되는 시스테인 프로테아제로:

- **특이성**: 아르기닌 잔기 C-말단에서 절단
- **최적 pH**: 7.5-8.5
- **분자량**: 약 50 kDa
- **임상적 중요성**: 치주질환과 관련

## 🧮 수학적 모델

### 미카엘리스-멘텐 방정식
```
v = (Vmax × [S]) / (Km + [S])
```

여기서:
- `v`: 반응 속도
- `Vmax`: 최대 반응 속도
- `[S]`: 기질 농도
- `Km`: 미카엘리스 상수

### 시간 경과 모델
```
d[S]/dt = -v
d[P]/dt = +v
```

### Hydrogel Peptide Kinetic 모델
```
X(t) = 1 - exp(-(k_eff × [E] + k0) × t)
```

여기서:
- `X(t)`: t 시점에서 절단된 펩타이드 분율
- `k_eff`: 효율성 상수 (M⁻¹s⁻¹), kcat/KM에 비례
- `[E]`: 효소 농도 (M)
- `k0`: 배경 반응 속도 (s⁻¹)
- `t`: 시간 (s)

이 모델은 하이드로겔에 고정된 FRET quenching 펩타이드의 절단을 설명합니다.

## 📈 시각화

- **대화형 플롯**: Plotly 기반 인터랙티브 차트
- **실시간 분석**: 데이터 변경 시 즉시 업데이트
- **다중 뷰**: 미카엘리스-멘텐, Lineweaver-Burk, 시간 경과 플롯

## 🛠️ 기술 스택

- **Python 3.8+**
- **Streamlit**: 웹 애플리케이션 프레임워크
- **NumPy/SciPy**: 수치 계산 및 최적화
- **Pandas**: 데이터 처리
- **Plotly**: 대화형 시각화
- **Matplotlib/Seaborn**: 정적 시각화

## 📚 참고문헌

Shin, D. S., Liu, Y., Gao, Y., Kwa, T., Matharu, Z., & Revzin, A. (2013). Micropatterned surfaces functionalized with electroactive peptides for detecting protease release from cells. *Analytical Chemistry*, 85(1), 220–227. https://doi.org/10.1021/ac302547p

## 📞 문의사항

프로젝트 관련 문의사항이나 버그 리포트는 이슈 탭에서 등록해주세요.
