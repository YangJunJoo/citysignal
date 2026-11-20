# CitySignal

신호 이벤트와 교차로 설정을 읽어 phase·cycle 프로파일과 교통 지표를 계산합니다.
추출 구현은 버전 구분 없이 `feature_extractor/`에서 관리합니다.
설치 패키지와 공개 import 이름은 `citysignal`입니다.
`src/citysignal/`은 공통 구현을 연결하는 공개 API이며, 추출 코드를 복제하지 않습니다.

```text
src/citysignal/          # 공개 Python API와 citysignal 명령
feature_extractor/
├── extractors/          # SPaT, volume, headway, gap, occupancy 등
├── profiles.py          # 차량·보행자 phase/cycle 생성
├── data_loader.py       # 신호·날짜별 입력 캐시
├── pipeline.py          # 공통 특징 계산과 의존성 처리
├── storage.py           # Parquet 저장·검증, 중첩 값과 메타데이터 복원
├── checked_execution.py # 검증, 임시 저장, 원자적 결과 발행
├── checked_runner.py    # 작업 발견, 병렬 실행, 실행 이력
└── feature_registry.py  # 특징 이름, 의존성, 저장 경로
data_downloader/         # 다운로드, 분할, 원본 품질·설정 수집
tests/                  # 회귀·통합 검증과 기준 데이터
docs/                   # 처리 규칙, 운영·인수인계 문서
scripts/                # 운영 및 분석 명령
run_pipeline.py         # 다운로드 → 품질 검사 → 분할 → 추출
```

## 설치와 실행

프로젝트 루트에서 현재 Python 환경에 설치합니다.

```bash
python -m pip install -e '.[test]' --config-settings editable_mode=strict
python -m citysignal --help
citysignal-pipeline --help
python -m pytest tests -q
```

설치 후 `citysignal`, `citysignal-extract`, `citysignal-pipeline` 명령을 사용할 수 있습니다.
기존 `citysignal/` 가상환경 폴더와 import 이름이 겹치므로 개발 설치에는 위의
strict editable 방식을 사용하고, 설치 중 생성된 `build/` 디렉터리를 유지합니다.
일반 설치는 `python -m pip install .`로 할 수 있습니다.
기존 `atspm-extract`, `atspm-pipeline` 명령도 유지합니다.
개발 환경에서는 `pyproject.toml`의 의존성으로 설치합니다.
루트 `requirements.txt`는 이전 운영 환경의 패키지 목록입니다.
현재 pandas 2.x를 지원합니다. pandas 3.x는 문자열·결측 dtype와 계산 동작이 달라
기존 결과의 동일성이 검증되기 전까지 설치 의존성에서 제외합니다.
NOEMI 설정 보고서를 브라우저로 수집할 때만 `.[reports]` 추가 의존성이 필요합니다.
추출기 import나 일반 데이터 처리에는 브라우저 패키지를 사용하지 않습니다.

운영 실행은 `checked_runner`와 `checked_execution`을 통과합니다.
입력 품질, 설정, 프로젝트의 8-phase 가정과 결과 검증을 적용한 뒤 결과를 발행합니다.
v1/v2 엔진 선택 옵션은 제거되었습니다.

## Phase와 cycle 계산 기준

기본적으로 phase 1~8, ring `[1,2,3,4]`와 `[5,6,7,8]`을 가정합니다.
기존 barrier 번호는 그대로 유지합니다: phase `1·2·5·6 → barrier 2`,
phase `3·4·7·8 → barrier 1`. 별도 ring/barrier 계획이나 적용 기간 자료를 요구하지 않습니다.
이 가정 자체나 차로 설정의 적용 기간 미확정을 이유로 계산을 차단하지 않습니다.

실제 관측된 phase·상태로 cycle을 구성하며 관측되지 않은 8개 phase를 만들어 넣지 않습니다.
1~8 밖의 관측 phase는 보존하되 barrier 0으로 두어 기본 cycle 경계를 결정하지 않게 합니다.
차로·채널 연결은 실제 설정에서 읽으며 숫자 phase 가정으로 detector 매핑을 생성하지 않습니다.
보행 이벤트는 기존처럼 eventParam의 phase 번호로 계산합니다.

결과 Parquet의 attrs와 mapping_audit에는 `mapping_basis=assumed_eight_phase`,
`barrier_mapping_assumed=true`, `barrier_mapping_verified=false`를 기록합니다.
명시적인 controller_mapping을 전달한 경우에만 그 계획과 적용 기간을 별도로 검증합니다.
관측 상태 누락·상충 cycle 격리·원본 공백·설정 오류 검사는 계속 적용합니다.

## 데이터와 검증

운영 입력·결과·실행 이력은 기본적으로 `/data2/ATSPM`에 저장되며 명령 옵션으로 변경할 수 있습니다.
`citysignal/Lib/site-packages`와 로컬 wheel을 소스나 의존성으로 사용하지 않습니다.
기존 `citysignal/` 가상환경은 자동 삭제하지 않으며, Git 추적 대상에서 제외합니다.

추출 결과와 내부 임시 데이터, 테스트 기준 데이터의 저장 형식은 **Parquet**입니다.
PyArrow의 Parquet 2.6과 ZSTD 압축을 사용하며, 신호·날짜별 기존 폴더 구조를 유지합니다.
저장 전에 열별 스키마를 정하고, 임시 파일을 다시 읽어 검증한 뒤 원자적으로 발행합니다.

프로파일과 특징 결과는 공통 API로 읽고 씁니다.

```python
from citysignal.storage import read_frame, write_frame

frame = read_frame("/path/to/2025-09-13.parquet")
write_frame(frame, "/path/to/copy.parquet")
```

파일 footer의 `atspm.dataframe` 계약에 원래 dtype·인덱스·중첩 컨테이너·결측 구분과
`DataFrame.attrs`를 저장합니다. 시간 구간은 Arrow의 `list<struct<begin, end>>`로 저장하고,
읽을 때 기존 계산 코드가 사용하는 `list[tuple[Timestamp, Timestamp]]`로 복원합니다.
나노초 정밀도와 로컬 시각을 유지하며 시간대를 임의로 변환하지 않습니다.
`read_frame`은 계약이 누락되거나 지원하지 않는 버전이면 실패 처리합니다.
외부 원본 이벤트 Parquet는 기존 원본 데이터 로더에서 읽습니다.

과거 실험의 여러 DataFrame 결과 묶음과 캐시 스냅샷은
`citysignal.archive_storage.read_artifact`로 읽는 Parquet 아카이브입니다.
DataFrame은 실제 Arrow list/struct 열에 저장하며, 제거된 클래스의 캐시는
클래스 코드 실행 없이 클래스 이름과 속성 데이터인 `ArchivedObjectState`로 복원합니다.

`pd.read_parquet`만으로는 계산에 필요한 `attrs`와 Python 컨테이너가 복원되지 않으므로,
추출 결과를 계산에 재사용할 때는 반드시 `read_frame`을 사용합니다.

기존 PKL의 일회성 전환은 `scripts/migrate_pickles_to_parquet.py`의 `plan → convert → delete`
단계로 수행합니다. 모든 계획 파일의 변환과 정확성 검증이 끝나야 삭제 단계가 허용되며,
삭제 직전 원본과 Parquet의 SHA-256을 다시 확인합니다. 실패·충돌·입력 변경 시 원본을 보존합니다.
변환 목록과 journal은 `scratch/parquet_migration_20261119/`에 보존합니다.
과거 실행 이력은 변경하지 않고 변환 provenance와 journal을 통해 새 경로를 확인하므로,
해당 journal은 검증 증거와 과거 이력의 재사용을 위해 유지해야 합니다.

작은 합성 입력을 사용하는 테스트는 운영 데이터 없이 실행됩니다.
`tests/golden_data` 회귀 검증 중 일부는 기존 원본·설정·프로파일도 필요하며,
해당 입력이 없으면 skip됩니다. 전체 검증 결과에서 skip 개수도 확인해야 합니다.

저장된 기준 결과와 새 결과는 읽기 전용 도구로 비교할 수 있습니다.

```bash
python scripts/compare_reference_outputs.py \
  --reference-dir /path/to/reference \
  --candidate-dir /path/to/candidate \
  --signal 1050 --date 2025-09-13 \
  --features spat volume
```

이 명령은 추출을 실행하거나 결과 파일을 덮어쓰지 않습니다.
기존 비교 프로그램의 원문은 `docs/archive/extraction_comparisons/`에 보존했습니다.
기존 파일명의 진입점은 위 읽기 전용 비교 도구로 연결됩니다.

새 실행 이력의 엔진 식별자는 `vectorized`입니다. 이전 이력은 수정하지 않으며,
이전 엔진 식별자로 저장된 결과를 새 실행에서 자동 재사용하지 않습니다.

처리·운영 절차는 [인수인계 문서](docs/handover/README.md)와
[처리 규칙](docs/checked_pipeline.md)을 참고하세요.

## GitHub에서 버전 관리

저장소 대상은 `YangJunJoo/citysignal`입니다.
코드·문서와 `tests/golden_data/`의 Parquet 기준 데이터를 추적하고,
운영 데이터·결과·가상환경·로그·인증정보는 `.gitignore`로 제외합니다.

push와 pull request마다 GitHub Actions가 Python 3.10·3.12·3.14에서 테스트하고,
wheel·소스 배포본을 빌드한 뒤 별도 환경에서 wheel import와 명령 실행을 확인합니다.
운영 입력이 필요한 회귀 테스트는 GitHub에서 skip될 수 있으며,
실제 출력의 동일성 검증은 해당 입력이 있는 운영 검증 환경에서도 수행해야 합니다.
빌드 파일과 테스트 결과는 Actions 실행의 artifact에서 확인할 수 있습니다.
이 workflow는 운영 실행, GitHub Release 생성, PyPI 게시를 자동으로 수행하지 않습니다.

버전은 `src/citysignal/__init__.py`의 `__version__` 한 곳에서 관리합니다.
검증된 커밋에 `v0.1.0` 같은 태그를 붙이고, 이후 변경은 별도 커밋으로 남깁니다.

## 우선 날짜 재추출과 모니터링

`scripts/run_extraction_campaign.py`는 지정된 NPTOI 우선 날짜를 처리한 뒤,
나머지 보관 원본을 최신 날짜부터 역순으로 처리합니다.
우선 현지 날짜 72일과 날짜 경계를 보완하는 원본 4일을 구분해 기록합니다.
기존 `processed_data`는 출력 대상으로 허용하지 않으며, 새 캠페인은 비어 있는 별도 출력 경로를 사용합니다.
동일 캠페인 디렉터리로 다시 실행하면 이미 시도한 날짜는 건너뜁니다.
partial·failed는 미완료로 남기고 다음 날짜로 진행하며 성공으로 집계하지 않습니다.
캠페인 전체가 공통 `.pipeline.lock`을 보유하므로 기존 cron 작업과 겹치지 않습니다.

```bash
python scripts/run_extraction_campaign.py \
  --campaign-dir /data2/ATSPM/operations/runs/nptoi_priority_20261120 \
  --processed-dir /data2/ATSPM/processed_data_reextract_20261120 \
  --workers 16

python dashboard/campaign_server.py \
  --campaign-dir /data2/ATSPM/operations/runs/nptoi_priority_20261120 \
  --host 127.0.0.1 --port 8050
```

읽기 전용 대시보드는 `http://localhost:8050/`에서 5초마다 날짜 진행률, 신호 작업,
특징 성공·차단 수와 현재 로그를 갱신합니다. 원격 서버에서는 SSH 또는 IDE 포트 전달을 사용합니다.
캠페인 상태·날짜별 로그는 캠페인 디렉터리에, 결과와 별도 실행 이력 DB는 새 출력 경로에 저장됩니다.
