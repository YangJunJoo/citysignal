# 검증된 파이프라인 운영

`scripts/run_daily_pipeline.sh`는 `run_pipeline.py`의 다운로드 → 원본 품질 검사 → 분할 → 특징 추출을 호출한다. CLI, 백로그, 대시보드와 직접 추출은 같은 완료 판정을 사용한다. 파일 크기 제한과 채터링 차단은 없다.

## 설정 사전 검증

```bash
python scripts/audit_signal_configs.py --help
```

CSV를 읽기 전에 공백·헤더·열·Phase·Channel·barrier 관계를 검증한다. 표준 열 이름과 숫자 표현은 메모리에서 정규화한다. 근거 없는 Phase 또는 detector 매핑을 생성하지 않는다. 유효한 일대다 채널 관계는 경고와 검토 표시로 남긴다.

`use`, `repair_pending`, `exclude`를 구분한다. 명시적 제외에는 사유·출처·적용 기간이 필요하다. `repair_pending`과 잘못된 설정은 추출 전에 차단된다. 설정의 SHA-256과 검증 규칙 버전이 재사용 판정에 포함된다.

## 결과와 완료 판정

기본 실행은 phase 1~8과 두 ring `[1,2,3,4]`, `[5,6,7,8]`을 가정한다.
기존 barrier 번호는 `{1,2,5,6}: 2`, `{3,4,7,8}: 1`이다.
controller 계획·검토자·적용 기간은 기본 실행의 필수 입력이 아니다.
차로 설정의 적용 기간 미확정도 이 가정 모드 자체의 차단 사유로 사용하지 않는다.
가정은 `mapping_basis=assumed_eight_phase`, `barrier_mapping_assumed=true`,
`barrier_mapping_verified=false`로 각 결과와 진단에 기록한다.

관측된 phase를 보존하고 1~8 밖의 phase는 barrier 0으로 두므로 기본 cycle 경계에 영향을 주지 않는다.
차로 inventory에 phase가 없더라도 그 관측 phase의 신호 상태를 삭제하지 않는다.
차로·검출기 기반 특징의 실제 대응은 설정을 사용한다. 채널 번호나 특수 phase 표기는 추정하지 않는다.
보행 이벤트는 기존 eventParam phase 연결을 사용한다.
기본 가정에서 상충하는 녹색 구간은 기존 quarantine 정책으로 해당 cycle을 격리한다.
명시적인 controller_mapping이 전달되면 그 기록의 식별자·적용 기간·내용 해시를 계속 검증한다.

`feature_extractor/feature_registry.py`가 프로파일 4종과 특징 15종의 이름·경로·의존성을 정의한다. 기본 필수 결과는 18종이며 `conflict_propensity`는 명시적으로 요청하는 선택 특징이다. `conflict_headway`는 `conflict`의 별칭이다. PET도 필수 목록에 포함된다. 특징을 선택하면 그 의존성을 함께 실행한다.

| 상태 | 의미 |
|---|---|
| `success` | 유효한 비어 있지 않은 결과를 검증하여 저장함. 정상적인 0 값도 성공이다. |
| `failed` | 계산 예외 또는 출력 검증·저장 오류. 원인을 보존한다. |
| `no_data` | 입력이나 산출물이 비었거나 필요한 특징 값이 없음. |
| `not_applicable` | 설정에 해당 검지기가 없다는 등 명시적 적용 불가 근거가 있음. |
| `blocked` | 설정·원본 품질·의존성 문제로 실행할 수 없음. |
| `skipped` | 근거와 기간이 있는 명시적 제외. |

필수 결과가 모두 `success` 또는 근거가 있는 `not_applicable`이어야 해당 추출 작업이 `complete`다. 실패한 의존성이 있으면 후속 특징을 차단하고 독립적인 특징은 계속 처리한다. `no_data`나 제외는 전체 성공에 포함되지 않는다. 보행 기능의 부재는 lane CSV만으로 추정하지 않는다.

출력은 작업마다 격리된 임시 경로에서 계산한다. 스키마·신호 ID·행 수·주기 구간·입력의 관측 시간 범위·직렬화 왕복을 검증한 뒤 파일별 `os.replace`로 게시한다. 실패한 새 시도가 과거 파일의 존재로 성공 처리되지 않는다.

기존 `processing_log.db`는 보존한다. 옆의 **`pipeline_runs.sqlite3`**에 실행·시도·특징 결과를 추가 기록한다. `latest_attempts`는 최신 시도만 보여준다. 과거 파일을 동기화할 때는 `imported_unverified`로 기록하며 성공을 만들어내지 않는다. 재사용에는 입력·설정·정책·원본 품질 버전과 출력 서명이 일치해야 한다. 동일 버전의 `no_data`는 반복 계산을 생략해도 미완료 상태를 유지한다.

8-phase 기본 가정을 도입한 실행 규칙은 `checked-extraction-6-eight-phase-default`다.
이전 규칙의 `barrier_mapping_unverified` 결과는 그대로 재사용하지 않고 새 시도로 평가한다.

종료 코드는 `0=complete/no_work`, `2=partial`, `1=시스템 실행 실패`, `75=다른 실행이 잠금을 보유`다. 대시보드는 부분 완료를 성공으로 집계하거나 자동 재시도하지 않는다.

## 원본 품질과 복구

원본별 최소·최대 시각, 시간별 전체 및 신호별 행 수, 잘못된 시각, 관측 범위 내부와 인접 날짜 사이 공백을 검사한다. 파일 서명과 규칙 버전으로 캐시하며 인접 원본이 바뀌어도 경계 판정을 무효화한다. HTTP 404의 게시 지연은 `source_pending`, 내부 누락이나 불완전 자료는 `source_incomplete`로 구분한다.

현재 공개 소스 명세에는 timestamp 시간대와 파일 날짜 경계가 명시되어 있지 않다. 따라서 20:00–20:00 같은 기대 구간을 사실로 고정하지 않으며 `contract_status=unconfirmed`, `expected_coverage_verified=false`를 보존한다. 관측 공백 검증을 통과하는 것과 전체 기대 시간 범위가 입증되는 것은 별도 정보다. 근거가 있는 계약을 전달하면 DST의 23/25시간 날짜도 검사한다.

다운로드는 HTTP 응답 길이·Content-Range·ETag/Last-Modified를 확인한다. 잘못된 이어받기 응답은 안전하게 재시작하며 검증이 끝나기 전 최종 파일을 교체하지 않는다. 복구 조사는 별도 staging에서 수행한다. 2026-08-26과 2026-09-01의 원격 전체 CSV도 기존 원본과 같은 공백을 가진 것으로 확인되어 운영 원본을 유지했다.

## 영향 대상만 재처리

```bash
python scripts/build_reprocess_manifest.py \
  --audit-dir scratch/pipeline_audit_20260909 \
  --event-dir /data2/ATSPM/raw_data/per_signal \
  --config-dir /data2/ATSPM/config_data/signal_config_2025/lanes \
  --quality-report scratch/raw_quality_20260911/final_quality.json \
  --output scratch/reprocess_20260911/jobs.json

python run_pipeline.py --steps extract \
  --jobs-file scratch/reprocess_20260911/jobs.json \
  --quality-report scratch/raw_quality_20260911/final_quality.json \
  --max-workers 12 \
  --summary-file scratch/reprocess_20260911/execution_summary.json
```

목록 생성은 추출하지 않는다. 목록에는 정확한 `(signal_id, date_str)` 쌍, 요청 특징과 의존성, 입력 서명·SHA-256, 설정 SHA-256, 선정 근거와 차단 사유가 들어간다. 신호 목록과 날짜 목록의 곱으로 작업을 확장하지 않는다. 입력이 변경되면 오래된 목록을 차단하고 다시 검토하게 한다. `--dry-run --summary-file PATH`로 추출 CLI의 검증 결과도 저장할 수 있다.

확인된 설정 오류와 복구되지 않은 원본은 작업 목록에 남겨 차단 상태를 기록한다. 수정 근거를 확보한 경우 해당 신호·날짜의 새 목록을 만들면 된다. 설정이 계속 없는 과거 작업 2,015건을 일괄 재실행하지 않는다.

## 검증 범위

회귀 테스트는 설정, 결과 상태·의존성, 원자적 저장 실패, 정확한 작업 목록, HTTP 이어받기, 공백·DST·유효하지 않은 날짜, 대시보드·동기화를 검증한다. 실제 표본은 정상 신호의 필수 18종 성공과 빈 주기 결과의 `no_data → blocked` 전파를 확인했다.

기존 `tests/test_features.py`의 골든 자료는 현재 보관된 1050/2025-09-13 원본과 시간 범위가 일치하지 않아 일부 수치 비교가 실패한다. 이 비교를 맞추기 위해 시간대를 임의 변경하거나 골든 값을 수정하지 않았다. 테스트 fixture는 임시 디렉토리를 사용해 운영 산출물에 쓰지 않는다.
