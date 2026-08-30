# SnnProject Git repository policy

## 목적

Git에는 다른 개발 환경에서 코드를 검토·수정·시험하는 데 필요한 최소 자산만 저장한다. 원본 radar/BIOPAC 생체 데이터, window 단위 값, cache, checkpoint, sealed pack, 대용량 복원 archive는 Git 밖의 제한된 복원 백업에서 관리한다.

## 포함 범위

- `src/snn_rr/`, `scripts/`, `tests/`, `configs/`
- `README.md`, `REPORT.md`, `AGENTS.md`, `RESTORE_GUIDE.md`
- `pyproject.toml`, 선별된 복원 script와 requirements
- `.gitignore`, `.gitattributes`, `.githooks/`, Git 안전 검사기
- `artifacts/`의 명시적으로 검토된 핵심 Markdown 문서만

## 제외 범위

- `HAI_EXPERIMENT/`, 두 raw ZIP, acquisition guideline PPTX
- `sync_tool_S02.m`: 폐기된 parser 가정과 개인 경로가 있는 legacy 도구
- RF/SVD/harmonic cache와 모든 `.npy`·`.npz`
- checkpoint와 model payload: `.pt`, `.pth`, `.ckpt`, `.onnx`
- per-window prediction, session CSV, review image, sync signal
- acquisition manifest, dataset audit, sealed pack, lifecycle/output tree, ledger
- `.venv`, Python/test cache, editor·agent local state
- Drive folder ID, API key, token, private key, credentials 또는 secret 파일

## 안전장치

1. `.gitignore`: private·binary·generated 경로 차단, `artifacts/`와 `restore/` allowlist 제한
2. `scripts/check_git_safety.py`: staged/tracked 파일의 경로, 확장자, 크기, symlink, 대표 secret pattern 검사
3. `.githooks/pre-commit`: commit 직전 안전 검사 자동 실행
4. 원격 저장소: 반드시 private으로 생성하고 collaborator 최소화

Git ignore는 이미 추적된 파일을 제거하지 않는다. 새 파일을 추가하기 전 다음 검사를 실행한다.

```bash
.venv/bin/python scripts/check_git_safety.py --staged
git diff --cached --stat
git diff --cached --name-only
```

원본 데이터부터 재학습하려면 Git clone만으로는 부족하다. 제한된 restore bundle을 별도로 복원하고 `AGENTS.md`와 `RESTORE_GUIDE.md`의 검증 순서를 따른다.
