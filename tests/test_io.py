import pytest

from subtleshift.io import DataError, read_jsonl, validate_chat_rows, validate_prompt_rows


def test_prompt_validation_rejects_duplicate_ids():
    row = {"id": "same", "prompt": "p", "group": "g", "split": "s"}
    with pytest.raises(DataError, match="Duplicate"):
        validate_prompt_rows([row, row])


def test_chat_validation_accepts_fixture():
    validate_chat_rows(
        [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}]
    )


def test_read_jsonl_reports_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(DataError, match=":2"):
        read_jsonl(path)

