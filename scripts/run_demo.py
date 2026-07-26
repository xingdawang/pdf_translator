from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from pdf_translator.config import AppConfig
from pdf_translator.workflow import TranslationWorkflow
from sample_assets import mock_translate_xlsx


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "examples" / "sample_source.pdf"
    destination = project_root / "output" / "pdf"
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pdf-translator-demo-") as temporary:
        workflow = TranslationWorkflow(AppConfig.from_env(Path(temporary) / "data"))
        task = workflow.create_task(source)
        task, packages = workflow.export_packages(task.task_id)
        translated = Path(temporary) / "translated.xlsx"
        mock_translate_xlsx(packages[0], translated)
        task, _, report = workflow.import_packages(task.task_id, [translated])
        if not report.can_generate:
            raise RuntimeError(report.to_dict())
        task, outputs = workflow.generate(task.task_id)
        for path in (
            outputs.chinese_pdf,
            outputs.bilingual_pdf,
            outputs.quality_html,
        ):
            if path:
                shutil.copy2(path, destination / path.name)
        quality = json.loads(outputs.quality_json.read_text(encoding="utf-8"))
        for key, output_path in (
            ("chinese_pdf", outputs.chinese_pdf),
            ("bilingual_pdf", outputs.bilingual_pdf),
        ):
            if output_path and key in quality["outputs"]:
                quality["outputs"][key]["path"] = str(
                    (destination / output_path.name).resolve()
                )
        (destination / outputs.quality_json.name).write_text(
            json.dumps(quality, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(destination)


if __name__ == "__main__":
    main()
