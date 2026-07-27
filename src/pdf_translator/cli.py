from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .config import AppConfig, DEFAULT_DPI, DPI_CHOICES
from .exceptions import PDFTranslatorError
from .models import TaskSettings
from .utils import file_size_label
from .workflow import TranslationWorkflow


def _config(args: argparse.Namespace) -> AppConfig:
    return AppConfig.from_env(getattr(args, "data_dir", None))


def _progress(current: int, total: int) -> None:
    print(f"\r处理中 {current}/{total}", end="", flush=True)
    if current == total:
        print()


def _settings(args: argparse.Namespace) -> TaskSettings:
    terms = [
        term.strip()
        for term in (getattr(args, "protect_term", None) or [])
        if term.strip()
    ]
    return TaskSettings(
        translate_headers=getattr(args, "translate_headers", False),
        translate_footers=getattr(args, "translate_footers", False),
        ignore_page_numbers=not getattr(args, "translate_page_numbers", False),
        page_start=getattr(args, "page_start", 1),
        page_end=getattr(args, "page_end", None),
        protected_terms=terms,
        ocr_mode=getattr(args, "ocr", "off"),
        ocr_dpi=getattr(args, "ocr_dpi", DEFAULT_DPI),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-translator",
        description="本地 PDF 英译中翻译包往返工具",
    )
    parser.add_argument("--data-dir", help="任务数据目录")
    subcommands = parser.add_subparsers(dest="command", required=True)

    def add_import_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--copy-source", action="store_true", help="复制原 PDF 到任务目录")
        command.add_argument("--translate-headers", action="store_true")
        command.add_argument("--translate-footers", action="store_true")
        command.add_argument("--translate-page-numbers", action="store_true")
        command.add_argument(
            "--page-start",
            type=int,
            default=1,
            help="开始处理的原 PDF 页码",
        )
        command.add_argument(
            "--page-end",
            type=int,
            help="结束处理的原 PDF 页码；留空表示最后一页",
        )
        command.add_argument(
            "--ocr",
            choices=("off", "vision"),
            default="off",
            help="扫描件文字识别：vision 仅支持 macOS",
        )
        command.add_argument(
            "--ocr-dpi",
            type=int,
            default=DEFAULT_DPI,
            choices=DPI_CHOICES,
            help="OCR 渲染清晰度",
        )
        command.add_argument(
            "--protect-term",
            action="append",
            help="额外保护的术语，可重复传入",
        )

    analyze = subcommands.add_parser("analyze", help="分析 PDF 并创建任务")
    analyze.add_argument("pdf")
    add_import_options(analyze)

    export = subcommands.add_parser("export", help="分析 PDF 并导出 Google XLSX 翻译包")
    export.add_argument("pdf")
    export.add_argument("--output", help="翻译包输出目录")
    add_import_options(export)

    package = subcommands.add_parser("package", help="为已有任务重新生成翻译包")
    package.add_argument("task_id")
    package.add_argument("--output")

    import_command = subcommands.add_parser("import", help="导入翻译后的 XLSX")
    import_command.add_argument("task_id")
    import_command.add_argument("xlsx", nargs="+")

    validate = subcommands.add_parser("validate", help="检查当前任务")
    validate.add_argument("task_id")
    validate.add_argument("--json", action="store_true")

    edit = subcommands.add_parser("edit", help="人工修改一个段落的译文")
    edit.add_argument("task_id")
    edit.add_argument("segment_id")
    edit.add_argument("translation")

    generate = subcommands.add_parser("generate", help="生成原版面中文 PDF")
    generate.add_argument("task_id")
    generate.add_argument(
        "--output-mode",
        choices=("layout", "source-layout"),
        default="layout",
        help="layout 为原版面中文版；source-layout 为原文页与对应中文版交错",
    )
    generate.add_argument(
        "--layout-dpi",
        type=int,
        default=DEFAULT_DPI,
        choices=DPI_CHOICES,
    )
    generate.add_argument("--font", help="中文字体文件路径")

    subcommands.add_parser("list", help="列出任务")
    ui = subcommands.add_parser("ui", help="启动本地 Flask 网页")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", default=5050, type=int)
    ui.add_argument("--no-browser", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    config = _config(args)
    workflow = TranslationWorkflow(config)

    if args.command in {"analyze", "export"}:
        task = workflow.create_task(
            args.pdf,
            settings=_settings(args),
            copy_source=args.copy_source,
            progress=_progress,
        )
        print(f"任务：{task.task_id}")
        print(
            f"页数：{task.page_count}，段落：{len(task.translatable_segments)}，"
            f"图片：{task.image_count}，大小：{file_size_label(task.source_size_bytes)}"
        )
        for warning in task.warnings:
            print(f"警告：{warning}")
        if args.command == "export":
            _, paths = workflow.export_packages(task.task_id, args.output)
            for path in paths:
                print(f"翻译包：{path}")
        return 0

    if args.command == "package":
        _, paths = workflow.export_packages(args.task_id, args.output)
        for path in paths:
            print(path)
        return 0

    if args.command == "import":
        _, results, report = workflow.import_packages(args.task_id, args.xlsx)
        for result in results:
            print(
                f"{result.filename}: 导入 {result.imported_segments}，"
                f"精确匹配 {result.exact_matches}，顺序匹配 {result.positional_matches}"
            )
        _print_report(report)
        return 0 if report.can_generate else 2

    if args.command == "validate":
        _, report = workflow.validate(args.task_id)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_report(report)
        return 0 if report.can_generate else 2

    if args.command == "edit":
        _, report = workflow.update_translations(
            args.task_id, {args.segment_id: args.translation}
        )
        _print_report(report)
        return 0 if report.can_generate else 2

    if args.command == "generate":
        source_layout = args.output_mode == "source-layout"
        _, outputs = workflow.generate(
            args.task_id,
            chinese=False,
            bilingual=False,
            layout=not source_layout,
            source_layout=source_layout,
            show_segment_ids=False,
            font_path=args.font,
            layout_dpi=args.layout_dpi,
            progress=_progress,
        )
        if not source_layout and outputs.layout_pdf:
            print(f"原版面中文版：{outputs.layout_pdf}")
        if source_layout and outputs.source_layout_pdf:
            print(f"原文 + 原版面中文版：{outputs.source_layout_pdf}")
        return 0

    if args.command == "list":
        for task in workflow.repository.list_tasks():
            print(
                f"{task.task_id}\t{task.status}\t{task.page_count} 页\t"
                f"{task.source_filename}"
            )
        return 0

    if args.command == "ui":
        command = [
            sys.executable,
            "-m",
            "pdf_translator.hybrid_app",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        if args.no_browser:
            command.append("--no-browser")
        if getattr(args, "data_dir", None):
            command.extend(["--data-dir", args.data_dir])
        return subprocess.call(command)
    return 1


def _print_report(report) -> None:
    print(
        f"段落 {report.translated_segments}/{report.total_segments}，"
        f"阻断错误 {report.blocking_errors}，警告 {report.warnings}，"
        f"可以生成：{'是' if report.can_generate else '否'}"
    )
    for issue in report.issues[:30]:
        location = f" [{issue.segment_id}]" if issue.segment_id else ""
        print(f"- {issue.severity}/{issue.code}{location}: {issue.message}")
    if len(report.issues) > 30:
        print(f"- 另有 {len(report.issues) - 30} 条问题未显示")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(run(args))
    except (PDFTranslatorError, FileNotFoundError, ValueError, KeyError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
