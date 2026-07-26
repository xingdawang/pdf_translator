document.addEventListener("submit", async (event) => {
  const form = event.target.closest("[data-job-form]");
  if (!form) return;
  event.preventDefault();

  const button = form.querySelector('button[type="submit"]');
  const progress = form.querySelector(".job-progress");
  const bar = form.querySelector("[data-job-bar]");
  const message = form.querySelector("[data-job-message]");
  const percent = form.querySelector("[data-job-percent]");
  const error = form.querySelector("[data-job-error]");
  const recovery = form.querySelector("[data-job-recovery]");
  if (form.dataset.kind === "generate") {
    const previousOutputs = document.querySelector("[data-output-files]");
    if (previousOutputs) previousOutputs.hidden = true;
  }
  button.disabled = true;
  progress.hidden = false;
  bar.style.width = "0%";
  message.textContent = "正在提交任务";
  percent.textContent = "0%";
  error.textContent = "";
  if (recovery) recovery.hidden = true;

  try {
    const response = await fetch(form.action, { method: "POST", body: new FormData(form) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "请求失败");
    await pollJob(payload.job_id);
  } catch (reason) {
    error.textContent = reason.message || String(reason);
    if (reason.code === "NO_TEXT_LAYER" && recovery) {
      const recoveryMessage = recovery.querySelector("[data-recovery-message]");
      recoveryMessage.textContent = `${reason.message} 可以一键改用本地 Vision OCR。`;
      recovery.hidden = false;
    }
    button.disabled = false;
  }

  async function pollJob(jobId) {
    while (true) {
      const response = await fetch(`/api/jobs/${jobId}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法读取任务进度");
      const value = payload.progress || 0;
      bar.style.width = `${value}%`;
      message.textContent = payload.message;
      percent.textContent = `${Math.round(value)}%`;
      if (payload.state === "completed") {
        const target = new URL(payload.redirect_url, window.location.origin);
        if (
          target.pathname === window.location.pathname &&
          target.search === window.location.search
        ) {
          history.replaceState(null, "", target.hash || "#generate");
          window.location.reload();
        } else {
          window.location.href = target.href;
        }
        return;
      }
      if (payload.state === "failed") {
        const failure = new Error(payload.error || "后台任务失败");
        failure.code = payload.error_code;
        throw failure;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 700));
    }
  }
});

const exportForm = document.querySelector("[data-export-form]");
if (exportForm) {
  const button = exportForm.querySelector('button[type="submit"]');
  const status = exportForm.querySelector("[data-export-status]");
  exportForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    button.disabled = true;
    status.classList.remove("is-error");
    status.textContent = "正在生成翻译包…";
    try {
      const response = await fetch(exportForm.action, {
        method: "POST",
        body: new FormData(exportForm),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || "生成翻译包失败");
      }
      const blob = await response.blob();
      const disposition = response.headers.get("Content-Disposition") || "";
      const encodedName = disposition.match(/filename\*=UTF-8''([^;]+)/i);
      const simpleName = disposition.match(/filename="?([^";]+)"?/i);
      const filename = encodedName
        ? decodeURIComponent(encodedName[1])
        : simpleName?.[1] || "google_translate_package.xlsx";
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
      const packageCount = response.headers.get("X-Package-Count") || "1";
      status.textContent = `已生成并下载 ${packageCount} 个翻译包。`;
      history.replaceState(null, "", "#google");
      document.querySelector("#google")?.scrollIntoView({ behavior: "smooth" });
    } catch (reason) {
      status.classList.add("is-error");
      status.textContent = reason.message || String(reason);
    } finally {
      button.disabled = false;
    }
  });
}

const googleTranslateLink = document.querySelector("[data-google-translate]");
if (googleTranslateLink) {
  googleTranslateLink.addEventListener("click", () => {
    history.replaceState(null, "", "#import");
  });
}

const importForm = document.querySelector("[data-import-form]");
if (importForm) {
  const files = importForm.querySelector("[data-translation-files]");
  const label = importForm.querySelector("[data-import-label]");
  const status = importForm.querySelector("[data-import-status]");
  files.addEventListener("change", () => {
    if (!files.files?.length) return;
    const count = files.files.length;
    label.textContent =
      count === 1 ? files.files[0].name : `已选择 ${count} 个 XLSX`;
    status.textContent = "正在导入，后台自动修复缺失内容…";
    files.setAttribute("aria-busy", "true");
    importForm.requestSubmit();
  });
}

const analysisForm = document.querySelector("[data-analysis-form]");
if (analysisForm) {
  const pdfPickerButton = analysisForm.querySelector("[data-select-pdf]");
  const pathInput = analysisForm.querySelector("[data-source-path]");
  const pathStatus = analysisForm.querySelector("[data-path-status]");
  const pageStart = analysisForm.querySelector("[data-page-start]");
  const pageEnd = analysisForm.querySelector("[data-page-end]");
  const inspection = analysisForm.querySelector("[data-pdf-inspection]");
  const inspectionTitle = inspection.querySelector("[data-inspection-title]");
  const inspectionMeta = inspection.querySelector("[data-inspection-meta]");
  const inspectionMessage = inspection.querySelector("[data-inspection-message]");
  const ocrMode = analysisForm.querySelector("[data-ocr-mode]");
  const ocrModeHint = analysisForm.querySelector("[data-ocr-mode-hint]");
  const ocrDpi = analysisForm.querySelector("[data-ocr-dpi]");
  const ocrDpiField = analysisForm.querySelector("[data-ocr-dpi-field]");
  const ocrDpiHint = analysisForm.querySelector("[data-ocr-dpi-hint]");
  const retryVision = analysisForm.querySelector("[data-retry-vision]");
  let inspectionRequest = 0;
  let inspectionTimer = 0;

  function syncOcrControls() {
    const textOnly = ocrMode.value === "off";
    ocrDpi.disabled = textOnly;
    ocrDpiField.classList.toggle("is-disabled", textOnly);
    ocrModeHint.textContent = textOnly
      ? "不会运行 OCR，适合确定带有正常文字层的 PDF。"
      : "正常文字页直接读取文字层，只对没有文字层的页面运行本地 Vision OCR。";
    ocrDpiHint.textContent = textOnly
      ? "当前为仅文字层模式，OCR 清晰度不生效。"
      : "仅在图片页面需要 OCR 时生效。";
  }

  function setInspectionState(state) {
    inspection.classList.remove(
      "inspection-text",
      "inspection-mixed",
      "inspection-image",
      "inspection-error"
    );
    if (state) inspection.classList.add(`inspection-${state}`);
  }

  async function inspectPdf() {
    if (inspectionTimer) {
      window.clearTimeout(inspectionTimer);
      inspectionTimer = 0;
    }
    const suppliedPath = pathInput.value.trim();
    if (!suppliedPath) {
      inspection.hidden = true;
      return;
    }
    const requestNumber = ++inspectionRequest;
    inspection.hidden = false;
    setInspectionState("");
    inspectionTitle.textContent = "正在快速检查 PDF…";
    inspectionMeta.textContent = "最多抽取 5 个代表页面，不会提前 OCR 整本文件。";
    inspectionMessage.textContent = "";
    const data = new FormData();
    data.set("source_path", suppliedPath);
    data.set("page_start", pageStart.value || "1");
    data.set("page_end", pageEnd.value);
    try {
      const response = await fetch("/api/inspect-pdf", {
        method: "POST",
        body: data,
      });
      const payload = await response.json();
      if (requestNumber !== inspectionRequest) return;
      if (!response.ok) throw new Error(payload.error || "PDF 快速检查失败");
      setInspectionState(payload.status);
      const titles = {
        text: "✓ 检测到可用文字层",
        mixed: "部分页面可能需要 OCR",
        image: "未检测到可用文字层",
      };
      inspectionTitle.textContent = titles[payload.status] || "PDF 快速检查";
      inspectionMeta.textContent =
        `${payload.filename} · ${payload.size_label} · 共 ${payload.page_count} 页 · ` +
        `当前第 ${payload.page_start}–${payload.page_end} 页 · 抽样 ${payload.sampled_pages.length} 页`;
      inspectionMessage.textContent = payload.recommendation;
    } catch (reason) {
      if (requestNumber !== inspectionRequest) return;
      setInspectionState("error");
      inspectionTitle.textContent = "PDF 快速检查失败";
      inspectionMeta.textContent = "";
      inspectionMessage.textContent = reason.message || String(reason);
    }
  }

  function schedulePdfInspection() {
    if (inspectionTimer) window.clearTimeout(inspectionTimer);
    inspectionTimer = window.setTimeout(inspectPdf, 450);
  }

  ocrMode.addEventListener("change", syncOcrControls);
  pathInput.addEventListener("input", schedulePdfInspection);
  pageStart.addEventListener("input", schedulePdfInspection);
  pageEnd.addEventListener("input", schedulePdfInspection);
  pathInput.addEventListener("change", inspectPdf);
  pageStart.addEventListener("change", inspectPdf);
  pageEnd.addEventListener("change", inspectPdf);
  retryVision.addEventListener("click", () => {
    ocrMode.value = "vision";
    syncOcrControls();
    const recovery = analysisForm.querySelector("[data-job-recovery]");
    recovery.hidden = true;
    analysisForm.requestSubmit();
  });
  syncOcrControls();

  pdfPickerButton.addEventListener("click", async () => {
    pdfPickerButton.disabled = true;
    pathStatus.textContent = "正在打开 macOS 文件选择器…";
    try {
      const response = await fetch("/api/select-pdf", { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法打开文件选择器");
      if (payload.path) {
        pathInput.value = payload.path;
        pathInput.focus();
        pathStatus.textContent = "已选择 PDF；也可以继续手动修改路径。";
        await inspectPdf();
      } else {
        pathStatus.textContent = "已取消选择；仍可直接粘贴 PDF 路径。";
      }
    } catch (reason) {
      pathStatus.textContent = reason.message || String(reason);
    } finally {
      pdfPickerButton.disabled = false;
    }
  });
}

const workflowLinks = Array.from(
  document.querySelectorAll("[data-workflow-link]")
);
const workflowSections = Array.from(
  document.querySelectorAll("[data-workflow-section]")
);
if (workflowLinks.length && workflowSections.length) {
  let scrollFrame = 0;

  function highlightWorkflowStep() {
    scrollFrame = 0;
    const topbar = document.querySelector(".topbar");
    const threshold = (topbar?.offsetHeight || 68) + 28;
    let activeSection = workflowSections[0];
    for (const section of workflowSections) {
      if (section.getBoundingClientRect().top <= threshold) {
        activeSection = section;
      } else {
        break;
      }
    }
    if (
      window.innerHeight + window.scrollY >=
      document.documentElement.scrollHeight - 8
    ) {
      activeSection = workflowSections[workflowSections.length - 1];
    }
    for (const link of workflowLinks) {
      const isActive = link.dataset.workflowLink === activeSection.id;
      link.classList.toggle("active", isActive);
      if (isActive) {
        link.setAttribute("aria-current", "step");
      } else {
        link.removeAttribute("aria-current");
      }
    }
  }

  function queueWorkflowHighlight() {
    if (!scrollFrame) {
      scrollFrame = window.requestAnimationFrame(highlightWorkflowStep);
    }
  }

  for (const link of workflowLinks) {
    link.addEventListener("click", () => {
      for (const item of workflowLinks) item.classList.remove("active");
      link.classList.add("active");
    });
  }
  window.addEventListener("scroll", queueWorkflowHighlight, { passive: true });
  window.addEventListener("resize", queueWorkflowHighlight);
  window.addEventListener("hashchange", queueWorkflowHighlight);
  queueWorkflowHighlight();
  window.setTimeout(queueWorkflowHighlight, 120);
}
