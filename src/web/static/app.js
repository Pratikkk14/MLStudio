// ML STUDIO — Modern Vanilla JS Application Controller

let currentStatus = "idle";
let logOffset = 0;
let statusPollInterval = null;
let logPollInterval = null;
let currentReportData = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initControls();
  initDatasets();
  fetchStatus();
  fetchReport();
  fetchLogs();

  // Start polling
  statusPollInterval = setInterval(fetchStatus, 2000);
  logPollInterval = setInterval(fetchLogs, 1500);
});

// ============================================================================
// Tabs Controller
// ============================================================================
function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) targetPane.classList.add("active");
    });
  });
}

// ============================================================================
// Controls & Event Listeners
// ============================================================================
function initControls() {
  const runBtn = document.getElementById("runAutoMLBtn");
  const stopBtn = document.getElementById("stopAutoMLBtn");
  const clearLogsBtn = document.getElementById("clearLogsBtn");
  const predictionForm = document.getElementById("predictionForm");
  const loadPresetBtn = document.getElementById("loadSampleInputsBtn");
  const datasetSelect = document.getElementById("datasetSelect");
  const uploadFolderBtn = document.getElementById("uploadFolderBtn");
  const folderInput = document.getElementById("folderInput");

  if (uploadFolderBtn && folderInput) {
    uploadFolderBtn.addEventListener("click", () => folderInput.click());
    folderInput.addEventListener("change", handleFolderUpload);
  }

  if (runBtn) {
    runBtn.addEventListener("click", triggerAutoMLRun);
  }

  if (stopBtn) {
    stopBtn.addEventListener("click", stopAutoMLRun);
  }

  if (clearLogsBtn) {
    clearLogsBtn.addEventListener("click", () => {
      document.getElementById("terminalBody").textContent = "";
      logOffset = 0;
    });
  }

  if (predictionForm) {
    predictionForm.addEventListener("submit", handlePredictionSubmit);
  }

  if (loadPresetBtn) {
    loadPresetBtn.addEventListener("click", loadSamplePreset);
  }

  if (datasetSelect) {
    datasetSelect.addEventListener("change", () => {
      fetchDatasetPreview(datasetSelect.value);
    });
  }
}

// ============================================================================
// Folder Upload Controller
// ============================================================================
async function handleFolderUpload(e) {
  const files = Array.from(e.target.files);
  if (!files || files.length === 0) return;

  const uploadBtn = document.getElementById("uploadFolderBtn");
  const originalText = uploadBtn.textContent;
  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading...";

  // Find root folder name from relative path (e.g. "my_data/train.csv" -> "my_data")
  let folderName = "uploaded_dataset";
  if (files[0].webkitRelativePath) {
    folderName = files[0].webkitRelativePath.split("/")[0] || folderName;
  }

  // Filter only tabular files
  const tabularFiles = files.filter(f => {
    const ext = f.name.split(".").pop().toLowerCase();
    return ["csv", "xlsx", "xls"].includes(ext);
  });

  if (tabularFiles.length === 0) {
    alert("No CSV or Excel dataset files (.csv, .xlsx, .xls) found inside the selected folder.");
    uploadBtn.disabled = false;
    uploadBtn.textContent = originalText;
    return;
  }

  try {
    const filePayloads = await Promise.all(tabularFiles.map(file => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const b64 = reader.result.split(",")[1];
          // Use relative path or filename
          const relPath = file.webkitRelativePath 
            ? file.webkitRelativePath.replace(/^[^/]+\//, "") 
            : file.name;
          resolve({ path: relPath, content_base64: b64 });
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    }));

    const res = await fetch("/api/upload-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        folder_name: folderName,
        files: filePayloads
      })
    });

    const data = await res.json();
    if (res.ok && data.success) {
      await initDatasets();
      const select = document.getElementById("datasetSelect");
      if (select) {
        select.value = data.dataset_path;
        fetchDatasetPreview(data.dataset_path);
      }
      
      // Switch to Dataset Audit tab so user immediately sees the detected split
      const dsTabBtn = document.getElementById("tabBtnDataset");
      if (dsTabBtn) dsTabBtn.click();

      // Update status notification
      const statusText = document.getElementById("globalStatusText");
      if (statusText) {
        statusText.textContent = data.message;
      }
    } else {
      alert(`Upload failed: ${data.error || "Unknown error"}`);
    }
  } catch (err) {
    alert(`Upload error: ${err}`);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = originalText;
    e.target.value = "";
  }
}

// ============================================================================
// Datasets
// ============================================================================
async function initDatasets() {
  try {
    const res = await fetch("/api/datasets");
    if (!res.ok) return;
    const data = await res.json();
    const select = document.getElementById("datasetSelect");
    if (data.datasets && data.datasets.length > 0) {
      const currentVal = select.value;
      select.innerHTML = "";
      data.datasets.forEach(ds => {
        const opt = document.createElement("option");
        opt.value = ds.path;
        opt.textContent = `${ds.name} (${formatBytes(ds.size_bytes)})`;
        select.appendChild(opt);
      });
      if (currentVal && Array.from(select.options).some(o => o.value === currentVal)) {
        select.value = currentVal;
      }
      fetchDatasetPreview(select.value);
    }
  } catch (err) {
    console.error("Error loading datasets list:", err);
  }
}

async function fetchDatasetPreview(datasetPath) {
  try {
    const filename = datasetPath.split(/[\\/]/).pop();
    const res = await fetch(`/api/dataset-preview?dataset=${encodeURIComponent(datasetPath)}`);
    if (!res.ok) return;
    const data = await res.json();
    renderDatasetPreview(data);
  } catch (err) {
    console.error("Error fetching preview:", err);
  }
}

function renderDatasetPreview(data) {
  // Stats
  document.getElementById("dsFileName").textContent = data.filename || "—";
  document.getElementById("dsShape").textContent = `${data.total_rows_sampled} rows × ${data.columns_count} cols (preview)`;
  
  const splitElem = document.getElementById("dsSplitStrategy");
  if (splitElem) {
    if (data.is_presplit) {
      splitElem.textContent = `Pre-split Train (${data.train_file}) + Test (${data.test_file})`;
      splitElem.className = "spec-val highlight-green mono";
    } else {
      splitElem.textContent = data.split_strategy || "Single Dataset (Stratified 80/20 Auto-Split)";
      splitElem.className = "spec-val highlight-blue mono";
    }
  }

  // Columns dictionary
  const colTbody = document.getElementById("columnsTbody");
  if (colTbody && data.columns) {
    colTbody.innerHTML = "";
    let numCount = 0;
    let catCount = 0;

    data.columns.forEach(col => {
      const isNum = col.dtype.includes("int") || col.dtype.includes("float");
      if (isNum) numCount++; else catCount++;

      let action = "Standard Feature";
      let actionBadge = "badge-subtle";

      if (currentReportData && currentReportData.dataset) {
        if (col.name === currentReportData.dataset.target) {
          action = "Prediction Target";
          actionBadge = "badge-success";
        } else if (currentReportData.preprocessing && currentReportData.preprocessing.exclude_columns && currentReportData.preprocessing.exclude_columns.includes(col.name)) {
          action = "Excluded (Risk/Constant)";
          actionBadge = "badge-warning";
        }
      }

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono font-semibold">${col.name}</td>
        <td class="mono text-secondary">${col.dtype}</td>
        <td class="mono">${col.null_count}</td>
        <td class="mono">${col.unique_count}</td>
        <td><span class="badge ${actionBadge}">${action}</span></td>
      `;
      colTbody.appendChild(tr);
    });

    document.getElementById("dsNumCount").textContent = numCount;
    document.getElementById("dsCatCount").textContent = catCount;
  }

  // Preview table
  const thead = document.getElementById("previewThead");
  const tbody = document.getElementById("previewTbody");
  if (thead && tbody && data.rows && data.rows.length > 0) {
    const cols = Object.keys(data.rows[0]);
    thead.innerHTML = `<tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr>`;
    tbody.innerHTML = data.rows.map(row => {
      return `<tr>${cols.map(c => `<td>${row[c]}</td>`).join("")}</tr>`;
    }).join("");
  }
}

// ============================================================================
// Status & Runner Management
// ============================================================================
async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();
    currentStatus = data.status;

    // Update global status badge
    const dot = document.querySelector(".status-indicator-dot");
    const text = document.getElementById("globalStatusText");
    const runBtn = document.getElementById("runAutoMLBtn");
    const stopBtn = document.getElementById("stopAutoMLBtn");
    const spinner = document.getElementById("btnSpinner");
    const label = document.getElementById("btnLabel");

    if (dot && text) {
      dot.className = `status-indicator-dot ${data.status}`;
      if (data.status === "running") {
        text.textContent = `Stage ${data.stage}/8: ${data.stage_name}`;
        runBtn.disabled = true;
        spinner.style.display = "inline-block";
        label.textContent = "Running...";
        stopBtn.style.display = "inline-flex";
      } else if (data.status === "completed") {
        text.textContent = `Completed (${data.elapsed_seconds}s)`;
        runBtn.disabled = false;
        spinner.style.display = "none";
        label.textContent = "Re-Run Pipeline";
        stopBtn.style.display = "none";
      } else if (data.status === "error") {
        text.textContent = `Error: ${data.stage_name}`;
        runBtn.disabled = false;
        spinner.style.display = "none";
        label.textContent = "Retry Run";
        stopBtn.style.display = "none";
      } else {
        text.textContent = "Ready";
        runBtn.disabled = false;
        spinner.style.display = "none";
        label.textContent = "Run Pipeline";
        stopBtn.style.display = "none";
      }
    }

    // Update stage tracker
    updateStagesTracker(data.stage, data.status);

    // Update timer
    const timer = document.getElementById("runnerTimer");
    if (timer) {
      timer.textContent = `Elapsed: ${data.elapsed_seconds}s`;
    }

    // If just completed and report not fresh, refetch report
    if (data.status === "completed" && (!currentReportData || currentReportData._loadedTime !== data.elapsed_seconds)) {
      fetchReport();
    }
  } catch (err) {
    console.error("Status fetch error:", err);
  }
}

function updateStagesTracker(currentStage, status) {
  const steps = document.querySelectorAll(".stage-step");
  steps.forEach(step => {
    const num = parseInt(step.dataset.step, 10);
    step.classList.remove("active", "completed");
    if (status === "completed") {
      step.classList.add("completed");
    } else if (num < currentStage) {
      step.classList.add("completed");
    } else if (num === currentStage && status === "running") {
      step.classList.add("active");
    }
  });
}

async function triggerAutoMLRun() {
  const dataset = document.getElementById("datasetSelect").value;
  const agentMode = document.getElementById("modeSelect").value;

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: dataset,
        agent_mode: agentMode
      })
    });
    const data = await res.json();
    if (res.ok) {
      logOffset = 0;
      document.getElementById("terminalBody").textContent = ">>> Launching ML Studio pipeline...\n";
      // Switch to runner tab
      document.querySelector('[data-tab="tab-runner"]').click();
      fetchStatus();
    } else {
      alert(`Could not start run: ${data.error || "Unknown error"}`);
    }
  } catch (err) {
    alert(`Request error: ${err}`);
  }
}

async function stopAutoMLRun() {
  try {
    await fetch("/api/stop", { method: "POST" });
    fetchStatus();
  } catch (err) {
    console.error("Stop run error:", err);
  }
}

// ============================================================================
// Terminal Logs
// ============================================================================
async function fetchLogs() {
  try {
    const res = await fetch(`/api/logs?offset=${logOffset}`);
    if (!res.ok) return;
    const data = await res.json();

    if (data.logs && data.logs.length > 0) {
      const term = document.getElementById("terminalBody");
      const autoScroll = document.getElementById("autoScrollCheckbox").checked;
      
      const fragment = document.createDocumentFragment();
      data.logs.forEach(line => {
        const lineNode = document.createTextNode(line + "\n");
        fragment.appendChild(lineNode);
      });
      term.appendChild(fragment);
      logOffset = data.total;

      if (autoScroll) {
        const container = document.getElementById("terminalOutput");
        container.scrollTop = container.scrollHeight;
      }
    }
  } catch (err) {
    // Silent fail on log poll
  }
}

// ============================================================================
// Report & Analytics Rendering
// ============================================================================
async function fetchReport() {
  try {
    const res = await fetch("/api/report");
    if (!res.ok) return;
    const report = await res.json();
    currentReportData = report;
    currentReportData._loadedTime = Date.now();

    renderKPIStrip(report);
    renderOverview(report);
    renderTournament(report);
    renderCognitiveTrace(report);
    renderQualityAudit(report);
  } catch (err) {
    console.error("Report fetch error:", err);
  }
}

function renderKPIStrip(rep) {
  const best = rep.best_model || {};
  const ds = rep.dataset || {};
  const metrics = best.test_metrics || {};

  document.getElementById("kpiChampion").textContent = best.name || "—";
  document.getElementById("kpiProblemType").textContent = `Task: ${formatTitle(ds.problem_type || "Classification")}`;

  const f1Val = metrics.f1 !== undefined ? metrics.f1.toFixed(4) : "—";
  document.getElementById("kpiF1").textContent = f1Val;
  document.getElementById("kpiCvF1").textContent = `5-Fold CV: ${best.cv_score ? best.cv_score.toFixed(4) : "—"}`;

  const accVal = metrics.accuracy !== undefined ? `${(metrics.accuracy * 100).toFixed(1)}%` : "—";
  document.getElementById("kpiAccuracy").textContent = accVal;
  document.getElementById("kpiRocAuc").textContent = `ROC-AUC: ${metrics.roc_auc ? metrics.roc_auc.toFixed(4) : "—"}`;

  const totalFeats = (best.top_features ? best.top_features.length : 0);
  const excludedCount = (rep.preprocessing && rep.preprocessing.exclude_columns ? rep.preprocessing.exclude_columns.length : 0);
  document.getElementById("kpiFeatures").textContent = `${totalFeats} Transformed`;
  document.getElementById("kpiExcluded").textContent = `Excluded: ${excludedCount} cols`;

  const rows = ds.shape ? ds.shape[0] : "—";
  const missing = rep.data_quality ? `${(rep.data_quality.missing_pct || 0).toFixed(1)}%` : "0.0%";
  document.getElementById("kpiRows").textContent = `${rows} Observations`;
  document.getElementById("kpiMissing").textContent = `Missing: ${missing}`;
}

function renderOverview(rep) {
  const best = rep.best_model || {};
  const ds = rep.dataset || {};
  const preproc = rep.preprocessing || {};

  // Specifications
  document.getElementById("specModelName").textContent = best.name || "—";
  document.getElementById("specTarget").textContent = ds.target || "—";
  document.getElementById("specMetric").textContent = "F1-Score (Macro)";
  document.getElementById("specScaling").textContent = formatTitle(preproc.scaling || "Standard");
  document.getElementById("specEncoding").textContent = formatTitle(preproc.categorical_encoding || "One-Hot");
  document.getElementById("specClassWeights").textContent = preproc.use_class_weights ? "Balanced (Active)" : "None";

  // Hyperparameters
  const hpTbody = document.getElementById("hyperparamsTbody");
  if (hpTbody) {
    hpTbody.innerHTML = "";
    const params = best.best_params || {};
    const keys = Object.keys(params);
    if (keys.length === 0) {
      hpTbody.innerHTML = `<tr><td colspan="2" class="text-muted">Default parameter configuration</td></tr>`;
    } else {
      keys.forEach(k => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="mono">${k}</td>
          <td class="mono highlight-green">${params[k]}</td>
        `;
        hpTbody.appendChild(tr);
      });
    }
  }

  // Holdout test metrics
  const m = best.test_metrics || {};
  document.getElementById("valAccuracy").textContent = m.accuracy ? m.accuracy.toFixed(4) : "—";
  document.getElementById("valF1").textContent = m.f1 ? m.f1.toFixed(4) : "—";
  document.getElementById("valPrecision").textContent = m.precision ? m.precision.toFixed(4) : "—";
  document.getElementById("valRecall").textContent = m.recall ? m.recall.toFixed(4) : "—";
  document.getElementById("valRocAuc").textContent = m.roc_auc ? m.roc_auc.toFixed(4) : "—";

  // Confusion matrix
  const cm = best.confusion_matrix;
  if (cm && cm.length === 2 && cm[0].length === 2) {
    document.querySelector("#cmTN .cm-number").textContent = cm[0][0];
    document.querySelector("#cmFP .cm-number").textContent = cm[0][1];
    document.querySelector("#cmFN .cm-number").textContent = cm[1][0];
    document.querySelector("#cmTP .cm-number").textContent = cm[1][1];
  }

  // Feature importances
  const fiList = document.getElementById("featureImportanceList");
  if (fiList && best.top_features) {
    fiList.innerHTML = "";
    const maxScore = best.top_features.length > 0 ? best.top_features[0][1] : 1;

    best.top_features.forEach(([featName, score]) => {
      const pct = (score * 100).toFixed(1);
      const barWidth = Math.max(3, (score / maxScore) * 100).toFixed(1);

      const div = document.createElement("div");
      div.className = "fi-row";
      div.innerHTML = `
        <div class="fi-info">
          <span class="fi-name">${featName}</span>
          <span class="fi-score">${pct}%</span>
        </div>
        <div class="fi-bar-bg">
          <div class="fi-bar-fill" style="width: ${barWidth}%;"></div>
        </div>
      `;
      fiList.appendChild(div);
    });
  }

  // Final recommendation narrative
  const rationaleBox = document.getElementById("finalRecommendationReason");
  if (rationaleBox) {
    rationaleBox.textContent = rep.final_recommendation_reason || "No deployment rationale provided.";
  }
}

function renderTournament(rep) {
  const tbody = document.getElementById("tournamentTbody");
  if (!tbody) return;

  const exps = rep.baseline_experiments || [];
  if (exps.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="text-muted text-center py-20">No cross-validation tournament data found.</td></tr>`;
    return;
  }

  // Deduplicate by model name and sort by CV mean score descending
  const modelsMap = new Map();
  exps.forEach(e => {
    if (e.stage === "baseline") {
      modelsMap.set(e.model, e);
    }
  });

  const sortedModels = Array.from(modelsMap.values()).sort((a, b) => {
    const scA = a.cv_mean !== undefined ? a.cv_mean : (a.metrics ? a.metrics.f1 : 0);
    const scB = b.cv_mean !== undefined ? b.cv_mean : (b.metrics ? b.metrics.f1 : 0);
    return scB - scA;
  });

  document.getElementById("tournamentTotalModels").textContent = `${sortedModels.length} Models Evaluated`;

  const championName = (rep.best_model && rep.best_model.name) ? rep.best_model.name.toLowerCase() : "";

  tbody.innerHTML = sortedModels.map((item, idx) => {
    const m = item.metrics || {};
    const cvMean = item.cv_mean !== undefined ? item.cv_mean.toFixed(4) : (m.f1 ? m.f1.toFixed(4) : "—");
    const cvStd = item.cv_std !== undefined ? `± ${item.cv_std.toFixed(4)}` : "";
    const isChampion = item.model.toLowerCase().includes(championName) || championName.includes(item.model.toLowerCase());

    const family = getModelFamily(item.model);

    return `
      <tr class="${isChampion ? "champion-row" : ""}">
        <td class="mono font-semibold">${idx + 1}</td>
        <td>
          <div class="flex items-center gap-2">
            <span class="font-medium">${item.model}</span>
          </div>
        </td>
        <td><span class="badge badge-subtle">${family}</span></td>
        <td class="mono">
          <span class="highlight-green">${cvMean}</span>
          <span class="text-muted text-sm ml-1">${cvStd}</span>
        </td>
        <td class="mono">${m.accuracy ? m.accuracy.toFixed(4) : "—"}</td>
        <td class="mono">${m.precision ? m.precision.toFixed(4) : "—"}</td>
        <td class="mono">${m.recall ? m.recall.toFixed(4) : "—"}</td>
        <td class="mono">${m.roc_auc ? m.roc_auc.toFixed(4) : "—"}</td>
        <td class="mono text-muted">${item.runtime_seconds ? item.runtime_seconds.toFixed(2) + "s" : "—"}</td>
        <td>
          ${isChampion ? '<span class="badge badge-success">Champion (Tuned)</span>' : '<span class="badge badge-subtle">Baseline</span>'}
        </td>
      </tr>
    `;
  }).join("");
}

function renderCognitiveTrace(rep) {
  const container = document.getElementById("cognitiveTimeline");
  if (!container) return;

  const history = rep.agent_workflow || [];
  if (history.length === 0) {
    container.innerHTML = `<div class="text-muted">No agent decision history found.</div>`;
    return;
  }

  container.innerHTML = history.map((step, idx) => {
    const d = step.decision || {};
    const confidencePct = d.confidence ? Math.round(d.confidence * 100) : 95;
    const actionLabel = formatTitle(step.action || step.stage);

    return `
      <div class="timeline-step">
        <div class="timeline-dot"></div>
        <div class="timeline-step-header">
          <span class="timeline-step-title">[${idx + 1}/4] ${actionLabel}</span>
          <span class="badge badge-info">${confidencePct}% Confidence</span>
        </div>

        <div class="timeline-block">
          <div class="timeline-label">Observation</div>
          <div class="timeline-text">${d.observation || "—"}</div>
        </div>

        <div class="timeline-block">
          <div class="timeline-label">Agent Cognitive Rationale</div>
          <div class="timeline-text highlight-text">${d.reason || d.decision || "—"}</div>
        </div>

        ${d.parameters ? `
          <div class="timeline-block">
            <div class="timeline-label">Action Parameters</div>
            <pre class="timeline-params">${JSON.stringify(d.parameters, null, 2)}</pre>
          </div>
        ` : ""}
      </div>
    `;
  }).join("");
}

function renderQualityAudit(rep) {
  const flagsList = document.getElementById("qualityFlagsList");
  const warnings = (rep.data_quality && rep.data_quality.warnings) || [];

  document.getElementById("qualityFlagsCount").textContent = `${warnings.length} Quality Flags`;
  document.getElementById("dsDuplicates").textContent = rep.data_quality ? rep.data_quality.duplicates : 0;
  document.getElementById("dsMissingPct").textContent = rep.data_quality ? `${(rep.data_quality.missing_pct || 0).toFixed(2)}%` : "0.0%";

  if (flagsList) {
    if (warnings.length === 0) {
      flagsList.innerHTML = `<div class="badge badge-success">✓ Clean Dataset: Zero quality or leakage warnings flagged.</div>`;
    } else {
      flagsList.innerHTML = warnings.map(w => {
        return `
          <div class="p-2 mb-2" style="background-color: var(--accent-amber-bg); border: 1px solid var(--accent-amber); border-radius: 4px; font-size: 12px; color: var(--accent-amber);">
            ⚠ ${w}
          </div>
        `;
      }).join("");
    }
  }
}

// ============================================================================
// Interactive Model Sandbox
// ============================================================================
async function handlePredictionSubmit(e) {
  e.preventDefault();

  const age = parseFloat(document.getElementById("inputAge").value);
  const tenure = parseFloat(document.getElementById("inputTenure").value);
  const monthlyCharges = parseFloat(document.getElementById("inputMonthlyCharges").value);
  const gender = document.getElementById("inputGender").value;

  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  btn.textContent = "Computing...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        features: {
          age: age,
          tenure: tenure,
          monthly_charges: monthlyCharges,
          gender: gender
        }
      })
    });

    const data = await res.json();
    if (res.ok) {
      document.getElementById("predictionPlaceholder").style.display = "none";
      document.getElementById("predictionContent").style.display = "block";

      document.getElementById("predTargetName").textContent = data.target || "churn";
      document.getElementById("predClassValue").textContent = data.prediction;
      
      const badge = document.getElementById("predInterpretation");
      if (data.prediction == 1) {
        badge.textContent = "High Risk / Positive Flag";
        badge.className = "pred-badge badge-warning";
      } else {
        badge.textContent = "Low Risk / Negative Flag";
        badge.className = "pred-badge badge-success";
      }

      document.getElementById("infLatency").textContent = `Latency: ${data.latency_ms}ms`;

      // Probability bars
      const probContainer = document.getElementById("probBarsContainer");
      probContainer.innerHTML = "";
      if (data.probabilities && typeof data.probabilities === "object") {
        Object.entries(data.probabilities).forEach(([cls, prob]) => {
          const pct = (prob * 100).toFixed(1);
          const div = document.createElement("div");
          div.className = "prob-item";
          div.innerHTML = `
            <div class="prob-header">
              <span>Class ${cls}</span>
              <span>${pct}%</span>
            </div>
            <div class="fi-bar-bg">
              <div class="fi-bar-fill" style="width: ${pct}%; background-color: ${cls == 1 ? "var(--accent-amber)" : "var(--accent-green)"}"></div>
            </div>
          `;
          probContainer.appendChild(div);
        });
      }
    } else {
      alert(`Prediction failed: ${data.error}`);
    }
  } catch (err) {
    alert(`Error: ${err}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Compute Prediction";
  }
}

function loadSamplePreset() {
  const presets = [
    { age: 52, tenure: 2, monthlyCharges: 110.0, gender: "Male" },
    { age: 29, tenure: 48, monthlyCharges: 45.0, gender: "Female" },
    { age: 41, tenure: 15, monthlyCharges: 79.5, gender: "Female" }
  ];
  const choice = presets[Math.floor(Math.random() * presets.length)];
  document.getElementById("inputAge").value = choice.age;
  document.getElementById("inputTenure").value = choice.tenure;
  document.getElementById("inputMonthlyCharges").value = choice.monthlyCharges;
  document.getElementById("inputGender").value = choice.gender;
}

// ============================================================================
// Utilities
// ============================================================================
function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function formatTitle(str) {
  if (!str) return "";
  return str.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function getModelFamily(name) {
  const n = name.toLowerCase();
  if (n.includes("gradient") || n.includes("boost")) return "Boosting";
  if (n.includes("forest") || n.includes("bagg")) return "Bagging";
  if (n.includes("tree")) return "Decision Tree";
  if (n.includes("svm") || n.includes("kernel")) return "Kernel SVM";
  if (n.includes("voting")) return "Voting Ensemble";
  if (n.includes("stacking")) return "Stacking Ensemble";
  if (n.includes("linear") || n.includes("ridge") || n.includes("logistic")) return "Linear Model";
  return "Algorithm";
}
