import http.server
import json
import os
import sys
import threading
import subprocess
import time
import urllib.parse
import base64
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

STATIC_DIR = Path(__file__).resolve().parent / "static"
PYTHON_EXE = sys.executable

class WorkflowRunner:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"  # idle | running | completed | error
        self.stage = 0
        self.stage_name = "Ready"
        self.start_time = 0.0
        self.duration = 0.0
        self.logs: List[str] = []
        self.error_message = ""
        self.process: Optional[subprocess.Popen] = None
        self.dataset_name = ""
        
        # Check if there is already an existing report
        report_file = WORKSPACE_ROOT / "outputs" / "reports" / "final_report.json"
        if report_file.exists():
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    rep = json.load(f)
                    self.status = "completed"
                    self.stage = 8
                    self.stage_name = "Completed"
                    self.dataset_name = rep.get("dataset", {}).get("filename", "")
            except Exception:
                pass

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = time.time() - self.start_time if self.status == "running" else self.duration
            return {
                "status": self.status,
                "stage": self.stage,
                "stage_name": self.stage_name,
                "elapsed_seconds": round(elapsed, 1),
                "dataset_name": self.dataset_name,
                "error_message": self.error_message,
                "log_count": len(self.logs)
            }

    def start_run(self, dataset_path: str, agent_mode: str = "real", target: str = "", metric: str = "") -> bool:
        with self.lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.stage = 1
            self.stage_name = "Initializing Pipeline"
            self.start_time = time.time()
            self.duration = 0.0
            self.logs = []
            self.error_message = ""
            self.dataset_name = Path(dataset_path).name

        thread = threading.Thread(
            target=self._run_worker,
            args=(dataset_path, agent_mode, target, metric),
            daemon=True
        )
        thread.start()
        return True

    def _run_worker(self, dataset_path: str, agent_mode: str, target: str, metric: str):
        cmd = [
            PYTHON_EXE, "-m", "src.main",
            "--dataset", dataset_path,
            "--agent-mode", agent_mode
        ]
        if target:
            cmd.extend(["--target", target])
        if metric:
            cmd.extend(["--metric", metric])

        stage_mapping = {
            "[1/8]": (1, "Dataset Loading & Validation"),
            "[2/8]": (2, "Statistical Profiling & EDA"),
            "[3/8]": (3, "Agent Reasoning & Preprocessing"),
            "[4/8]": (4, "Preprocessing & Transformation"),
            "[5/8]": (5, "Agent Model Suite Selection"),
            "[6/8]": (6, "Cross-Validation Tournament"),
            "[7/8]": (7, "Hyperparameter Optimization"),
            "[8/8]": (8, "Holdout Evaluation & Artifact Export")
        }

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(WORKSPACE_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )

            for raw_line in iter(self.process.stdout.readline, ""):
                line = raw_line.rstrip()
                if not line:
                    continue
                with self.lock:
                    self.logs.append(line)
                    # Limit buffer size
                    if len(self.logs) > 3000:
                        self.logs.pop(0)

                    for marker, (st_num, st_name) in stage_mapping.items():
                        if marker in line:
                            self.stage = st_num
                            self.stage_name = st_name
                            break

            self.process.stdout.close()
            return_code = self.process.wait()

            with self.lock:
                self.duration = time.time() - self.start_time
                if return_code == 0:
                    self.status = "completed"
                    self.stage = 8
                    self.stage_name = "Workflow Completed"
                else:
                    self.status = "error"
                    self.error_message = f"Process exited with non-zero exit code {return_code}"
        except Exception as e:
            with self.lock:
                self.status = "error"
                self.error_message = str(e)
                self.duration = time.time() - self.start_time
        finally:
            self.process = None

    def stop_run(self):
        with self.lock:
            if self.process and self.status == "running":
                try:
                    self.process.terminate()
                except Exception:
                    pass
                self.status = "idle"
                self.stage_name = "Cancelled by user"

runner = WorkflowRunner()

class MLStudioRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format, *args):
        # Mute standard noisy HTTP access logs
        pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/status":
            self._send_json(runner.get_state())
            return

        if path == "/api/report":
            report_path = WORKSPACE_ROOT / "outputs" / "reports" / "final_report.json"
            if report_path.exists():
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._send_json(data)
                except Exception as e:
                    self._send_json({"error": f"Failed reading report: {e}"}, status=500)
            else:
                self._send_json({"error": "Report not generated yet"}, status=404)
            return

        if path == "/api/experiments":
            exp_path = WORKSPACE_ROOT / "outputs" / "experiments" / "experiment_history.json"
            if exp_path.exists():
                try:
                    with open(exp_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._send_json(data)
                except Exception as e:
                    self._send_json({"error": f"Failed reading experiments: {e}"}, status=500)
            else:
                self._send_json({"error": "Experiments not found"}, status=404)
            return

        if path == "/api/logs":
            offset = 0
            if "offset" in query:
                try:
                    offset = int(query["offset"][0])
                except ValueError:
                    offset = 0
            
            with runner.lock:
                if runner.logs:
                    log_lines = runner.logs[offset:]
                    total = len(runner.logs)
                else:
                    # Fallback to outputs/logs/workflow.log
                    log_file = WORKSPACE_ROOT / "outputs" / "logs" / "workflow.log"
                    if log_file.exists():
                        try:
                            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                                all_lines = [l.rstrip() for l in f.readlines()]
                                log_lines = all_lines[offset:]
                                total = len(all_lines)
                        except Exception:
                            log_lines = []
                            total = 0
                    else:
                        log_lines = []
                        total = 0

            self._send_json({
                "logs": log_lines,
                "total": total,
                "status": runner.status,
                "stage": runner.stage
            })
            return

        if path == "/api/datasets":
            datasets_dir = WORKSPACE_ROOT / "datasets"
            res = []
            if datasets_dir.exists():
                for p in datasets_dir.iterdir():
                    if p.name.startswith("."):
                        continue
                    if p.is_file() and p.suffix.lower() in [".csv", ".xlsx", ".xls"]:
                        res.append({
                            "name": p.name,
                            "path": str(p.relative_to(WORKSPACE_ROOT)),
                            "is_directory": False,
                            "is_presplit": False,
                            "split_strategy": "Single File (Stratified 80/20 Auto-Split)",
                            "size_bytes": p.stat().st_size,
                            "extension": p.suffix.lower().replace(".", "")
                        })
                    elif p.is_dir():
                        from src.data.loader import DatasetLoader
                        try:
                            loader = DatasetLoader(p)
                            info = loader.get_info()
                            total_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                            res.append({
                                "name": p.name + " 📁",
                                "path": str(p.relative_to(WORKSPACE_ROOT)),
                                "is_directory": True,
                                "is_presplit": info["is_presplit"],
                                "split_strategy": info["split_strategy"],
                                "train_file": Path(info["train_file"]).name if info.get("train_file") else None,
                                "test_file": Path(info["test_file"]).name if info.get("test_file") else None,
                                "size_bytes": total_size,
                                "extension": "folder"
                            })
                        except Exception:
                            pass
            self._send_json({"datasets": res})
            return

        if path == "/api/dataset-preview":
            filename = query.get("dataset", ["synthetic_dataset.xlsx"])[0]
            # sanitize filename / path
            safe_name = Path(filename).name
            target_path = WORKSPACE_ROOT / "datasets" / safe_name
            if not target_path.exists():
                # Check direct or relative path
                direct = WORKSPACE_ROOT / filename if not Path(filename).is_absolute() else Path(filename)
                if direct.exists():
                    target_path = direct
                else:
                    self._send_json({"error": f"Dataset '{filename}' not found"}, status=404)
                    return

            try:
                from src.data.loader import DatasetLoader
                loader = DatasetLoader(target_path)
                info = loader.get_info()
                df = loader.load()

                columns_info = []
                for col in df.columns:
                    columns_info.append({
                        "name": str(col),
                        "dtype": str(df[col].dtype),
                        "null_count": int(df[col].isna().sum()),
                        "unique_count": int(df[col].nunique())
                    })

                preview_rows = df.head(15).fillna("—").to_dict(orient="records")
                self._send_json({
                    "filename": target_path.name,
                    "is_directory": info["is_directory"],
                    "is_presplit": info["is_presplit"],
                    "split_strategy": info["split_strategy"],
                    "train_file": Path(info["train_file"]).name if info.get("train_file") else None,
                    "test_file": Path(info["test_file"]).name if info.get("test_file") else None,
                    "total_rows_sampled": len(df),
                    "columns_count": len(df.columns),
                    "columns": columns_info,
                    "rows": preview_rows
                })
            except Exception as e:
                self._send_json({"error": f"Preview error: {e}"}, status=500)
            return

        # Static assets
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            body_raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                payload = json.loads(body_raw)
            except Exception:
                payload = {}

            dataset = payload.get("dataset", "datasets/synthetic_dataset.xlsx")
            agent_mode = payload.get("agent_mode", "real")
            target = payload.get("target", "")
            metric = payload.get("metric", "")

            # resolve dataset path relative to root
            ds_path = WORKSPACE_ROOT / dataset if not Path(dataset).is_absolute() else Path(dataset)
            if not ds_path.exists():
                self._send_json({"error": f"Dataset file '{dataset}' does not exist."}, status=400)
                return

            started = runner.start_run(str(ds_path), agent_mode=agent_mode, target=target, metric=metric)
            if started:
                self._send_json({"success": True, "message": "AutoML pipeline started"})
            else:
                self._send_json({"error": "AutoML pipeline is already executing."}, status=409)
            return

        if path == "/api/stop":
            runner.stop_run()
            self._send_json({"success": True, "message": "AutoML pipeline cancelled."})
            return

        if path == "/api/upload-folder":
            length = int(self.headers.get("Content-Length", 0))
            body_raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                payload = json.loads(body_raw)
            except Exception as e:
                self._send_json({"error": f"Invalid JSON payload: {e}"}, status=400)
                return

            folder_name = payload.get("folder_name", "").strip()
            files = payload.get("files", [])

            if not folder_name:
                folder_name = f"dataset_folder_{int(time.time())}"

            # Sanitize folder name
            safe_folder = re.sub(r"[^\w\-_\.]", "_", folder_name)
            dest_dir = WORKSPACE_ROOT / "datasets" / safe_folder
            dest_dir.mkdir(parents=True, exist_ok=True)

            saved_files = []
            for item in files:
                rel_path = item.get("path", "")
                content_b64 = item.get("content_base64", "")
                if not rel_path or not content_b64:
                    continue

                clean_rel = Path(rel_path)
                if clean_rel.is_absolute() or ".." in clean_rel.parts or str(clean_rel).startswith("/"):
                    clean_rel = Path(clean_rel.name)

                out_file = dest_dir / clean_rel
                out_file.parent.mkdir(parents=True, exist_ok=True)

                try:
                    data_bytes = base64.b64decode(content_b64)
                    with open(out_file, "wb") as f_out:
                        f_out.write(data_bytes)
                    saved_files.append(str(clean_rel))
                except Exception as e:
                    pass

            from src.data.loader import DatasetLoader
            try:
                loader = DatasetLoader(dest_dir)
                info = loader.get_info()
                self._send_json({
                    "success": True,
                    "folder_name": safe_folder,
                    "dataset_path": f"datasets/{safe_folder}",
                    "saved_files_count": len(saved_files),
                    "is_presplit": info["is_presplit"],
                    "split_strategy": info["split_strategy"],
                    "train_file": Path(info["train_file"]).name if info.get("train_file") else None,
                    "test_file": Path(info["test_file"]).name if info.get("test_file") else None,
                    "primary_file": Path(info["primary_file"]).name if info.get("primary_file") else None,
                    "message": f"Successfully loaded folder '{safe_folder}' ({info['split_strategy']})"
                })
            except Exception as e:
                self._send_json({"error": f"Saved {len(saved_files)} files, but failed inspecting dataset: {e}"}, status=400)
            return

        if path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body_raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                payload = json.loads(body_raw)
            except Exception:
                payload = {}

            features = payload.get("features", {})
            if not features:
                self._send_json({"error": "No input feature dictionary provided."}, status=400)
                return

            try:
                import joblib
                import pandas as pd
                from src.tools.ml_tools import MLTools

                report_path = WORKSPACE_ROOT / "outputs" / "reports" / "final_report.json"
                model_path = WORKSPACE_ROOT / "outputs" / "models" / "best_model.joblib"

                if not report_path.exists() or not model_path.exists():
                    self._send_json({"error": "No trained model artifacts available. Please run AutoML first."}, status=400)
                    return

                with open(report_path, "r", encoding="utf-8") as f:
                    rep = json.load(f)

                t0 = time.time()
                model = joblib.load(model_path)

                # Recreate preprocessor pipeline and fit on training set
                dataset_file = WORKSPACE_ROOT / "datasets" / rep["dataset"]["filename"]
                raw_df = MLTools.load_dataset(str(dataset_file))
                target_col = rep["dataset"]["target"]
                X_train_raw = raw_df.drop(columns=[target_col])

                preproc = MLTools.build_preprocessing_pipeline(rep["preprocessing"])
                preproc.fit_transform(X_train_raw)

                # Prepare single row input
                input_df = pd.DataFrame([features])
                # Ensure missing columns in input get filled or handled
                for col in X_train_raw.columns:
                    if col not in input_df.columns:
                        input_df[col] = raw_df[col].iloc[0]

                X_input_proc = preproc.transform(input_df)
                pred = model.predict(X_input_proc)

                prob_dict = {}
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_input_proc)[0]
                    if hasattr(model, "classes_"):
                        for cls_name, p in zip(model.classes_, probs):
                            prob_dict[str(cls_name)] = round(float(p), 4)
                    else:
                        prob_dict = [round(float(p), 4) for p in probs]

                latency_ms = round((time.time() - t0) * 1000, 2)

                self._send_json({
                    "prediction": int(pred[0]) if hasattr(pred[0], "item") else str(pred[0]),
                    "probabilities": prob_dict,
                    "latency_ms": latency_ms,
                    "target": target_col,
                    "model_name": rep.get("best_model", {}).get("name", "Best Model")
                })
            except Exception as e:
                self._send_json({"error": f"Prediction failed: {e}"}, status=500)
            return

        self._send_json({"error": "Not Found"}, status=404)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def start_server(port: int = 8501, host: str = "127.0.0.1"):
    server_address = (host, port)
    httpd = http.server.ThreadingHTTPServer(server_address, MLStudioRequestHandler)
    print(f"ML Studio UI Server listening on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down ML Studio server.")
        httpd.server_close()

if __name__ == "__main__":
    port = 8501
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_server(port=port)
