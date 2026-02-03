import os
import json
import tempfile
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import fnmatch
import requests
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from git import Repo

# App Setup and configuration
# Fastapi and HTML template configuration
app = FastAPI(title="Git commit mining & feature prediction web app")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Output directory for mined,predicted,reviewed JSON files
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Ignore commits that only change docs,meta files to reduce prediction noise
PREDICT_IGNORE_GLOBS = [
    "README*", "CHANGELOG*", "LICENSE*",
    "docs/**", ".github/**",
    "*.md", "*.txt",
    "build/**", "dist/**", "target/**", "out/**",
    "node_modules/**", "vendor/**",
    ".git/**",
    "*.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "*.min.js", "*.map",
]

# ECCO connection settings
ECCO_BASE_URL = os.environ.get("ECCO_BASE_URL", "http://localhost:8081").rstrip("/")
ECCO_USER = os.environ.get("ECCO_USER", "Tobias")
ECCO_PASS = os.environ.get("ECCO_PASS", "admin")

# Local cache for cloned repos used for ECCO pushes
REPO_CACHE_DIR = Path("repo_cache")
REPO_CACHE_DIR.mkdir(exist_ok=True)

# ECCO push file size limit per file. To reduce upload size and prevent ECCO crashes.
ECCO_MAX_FILE_BYTES = int(os.environ.get("ECCO_MAX_FILE_BYTES", str(1 * 1024 * 1024)))  # 1 MB per file

# Ignore commit files for ECCO upload (binaries, assets, large files)
ECCO_IGNORE_GLOBS = [
    *PREDICT_IGNORE_GLOBS,
    "__pycache__/**", ".venv/**", "venv/**",
    "*.class", "*.jar", "*.war", "*.exe", "*.dll","assets/**", "static/**", "public/**",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg",
    "*.mp4", "*.zip", "*.tar", "*.gz",
]

# Ollama endpoint used for feature prediction
OLLAMA_URL = "http://localhost:11434/api"

# Pydantic models (mining output schema)
class RepositoryInfo(BaseModel):
    name: str
    url: str

class ChangedFile(BaseModel):
    path: str
    status: str

class CommitInfo(BaseModel):
    hash: str
    message: str
    changed_files: List[ChangedFile]
    patch_excerpt: str  

class MiningResult(BaseModel):
    repository: RepositoryInfo
    commits: List[CommitInfo]

# Shared helper functions
# Reads JSON file from data directory
def load_json_file(file_name: str) -> Dict[str, Any]:
    filepath = os.path.join(DATA_DIR, file_name)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="JSON file not found")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    
# Check if a path matches any of the given glob patterns
def matches_any_glob(path_posix: str, globs: List[str]) -> bool:
    p = (path_posix or "").replace("\\", "/").lstrip("./")
    for g in globs:
        gg = (g or "").replace("\\", "/").lstrip("./")
        if fnmatch.fnmatch(p, gg):
            return True
    return False

# Cleans a feature name so ECCO accepts it
def clean_feature_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)
    return name or "Feature"

# Extracts a clean and stable repository name from a Git URL so the same name can be safely used across ECCO and local cache
def derive_repo_name_from_git_url(git_url: str) -> str:
    name = git_url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return clean_feature_name(name)

# Mining section
# Collect changed files (added/modified/deleted/renamed) for a commit so LLM know what parts of the system were affected (used in mining and prediction)
def get_changed_files(commit) -> List[ChangedFile]:
    changed_files: List[ChangedFile] = []
    # Root commit: treat all files as "added"
    if not commit.parents:
        for obj in commit.tree.traverse():
            if obj.type == "blob":
                changed_files.append(ChangedFile(path=obj.path, status="added"))
        return changed_files

    parent = commit.parents[0]
    diffs = parent.diff(commit)

    for diff in diffs:
        if diff.new_file:
            status = "added"
            path = diff.b_path
        elif diff.deleted_file:
            status = "deleted"
            path = diff.a_path
        elif diff.renamed:
            status = "renamed"
            path = diff.b_path
        else:
            status = "modified"
            path = diff.b_path

        if path is None:
            path = "(unknown)"

        changed_files.append(ChangedFile(path=path, status=status))

    return changed_files

# Extract only added lines from the diff (skips root commits and ignores noisy files)
def get_added_code_lines(
    commit,
    max_added_lines_total: int = 80,
    max_added_lines_per_file: int = 20,
    max_chars_per_line: int = 300,
) -> str:
    if max_added_lines_total <= 0:
        return ""
    if not commit.parents:
        return ""

    parent = commit.parents[0]
    diffs = parent.diff(commit, create_patch=True)

    parts: List[str] = []
    total_added = 0

    for d in diffs:
        if total_added >= max_added_lines_total:
            break

        path = (d.b_path or d.a_path or "").strip()
        if not path:
            continue

        # Ignore noisy files for better prompts
        if matches_any_glob(path, PREDICT_IGNORE_GLOBS):
            continue

        raw = d.diff
        if not raw:
            continue

        patch_text = ""
        try:
            patch_text = raw.decode("utf-8", errors="ignore")
        except Exception:
            continue

        added_lines = []
        for line in patch_text.splitlines():
            if total_added >= max_added_lines_total:
                break

            if line.startswith("+") and not line.startswith("+++"):
                content = line[1:]
                if len(content) > max_chars_per_line:
                    content = content[:max_chars_per_line] + " …"
                added_lines.append(content)
                total_added += 1

                if len(added_lines) >= max_added_lines_per_file:
                    break

        if added_lines:
            parts.append(f"\n--- Added lines in {path} ---")
            parts.extend(added_lines)

    return "\n".join(parts).strip()

# Clone and mine a Git repository
def mine_repository(
    git_url: str,
    max_commits,
    patch_lines_per_commit: int = 80,
) -> MiningResult:
    tmp_dir = tempfile.mkdtemp(prefix="repo-")
    try:
        repo = Repo.clone_from(git_url, tmp_dir)

        repo_name = derive_repo_name_from_git_url(git_url)
        repo_info = RepositoryInfo(name=repo_name, url=git_url)

        commits: List[CommitInfo] = []

        if max_commits is None:
            commit_iter = repo.iter_commits("HEAD")
        else:
            commit_iter = repo.iter_commits("HEAD", max_count=max_commits)

        for commit in commit_iter:
            commits.append(
                CommitInfo(
                    hash=commit.hexsha,
                    message=commit.message.strip(),
                    changed_files=get_changed_files(commit),
                    patch_excerpt=get_added_code_lines(
                        commit,
                        max_added_lines_total=patch_lines_per_commit,
                    ),
                )
            )

        return MiningResult(repository=repo_info, commits=commits)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Mining failed: {e}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# Save mined result to data directory with unique filename
def save_mining_result(result: MiningResult) -> str:
    repo_name = result.repository.name
    prefix = f"mined-{repo_name}-"

    existing = [
        f for f in os.listdir(DATA_DIR)
        if f.startswith(prefix) and f.endswith(".json")
    ]

    idx = len(existing) + 1
    filename = f"{prefix}{idx}.json"
    filepath = os.path.join(DATA_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    return filename

# Prediction section
# Filter out changed files that match ignore globs
def filter_changed_files(commit, globs):
    out = []
    for cf in (commit.get("changed_files", []) or []):
        p = (cf.get("path") or "")
        if not matches_any_glob(p, globs):
            out.append(cf)
    return out

# Format a single commit into text for the model prompt
def format_commit_for_prompt(
    commit: Dict[str, Any],
    *,
    include_patch: bool,
    ignore_globs: List[str],
    max_files: int = 25,   # limit helps batch stability
) -> str:
    msg = (commit.get("message") or "").strip()
    changed_files = filter_changed_files(commit, ignore_globs)

    shown = changed_files[:max_files]
    omitted = len(changed_files) - len(shown)

    lines = []
    for cf in shown:
        lines.append(f"- {cf.get('status','?')}: {cf.get('path','?')}")

    if omitted > 0:
        lines.append(f"- (omitted {omitted} more files)")

    files_block = "\n".join(lines) if lines else "(no relevant files after ignore rules)"

    text = (
        f"hash: {commit.get('hash')}\n"
        f"message: {msg}\n"
        f"files:\n{files_block}\n"
    )

    if include_patch:
        patch = commit.get("patch_excerpt") or "(no patch excerpt available)"
        text += f"patch_excerpt:\n{patch}\n"

    return text

# Get installed Ollama models for the UI so that users can select from available options
def get_available_models() -> List[str]:
    try:
        resp = requests.get(f"{OLLAMA_URL}/tags", timeout=5)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []

# Call Ollama API to generate response for given prompt
def call_ollama(model: str, prompt: str) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 2000  # allow longer JSON output in batch mode
                }
            },
            timeout=180,
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=500, detail="Could not connect to Ollama. Is the Ollama server running?")

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ollama error: {resp.status_code} {resp.text}")

    data = resp.json()
    return data.get("response", "").strip()

#  Safely parse JSON from model response, handling common formatting issues and extracting the JSON object within the text
def safe_parse_json(raw_response: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw_response)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    text = (raw_response or "").strip()

    if "```" in text:
        parts = text.split("```")
        biggest = ""
        for i in range(1, len(parts), 2):
            block = parts[i]
            block = block.split("\n", 1)[-1] if "\n" in block else block
            if len(block) > len(biggest):
                biggest = block
        if biggest.strip():
            text = biggest.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    candidate = text[start:end + 1].strip()

    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    # Fallback try simple repairs (smart quotes, trailing commas)
    fixed = candidate
    fixed = fixed.replace("“", "\"").replace("”", "\"").replace("’", "'")

    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    start2 = fixed.find("{")
    end2 = fixed.rfind("}")
    if start2 == -1 or end2 == -1 or end2 <= start2:
        return {}

    fixed = fixed[start2:end2 + 1].strip()

    try:
        data = json.loads(fixed)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# Extract results from batch prediction response, mapping short hashes to full commit hashes
def extract_batch_results(data: Dict[str, Any], batch_for_prompt: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = data.get("results", {})
    if not isinstance(results, dict):
        return {}

    full_hashes = [str(c.get("hash") or "") for c in batch_for_prompt if c.get("hash")]
    out: Dict[str, Any] = {}

    for h in full_hashes:
        if h in results:
            out[h] = results[h]

    return out

# Normalize features returned by the model
def normalize_features(features: Any) -> List[Dict[str, Any]]:
    cleaned = []
    if not isinstance(features, list):
        return cleaned

    for f in features:
        if not isinstance(f, dict):
            continue

        name = str(f.get("name") or "").strip()
        if not name:
            continue

        op = str(f.get("operation") or "UPDATE").strip().upper()
        if op not in ("ADD", "UPDATE"):
            op = "UPDATE"

        files = f.get("files") or []
        if not isinstance(files, list):
            files = []

        cleaned.append({
            "name": name,
            "operation": op,
            "files": files,
        })

    return cleaned

# Build the predicted commit structure
def build_predicted_commit(commit: Dict[str, Any], filtered_changed_files: List[Dict[str, Any]], features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "hash": commit.get("hash"),
        "message": commit.get("message"),
        "changed_files": filtered_changed_files,
        "patch_excerpt": commit.get("patch_excerpt", None),
        "features": features,
    }

# Build a prompt for single or batch prediction mode
def build_prompt(*, mode: str, commit_blocks: List[str], extra_prompt: str) -> str:
    context = ""
    if extra_prompt.strip():
        context = "Additional instructions from the user:\n" + extra_prompt.strip() + "\n\n"

    if mode == "batch":
        instructions = (
            "Task: Predict high-level software FEATURES for each commit.\n"
            "Commits are ordered OLDEST to MOST RECENT.\n\n"
            "Rules:\n"
            "- The VERY FIRST character of your reply must be '{' and the VERY LAST character must be '}'.\n"
            "- Output ONLY raw JSON (no markdown, no backticks, no code fences).\n"
            "- Do NOT write any sentences like 'Here is the JSON...'.\n"
            "- No extra text before or after the JSON.\n"
            "- Use the EXACT full commit hash as the key.\n"
            "- Include EVERY commit shown (features can be []).\n"
            "- Never use '...' anywhere.\n"
            "- For each feature, list at most 10 file paths (truncate if more).\n\n"
            "Return exactly this JSON format (note the braces and commas):\n"
            "{\n"
            "  \"results\": {\n"
            "    \"<commit_hash>\": {\n"
            "      \"features\": [\n"
            "        {\n"
            "          \"name\": \"Feature Name\",\n"
            "          \"operation\": \"ADD\",\n"
            "          \"files\": [\"path1\", \"path2\"]\n"
            "        }\n"
            "      ]\n"
            "    }\n"
            "  }\n"
            "}\n"
        )
        commits_text = "\n\n".join([f"[{i+1}]\n{blk}" for i, blk in enumerate(commit_blocks)])
        return f"{context}{instructions}\nCOMMITS:\n{commits_text}\n"

    # single mode
    instructions = (
        "You are helping the user analyze a Git commit to predict the FEATURES of a software system.\n"
        "A feature is a user-visible function or capability (e.g., Login, Search, Payment).\n\n"
        "For this commit:\n"
        "1) Identify the high-level feature(s)\n"
        "2) Decide if each feature is NEW (ADD) or an UPDATE (UPDATE)\n"
        "3) Map each feature to relevant changed files\n\n"
        "Please respond with raw JSON only and do not add any extra text.\n"
        "Return the JSON in exactly this format:\n"
        "{\n"
        "  \"features\": [\n"
        "    {\"name\": \"...\", \"operation\": \"ADD\"|\"UPDATE\", \"files\": [\"...\"]}\n"
        "  ]\n"
        "}\n"
    )
    return f"{context}{instructions}\nCOMMIT:\n{commit_blocks[0]}\n"


# Run prediction on mined data using the specified model and parameters
def run_prediction_on_mined_data(mined_data, model, max_commits, extra_prompt, prediction_mode="single", batch_size=5, include_patch=True):
    all_commits = mined_data.get("commits", [])
    commits = all_commits if max_commits is None else all_commits[:max_commits]

    kept = []
    for c in commits:
        changed = c.get("changed_files", []) or []
        if not changed:
            kept.append(c)
            continue
        if len(filter_changed_files(c, PREDICT_IGNORE_GLOBS)) > 0:
            kept.append(c)

    commits = kept

    predicted_commits = []
    if prediction_mode != "batch":
        prediction_mode = "single"

    if prediction_mode == "single":
        for commit in commits:
            commit_text = format_commit_for_prompt(
                commit,
                include_patch=include_patch,
                ignore_globs=PREDICT_IGNORE_GLOBS,
            )

            prompt_text = build_prompt(
                mode="single",
                commit_blocks=[commit_text],
                extra_prompt=extra_prompt,
            )

            raw_response = call_ollama(model, prompt_text)

            data = safe_parse_json(raw_response)
            if not data:
                raise HTTPException(status_code=500, detail="Single mode: model did not return valid JSON")

            filtered = filter_changed_files(commit, PREDICT_IGNORE_GLOBS)
            norm_features = normalize_features(data.get("features", []))
            predicted_commits.append(build_predicted_commit(commit, filtered, norm_features))

    else:
        if batch_size <= 0:
            batch_size = 5

        i = 0
        while i < len(commits):
            batch = commits[i:i + batch_size]

            batch_for_prompt = list(reversed(batch))
            use_patch = include_patch

            commit_blocks = []
            for c in batch_for_prompt:
                commit_blocks.append(
                    format_commit_for_prompt(c, include_patch=use_patch, ignore_globs=PREDICT_IGNORE_GLOBS)
                )

            prompt_text = build_prompt(mode="batch", commit_blocks=commit_blocks, extra_prompt=extra_prompt)
            raw_response = call_ollama(model, prompt_text)

            data = safe_parse_json(raw_response)
            if not data:
                snippet = (raw_response or "").strip().replace("\n", " ")
                if len(snippet) > 800:
                    snippet = snippet[:800] + " …"
                raise HTTPException(status_code=500, detail=f"Batch mode: invalid JSON from model. Output starts with: {snippet}")

            mapped = extract_batch_results(data, batch_for_prompt)

            for c in batch_for_prompt:
                h = str(c.get("hash") or "")
                if h and h not in mapped:
                    mapped[h] = {"features": []}

            for commit in batch_for_prompt:
                h = str(commit.get("hash") or "")
                commit_result = mapped.get(h, {}) or {}

                filtered = filter_changed_files(commit, PREDICT_IGNORE_GLOBS)
                norm_features = normalize_features(commit_result.get("features", []))
                predicted_commits.append(build_predicted_commit(commit, filtered, norm_features))

            i += batch_size

    feature_counts: Dict[str, int] = {}

    for c in predicted_commits:
        feats = c.get("features", []) or []
        for f in feats:
            name = (f.get("name") or "").strip()
            if not name:
                continue
            if name not in feature_counts:
                feature_counts[name] = 0
            feature_counts[name] += 1

    features_list = []
    for name in sorted(feature_counts.keys(), key=lambda s: s.lower()):
        features_list.append({
            "name": name,
            "occurrences": feature_counts[name],
        })

    summary = {
        "commit_count": len(predicted_commits),
        "unique_feature_count": len(feature_counts),
        "features": features_list,
        "mode": prediction_mode,
        "batch_size": batch_size if prediction_mode == "batch" else 1,
        "include_patch": include_patch,
    }

    return {
        "repository": mined_data.get("repository"),
        "commits": predicted_commits,
        "summary": summary,
        "meta": {
            "model": model,
            "prompt": extra_prompt,
            "max_commits": max_commits,
            "prediction_mode": prediction_mode,
            "batch_size": batch_size,
            "include_patch": include_patch,
        },
    }

# Save predicted result to data directory with unique filename
def save_predicted_result(predicted_data: Dict[str, Any]) -> str:
    repo = predicted_data.get("repository", {})
    repo_name = repo.get("name", "repo")

    prefix = f"predicted-{repo_name}-"

    existing = [
        f for f in os.listdir(DATA_DIR)
        if f.startswith(prefix) and f.endswith(".json")
    ]

    idx = len(existing) + 1
    filename = f"{prefix}{idx}.json"
    filepath = os.path.join(DATA_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(predicted_data, f, indent=2)

    return filename

# List the last mined JSON file in the data directory
def list_mined_files() -> List[str]:
    return sorted([n for n in os.listdir(DATA_DIR) if n.startswith("mined-") and n.endswith(".json")])


# ECCO functions section
# Login to ECCO and get access token
def ecco_login() -> str:
    try:
        resp = requests.post(
            f"{ECCO_BASE_URL}/login",
            json={"username": ECCO_USER, "password": ECCO_PASS},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=500, detail=f"Could not connect to {ECCO_BASE_URL}")

    if resp.status_code >= 400:
        body = (resp.text or "").strip()
        if len(body) > 1500:
            body = body[:1500] + " …"
        raise HTTPException(status_code=500, detail=f"ECCO login failed: {resp.status_code} {resp.reason}: {body}")

    data = resp.json() if resp.content else {}
    token = data.get("access_token")
    if not token:
        raise HTTPException(status_code=500, detail=f"ECCO login missing access_token: {data}")
    return token

# List all repositories in ECCO
def ecco_list_repos(token: str) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(
            f"{ECCO_BASE_URL}/api/repository/all",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=500, detail=f"Could not connect to {ECCO_BASE_URL}")

    if resp.status_code >= 400:
        body = (resp.text or "").strip()
        if len(body) > 1500:
            body = body[:1500] + " …"
        raise HTTPException(status_code=500, detail=f"ECCO list repos failed: {resp.status_code} {resp.reason}: {body}")

    data = resp.json() if resp.content else []
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail=f"ECCO /api/repository/all returned unexpected JSON: {data}")
    return data

# Get a specific repository by ID from ECCO
def ecco_get_repo(token: str, repo_id: int) -> Dict[str, Any]:
    try:
        resp = requests.get(
            f"{ECCO_BASE_URL}/api/repository/{repo_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=500, detail=f"Could not connect to {ECCO_BASE_URL}")

    if resp.status_code >= 400:
        body = (resp.text or "").strip()
        if len(body) > 1500:
            body = body[:1500] + " …"
        raise HTTPException(
            status_code=500,
            detail=f"ECCO get repo failed: {resp.status_code} {resp.reason} (repo_id={repo_id}): {body}",
        )

    return resp.json() if resp.content else {}

# Ensure a repository with the desired name exists in ECCO, creating it if necessary
def ecco_ensure_repo(token: str, desired_name: str) -> Dict[str, Any]:
    desired_name = (desired_name or "").strip() or "repo"

    # If repo already exists, return it
    repos = ecco_list_repos(token)
    for r in repos:
        if isinstance(r, dict) and r.get("name") == desired_name:
            return r

    # Create a new repo
    try:
        resp = requests.put(
            f"{ECCO_BASE_URL}/api/repository/{desired_name}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=500, detail=f"Could not connect to {ECCO_BASE_URL}")

    if resp.status_code >= 400:
        body = (resp.text or "").strip()
        if len(body) > 1500:
            body = body[:1500] + " …"
        raise HTTPException(status_code=500, detail=f"ECCO create repo failed: {resp.status_code} {resp.reason}: {body}")

    data = resp.json() if resp.content else {}

    # Check response for created repo
    if isinstance(data, dict) and data.get("repositoryHandlerId") is not None:
        return data
    if isinstance(data, list):
        for r in data:
            if isinstance(r, dict) and r.get("name") == desired_name:
                return r

    # Re-check all repos to find the newly created one
    repos2 = ecco_list_repos(token)
    for r in repos2:
        if isinstance(r, dict) and r.get("name") == desired_name:
            return r

    raise HTTPException(status_code=500, detail=f"Repo '{desired_name}' could not be created/found.")

# Determine if an ECCO commit already exists for the given Git hash
def ecco_commit_exists(repo_json: Dict[str, Any], git_hash: str) -> bool:
    needle = f"[git:{git_hash}]"
    for c in (repo_json.get("commits") or []):
        msg = (c.get("commitMessage") or "")
        if needle in msg:
            return True
    return False

# Build ECCO config string from feature names and revision suffix
def build_ecco_config_string(feature_names: List[str], revision_suffix: str) -> str:
    suffix = (revision_suffix or "").strip() or "1"
    seen = set()
    out: List[str] = []
    for n in feature_names:
        safe = clean_feature_name(n)
        rev = f"{safe}.{suffix}"
        if rev not in seen:
            seen.add(rev)
            out.append(rev)
    return ", ".join(out)

# Clone repo into local cache (reuses it on later pushes)
def ensure_repo_cached(git_url: str) -> Path:
    repo_name = derive_repo_name_from_git_url(git_url)
    local_path = REPO_CACHE_DIR / repo_name

    if local_path.exists() and (local_path / ".git").exists():
        return local_path

    if local_path.exists():
        shutil.rmtree(local_path, ignore_errors=True)

    Repo.clone_from(git_url, str(local_path))
    return local_path

# Export relevant files from a specific commit into a temporary folder for ECCO upload
def export_files_for_commit(repo_path: Path, commit_hash: str, changed_files: List[Dict[str, Any]]) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="ecco-commit-"))
    repo = Repo(str(repo_path))
    commit = repo.commit(commit_hash)

    files_to_export: List[str] = []
    for cf in changed_files or []:
        status = (cf.get("status") or "").lower().strip()
        path = (cf.get("path") or "").strip()
        if not path or path == "(unknown)" or status == "deleted":
            continue
        rel = path.replace("\\", "/")
        if matches_any_glob(rel, ECCO_IGNORE_GLOBS):
            continue
        files_to_export.append(rel)

    files_to_export = sorted(set(files_to_export))

    exported = 0
    skipped_big: List[str] = []

    for rel in files_to_export:
        try:
            blob = commit.tree / rel
            if getattr(blob, "type", None) != "blob":
                continue
            if getattr(blob, "size", 0) > ECCO_MAX_FILE_BYTES:
                skipped_big.append(f"{rel} ({blob.size} bytes)")
                continue

            out_path = tmp_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(blob.data_stream.read())
            exported += 1
        except Exception:
            continue

    if exported == 0:
        placeholder = tmp_dir / "ECCO_EMPTY_EXPORT.txt"
        placeholder.write_text(
            "No files exported (deleted/ignored/missing/too large).\n"
            + ("\nSkipped big:\n" + "\n".join(skipped_big[:200]) if skipped_big else ""),
            encoding="utf-8",
        )

    return tmp_dir

# Add a commit to ECCO with files from the specified folder for upload
def ecco_commit_add(token, repo_id, message, username, config, commit_folder):
    url = f"{ECCO_BASE_URL}/api/{repo_id}/commit/add"
    headers = {"Authorization": f"Bearer {token}"}

    # Prevent ECCO from crashing with too many / too large uploads
    MAX_PER_FILE = ECCO_MAX_FILE_BYTES
    MAX_TOTAL_BYTES = 3 * 1024 * 1024
    MAX_FILES = 75

    total_bytes = 0
    file_count = 0
    skipped = []

    files_payload = []
    opened_files = [] 

    try:
        for p in commit_folder.rglob("*"):
            if not p.is_file():
                continue

            rel_name = str(p.relative_to(commit_folder)).replace("\\", "/")
            size = p.stat().st_size

            if size > MAX_PER_FILE:
                skipped.append(f"{rel_name} (too big)")
                continue
            if file_count >= MAX_FILES:
                skipped.append(f"{rel_name} (too many files)")
                continue
            if total_bytes + size > MAX_TOTAL_BYTES:
                skipped.append(f"{rel_name} (total limit)")
                continue

            fh = open(p, "rb")
            opened_files.append(fh)
            files_payload.append(("file", (rel_name, fh, "application/octet-stream")))
            total_bytes += size
            file_count += 1

        if file_count == 0:
            placeholder_path = commit_folder / "ECCO_EMPTY_EXPORT.txt"
            if not placeholder_path.exists():
                placeholder_path.write_text(
                    "No files exported (deleted/ignored/missing/too large).\n"
                    + ("Skipped:\n" + "\n".join(skipped[:200]) if skipped else ""),
                    encoding="utf-8",
                )
            fh = open(placeholder_path, "rb")
            opened_files.append(fh)
            files_payload.append(("file", ("ECCO_EMPTY_EXPORT.txt", fh, "text/plain")))

        try:
            resp = requests.post(
                url,
                headers=headers,
                data={"message": message, "username": username, "config": config},
                files=files_payload,
                timeout=(30, 900),
            )
        except requests.exceptions.RequestException as ex:
            raise HTTPException(status_code=500, detail=f"ECCO commit/add request failed: {type(ex).__name__}: {ex}")

        if resp.status_code >= 400:
            body = (resp.text or "").strip()
            if len(body) > 1500:
                body = body[:1500] + " …"
            raise HTTPException(status_code=500, detail=f"ECCO commit/add failed: {resp.status_code} {resp.reason}: {body}")

        return resp.json() if resp.content else {}

    finally:
        for fh in opened_files:
            try:
                fh.close()
            except Exception:
                pass


# Routes (UI + actions)
@app.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    return templates.TemplateResponse("home_mining.html", {"request": request})

@app.post("/mine", response_class=HTMLResponse)
def mine_form(
    request: Request,
    git_url: str = Form(...),
    commit_limit: str = Form("100"),
    patch_lines_per_commit: int = Form(80),
):
    try:
        max_commits = None if commit_limit == "max" else int(commit_limit)

        # Clamp patch excerpt size to keep prompts reasonable
        if patch_lines_per_commit < 0:
            patch_lines_per_commit = 0
        if patch_lines_per_commit > 400:
            patch_lines_per_commit = 400

        u = git_url.strip()
        if not (u.startswith("http://") or u.startswith("https://") or u.startswith("git@") or u.startswith("ssh://")):
            u = "https://" + u
        git_url = u

        result = mine_repository(git_url, max_commits, patch_lines_per_commit)
        json_file = save_mining_result(result)

        return templates.TemplateResponse(
            "mine_result.html",
            {
                "request": request,
                "repo": result.repository,
                "count": len(result.commits),
                "file": json_file,
                "limit_label": commit_limit,
                "patch_lines_per_commit": patch_lines_per_commit,
            },
        )
    except Exception as e:
        return templates.TemplateResponse("mine_result.html", {"request": request, "error": str(e)})

@app.get("/prediction", response_class=HTMLResponse)
def prediction_form(request: Request, file: str = None):
    if file:
        mined_files = [file]
    else:
        mined_files = list_mined_files()
        mined_files = mined_files[-1:]  
    models = get_available_models() or ["mistral"]
    return templates.TemplateResponse("predict.html", {"request": request, "mined_files": mined_files, "available_models": models})

@app.post("/prediction", response_class=HTMLResponse)
def prediction_run(
    request: Request,
    mined_file: str = Form(...),
    model_name: str = Form("mistral"),
    commit_limit_prediction: str = Form("100"),
    prompt: str = Form(""),
    prediction_mode: str = Form("single"),
    batch_size: int = Form(5),
    include_patch: str = Form("yes"),
):
    try:
        mined_data = load_json_file(mined_file)

        max_commits_pred = None if commit_limit_prediction == "max" else int(commit_limit_prediction)
        include_patch_bool = (include_patch == "yes")

        predicted_data = run_prediction_on_mined_data(
            mined_data,
            model=model_name,
            max_commits=max_commits_pred,
            extra_prompt=prompt,
            prediction_mode=prediction_mode,
            batch_size=batch_size,
            include_patch=include_patch_bool,
        )

        predicted_file = save_predicted_result(predicted_data)
        summary = predicted_data.get("summary", {})

        return templates.TemplateResponse(
            "predict_result.html",
            {
                "request": request,
                "mined_file": mined_file,
                "predicted_file": predicted_file,
                "summary": summary,
                "model_name": model_name,
                "limit_label": commit_limit_prediction,
                "prompt": prompt,
                "prediction_mode": prediction_mode,
                "batch_size": batch_size,
                "include_patch": include_patch_bool,
            },
        )
    except Exception as e:
        return templates.TemplateResponse("predict_result.html", {"request": request, "error": str(e)})

@app.get("/download/{file_name}")
def download(file_name: str):
    filepath = os.path.join(DATA_DIR, file_name)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, media_type="application/json", filename=file_name)

@app.get("/review", response_class=HTMLResponse)
def review_page(request: Request, file: str):
    predicted_data = load_json_file(file)
    commits = predicted_data.get("commits", [])
    return templates.TemplateResponse(
        "review.html",
        {
            "request": request,
            "predicted_file": file,
            "repo": predicted_data.get("repository", {}),
            "meta": predicted_data.get("meta", {}),
            "commits": commits,
        },
    )

@app.post("/review/save", response_class=HTMLResponse)
async def review_save(request: Request):
    form = await request.form()

    predicted_file = str(form.get("predicted_file"))
    predicted_data = load_json_file(predicted_file)

    stored_commits = []
    accepted_total = 0

    for c in predicted_data.get("commits", []):
        chash = (c.get("hash") or "").strip()
        if not chash:
            continue
        accepted_features = []

        for idx, feat in enumerate(c.get("features", [])):
            decision_key = f"decision::{chash}::{idx}"
            name_key = f"name::{chash}::{idx}"

            decision = str(form.get(decision_key, "rejected"))
            if decision != "accepted":
                continue

            edited_name = str(form.get(name_key, "")).strip()
            final_name = edited_name if edited_name else (feat.get("name") or "UNKNOWN")

            accepted_features.append({
                "original_name": feat.get("name"),
                "name": final_name,
                "edited": final_name != (feat.get("name") or ""),
                "files": feat.get("files", []),
                "operation": feat.get("operation"),
                "source": "llm",
            })

        if not accepted_features:
            continue

        accepted_total += len(accepted_features)
        stored_commits.append({
            "hash": chash,
            "message": c.get("message"),
            "changed_files": c.get("changed_files", []),
            "features": accepted_features,
        })

    reviewed_data = {
        "repository": predicted_data.get("repository"),
        "meta": {
            "reviewed_at": datetime.now().isoformat(),
            "based_on_predicted_file": predicted_file,
            "model": (predicted_data.get("meta") or {}).get("model"),
            "prompt": (predicted_data.get("meta") or {}).get("prompt"),
        },
        "commits": stored_commits,
        "summary": {
            "stored_commit_count": len(stored_commits),
            "accepted_features_total": accepted_total,
        },
    }

    repo_name = (predicted_data.get("repository") or {}).get("name", "repo")
    prefix = f"reviewed-{repo_name}-"

    existing = [
        f for f in os.listdir(DATA_DIR)
        if f.startswith(prefix) and f.endswith(".json")
    ]

    idx = len(existing) + 1
    reviewed_file = f"{prefix}{idx}.json"

    reviewed_path = os.path.join(DATA_DIR, reviewed_file)
    with open(reviewed_path, "w", encoding="utf-8") as f:
        json.dump(reviewed_data, f, indent=2)

    return templates.TemplateResponse(
        "review_result.html",
        {
            "request": request,
            "predicted_file": predicted_file,
            "reviewed_file": reviewed_file,
            "summary": reviewed_data["summary"],
        },
    )

@app.post("/ecco/push", response_class=HTMLResponse)
async def ecco_push(request: Request):
    form = await request.form()

    file_name = (form.get("reviewed_file") or form.get("result_file") or "").strip()
    repo_id_raw = (form.get("ecco_repo_id") or form.get("repository_handler_id") or "").strip()
    mode = (form.get("mode") or "reviewed").strip()

    if not file_name:
        return templates.TemplateResponse("ecco_result.html", {"request": request, "error": "Missing reviewed_file/result_file."})

    try:
        data = load_json_file(file_name)
        git_url = (data.get("repository") or {}).get("url")
        if not git_url:
            raise HTTPException(status_code=400, detail="JSON missing repository.url")

        token = ecco_login()

        # Use user-provided repo id, otherwise ensure repo by derived name
        if repo_id_raw:
            ecco_repo_id = int(repo_id_raw)
        else:
            repo_name = derive_repo_name_from_git_url(git_url)
            repo_obj = ecco_ensure_repo(token, repo_name)
            ecco_repo_id = int(repo_obj["repositoryHandlerId"])

        repo_path = ensure_repo_cached(git_url)
        repo_json = ecco_get_repo(token, ecco_repo_id)

        pushed_hashes = []
        skipped_hashes = []
        errors = []

        for c in data.get("commits", []):
            commit_hash = (c.get("hash") or "").strip()
            base_msg = (c.get("message") or "").strip() or "commit from feature miner"
            changed_files = c.get("changed_files", []) or []
            feats = c.get("features", []) or []

            # Skip commits that would export nothing useful
            relevant = []
            for cf in changed_files:
                p = (cf.get("path") or "").strip()
                if p and p != "(unknown)" and not matches_any_glob(p, ECCO_IGNORE_GLOBS):
                    relevant.append(p)

            if changed_files and not relevant:
                skipped_hashes.append(commit_hash)
                continue

            if not commit_hash:
                skipped_hashes.append("(missing-hash)")
                continue

            feature_names = [(f.get("name") or "").strip() for f in feats]
            feature_names = [n for n in feature_names if n] or ["UNKNOWN_FEATURE"]

            config_str = build_ecco_config_string(feature_names, revision_suffix=commit_hash[:7])
            tagged_msg = f"{base_msg} [git:{commit_hash}]"

            if ecco_commit_exists(repo_json, commit_hash):
                skipped_hashes.append(commit_hash)
                continue

            commit_folder = export_files_for_commit(repo_path, commit_hash, changed_files)

            try:
                ecco_commit_add(
                    token=token,
                    repo_id=ecco_repo_id,
                    message=tagged_msg,
                    username=ECCO_USER,
                    config=config_str,
                    commit_folder=commit_folder,
                )
                pushed_hashes.append(commit_hash)

                repo_json.setdefault("commits", []).append({
                    "commitMessage": tagged_msg,
                    "configuration": {"configurationString": config_str},
                })

            except Exception as e:
                errors.append(f"{commit_hash}: {str(e)}")
            finally:
                shutil.rmtree(commit_folder, ignore_errors=True)

        return templates.TemplateResponse(
            "ecco_result.html",
            {
                "request": request,
                "mode": mode,
                "source_file": file_name,
                "ecco_repo_id": ecco_repo_id,
                "result": {
                    "pushed_commits": pushed_hashes,
                    "skipped_commits": skipped_hashes,
                    "pushed_count": len(pushed_hashes),
                    "skipped_count": len(skipped_hashes),
                    "errors": errors,
                },
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "ecco_result.html",
            {
                "request": request,
                "mode": mode,
                "source_file": file_name,
                "ecco_repo_id": repo_id_raw or "",
                "error": str(e),
            },
        )

@app.get("/ecco/debug/repos")
def ecco_debug_repos():
    token = ecco_login()
    repos = ecco_list_repos(token)
    return {"base_url": ECCO_BASE_URL, "repo_count": len(repos), "repos": repos}