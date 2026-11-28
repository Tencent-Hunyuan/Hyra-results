# Installation and first-time setup

Run the commands in this guide from the repository root on a Linux host. The installer downloads the Linux build of Node.js and installs Python dependencies with `venv` and pip. You need Python 3.10 or later, `bash`, `curl`, `tar`, xz, and `setsid`, plus network access to the dependency repositories and the model service.

## 1. Prepare the host and configuration

```bash
cp -n config/codex.env.example config/codex.env
cp -n config/hyprover.env.example config/hyprover.env
chmod 600 config/codex.env config/hyprover.env
```

`cp -n` preserves any existing configuration. The files use Bash syntax, so quote values that contain special characters. Keep API keys and account sign-in data on the machine where you run HyProver.

## 2. Choose a model backend

Research runs through the Codex CLI. Choose the backend that fits your deployment.

### Option A for your own Responses API endpoint

Set the API base URL, key, and a supported model in `config/codex.env`.

```bash
CODEX_BACKEND='api'
CODEX_API_BASE_URL='https://api.example.com/v1'
HYPROVER_CODEX_API_KEY='replace-with-your-key'
CODEX_API_MODEL='gpt-6-astra'
```

Your endpoint must support the Responses API and the tool calls Codex uses. Omit `/responses` from the base URL. If `config/hyprover.env` contains a previous `CODEX_BACKEND=chatgpt` override, remove it.

### Option B for Codex account sign-in

Set the backend to `chatgpt` in `config/hyprover.env`. Use an account with Codex access. You can leave the API URL and key unset.

```bash
CODEX_BACKEND='chatgpt'
```

### Model and worker settings for either backend

New projects default to `gpt-6-astra`, `max` reasoning effort, and eight workers. Adjust the model and effort if your backend supports different options. To try a run with one worker, create the project with `./bin/hyprover new demo --roles max:1` and follow the operations guide. [Model precedence](configuration.md#model-precedence-and-worker-count) · [Create a project](operating-guide.md#create-a-project)

Manuscript generation uses the Responses API. If you use account sign-in for research, you can enable the API backend for the manuscript process alone. [Manuscript generation guide](article.md#generate-a-manuscript)

## 3. Install runtime dependencies

For an existing deployment, stop the relevant processes first. To preserve custom Codex settings, back up `$CODEX_HOME/config.toml` before installing. The installer can generate this file when all API settings are present.

```bash
bash scripts/bootstrap.sh
. ./scripts/env.sh
./bin/codex --version
```

The installer places Node.js in `runtime/node22/`, the Codex CLI in `runtime/codex-npm/`, and the Python environment in `runtime/venv/`. It records their paths in `runtime/runtime.env` so the repository wrappers can launch Codex. The installer reuses existing components and rebuilds an unusable Python environment. The default location for Codex configuration and sign-in data is `runtime/codex-home/`. [Recovery and upgrades](operating-guide.md#recover-or-update)

## 4. Enable the selected backend

Run the commands for the backend you selected. Setting up the API backend overwrites `$CODEX_HOME/config.toml`. When you switch to account sign-in, the script removes that file if it contains the current `hyprover_api` provider configuration.

### A. Generate the API provider configuration

```bash
bash scripts/setup-codex.sh api
bash scripts/setup-codex.sh status
```

The script generates a Responses API provider named `hyprover_api`. Codex reads the key from the environment at runtime. Regenerate the configuration after changing the API URL, model, or retry settings.

### B. Complete Codex device sign-in

```bash
bash scripts/setup-codex.sh login
bash scripts/setup-codex.sh status
```

Follow the terminal instructions to enter and authorize the device code in your browser. You can run the terminal commands on a server over SSH. Sign-in data is stored in the deployment's `CODEX_HOME`. The `status` command should report the `chatgpt` backend and your sign-in status. The `CODEX_BACKEND=chatgpt` setting from step 2 persists the backend choice.

<a id="verify-model-access"></a>

## 5. Start the verification service and check model access

```bash
bash scripts/services.sh up verify
bash scripts/services.sh status
curl --fail --silent --show-error "http://127.0.0.1:${VERIFY_PORT}/health"
```

The verification service listens on `127.0.0.1:8091` by default. The `/health` endpoint returns its status and process ID. Run `services.sh status` to check that the process belongs to this deployment. Then check model access separately.

```bash
bash scripts/check-codex.sh
```

In API mode, the check sends a billable Responses request using `CODEX_API_MODEL` and scans recent verifier and worker logs. In account mode, it checks sign-in status. Both checks write to `$HYPROVER_RUNTIME/logs/codex-health.jsonl`. The `doctor.sh` and `services.sh test` commands can also send billable requests in API mode. [Check the deployment](operating-guide.md#service-health)

## 6. Start your first research session

```bash
./bin/codex
```

Provide the complete problem, the criteria for success, any materials available to the agents, limits on search and computation, the budget, and stopping conditions. Before you start, check that the verification service is ready and that the backend supports the models and reasoning effort configured for the main agent, workers, and verifiers. To run the main agent in the background, create a project, fill in `PROBLEM.md`, and run `./bin/hyprover main start demo`, where `demo` is the project name. [Operations guide](operating-guide.md)

## 7. Set up a development environment and run offline checks

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q hyprover
```

The offline tests use temporary projects and mock models, so you can run them without real credentials or running services. Tests for isolated PDF compilation and text extraction require additional Linux tools and are skipped when those tools are unavailable. To install dependencies for individual components, use the `api`, `gateway`, `verify`, or `observability` extra. [Manuscript PDF toolchain](article.md#compile-a-pdf)
