# GitHub Actions CI/CD

This project deploys from GitHub Actions when code is pushed to `main`.

## What Runs

- Pull requests to `main`: install dependencies, compile Python modules, start the FastAPI app, and call `/health`.
- Pushes to `main`: run the same CI checks, then deploy by SSH when deployment is enabled.
- Manual runs: available from the GitHub Actions tab through `workflow_dispatch`.

## Required GitHub Settings

In the repository, open **Settings > Secrets and variables > Actions**.

Add this repository variable:

| Type | Name | Value |
| --- | --- | --- |
| Variable | `ENABLE_DEPLOY` | `true` |

Add these repository secrets:

| Type | Name | Example |
| --- | --- | --- |
| Secret | `DEPLOY_HOST` | `203.0.113.10` |
| Secret | `DEPLOY_USER` | `ubuntu` |
| Secret | `DEPLOY_SSH_KEY` | Private SSH key with access to the server |
| Secret | `DEPLOY_PATH` | `/opt/coverland-forecast-api` |
| Secret | `DEPLOY_PORT` | `22` |

`DEPLOY_PORT` can be omitted when the server uses port `22`.

## Server Setup

Create the deployment directory on the server and make sure the deploy user owns it:

```bash
sudo mkdir -p /opt/coverland-forecast-api
sudo chown -R ubuntu:ubuntu /opt/coverland-forecast-api
```

The workflow can restart either a system service named `coverland-forecast-api` or, if no service exists, run uvicorn in the background.
If you use the system service option, allow the deploy user to restart this one service with passwordless sudo.

Recommended systemd service:

```ini
[Unit]
Description=Coverland Forecast API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/coverland-forecast-api
EnvironmentFile=/opt/coverland-forecast-api/.env
ExecStart=/opt/coverland-forecast-api/.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Install and start it:

```bash
sudo tee /etc/systemd/system/coverland-forecast-api.service < coverland-forecast-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now coverland-forecast-api
```

Keep production environment variables, database credentials, and other secrets in the server-side `.env` file. The workflow does not overwrite `.env`.
