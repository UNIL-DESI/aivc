# Google Drive Sync Setup Guide

## Overview

AIVC supports optional cloud synchronization to share your agent's memory across multiple machines. 
This guide walks you through setting up native Google Drive sync.

## Prerequisites

- A Google account
- AIVC installed with sync dependencies: `pip install aivc[sync]` or `pip install aivc[all]`

## Step 1: Create Google Cloud OAuth Credentials

1. Go to the [Google Cloud Console — Credentials](https://console.cloud.google.com/apis/credentials).
2. Create a new project (or select an existing one).
3. Click **"+ CREATE CREDENTIALS"** → **"OAuth client ID"**.
4. If prompted to configure the consent screen:
   - User Type: **"External"** → Create
   - App name: **"AIVC"** → Save and Continue (skip optional fields)
   - Add your email as a **test user** → Save and Continue
5. Back in Credentials:
   - Application type: **"Desktop app"**
   - Name: **"AIVC CLI"**
   - Click **"Create"**
6. Copy the **Client ID** and **Client Secret** shown in the popup.

## Step 2: Enable the Google Drive API

Go to [Google Drive API](https://console.cloud.google.com/apis/library/drive.googleapis.com) 
and click **"ENABLE"**.

## Step 3: Run AIVC Sync Setup

```bash
# Standard interactive setup (opens browser)
aivc sync setup

# Headless setup (for remote servers, SSH sessions)
aivc sync setup --headless
```

The standard command will:
1. Display these instructions as a reminder.
2. Ask you to paste your **Client ID** and **Client Secret**.
3. Open your browser for Google authorization.
4. Save the OAuth token to `~/.aivc/token.json`.
5. Enable cloud sync in `~/.aivc/config.json`.

In **headless mode** (`--headless`), AIVC will not attempt to open a local browser or start a local HTTP server. Instead:
1. It prints the authorization URL. Copy it and open it in a browser on any machine.
2. After authorizing, copy the redirected URL from your browser's address bar (even if the page fails to load).
3. Paste it back into the terminal.

## Alternative: Direct Configuration Copy

If you already have AIVC authenticated on one machine, you can copy the setup directly:
1. Copy the entire `~/.aivc/` directory from the source machine to the destination machine.
2. The folder contains `config.json`, `credentials.json`, and `token.json` which fully represent the authentication and configuration state.

## Step 4: Verify

```bash
aivc sync status
```

This will show your machine ID, sync status, and any remote machines found on your Drive.

## Architecture

Once enabled, AIVC creates this structure in your Google Drive:

```
AIVC_Sync/
├── blobs/                  # Global blob pool (SHA-256 deduplication)
├── <machine-1-hostname>/
│   └── commits/            # Commit JSONs from machine 1
└── <machine-2-hostname>/
    └── commits/            # Commit JSONs from machine 2
```

- **Blobs** are shared globally, leveraging SHA-256 for cross-machine deduplication.
- **Commits** are isolated per machine to prevent write conflicts.
- Sync happens asynchronously in background threads — it never blocks your agent.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `aivc sync setup` fails with import error | Run `pip install aivc[sync]` to install Google API dependencies |
| Browser doesn't open | Copy the authorization URL from the terminal and paste it manually |
| Token expired | Run `aivc sync setup` again — it will detect the expired token and re-authenticate |
| Sync not working | Check `aivc sync status` — ensure `Sync Enabled: True` and `Auth Token: ✓ present` |
