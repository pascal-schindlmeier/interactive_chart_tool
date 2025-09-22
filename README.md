# interactive_chart_tool
Interactive Chart Tool - Herd Behavior Simulation A real-time interactive web tool built with Python (FastAPI, WebSockets) and Chart.js, designed for live presentations. Participants can join via QR code, buy/sell “shares,” and collectively influence a live price chart, visualizing herd behavior in markets.

# Interactive Chart Tool

A real-time interactive tool to simulate **herd behavior** during presentations.  
Participants join via QR code on their phones, buy or sell shares, and immediately see the price impact on a live chart shared across all clients.

Built with **FastAPI (Python)**, **WebSockets**, and **Chart.js**, and deployable with **Docker Compose**.

---

## Current Setup (Raspberry Pi Deployment)

- **Host:** Raspberry Pi 4B, Ubuntu Server OS  
- **Networking:**  
  - [Tailscale](https://tailscale.com) for private access  
  - [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/) for public HTTPS access  
- **Interactive Chart Tool:**  
  - Runs in a Docker container on port `8000` (exposed as `8083` locally, proxied via Cloudflare Tunnel)  
  - Accessed by participants via a custom domain (e.g. `https://herd.yourdomain.com/public/index.html`)  
  - In-memory state (reset on container restart)

---

## Features

- **Real-time chart:** Updates every 250 ms for all connected clients
- **Individual portfolios:** Each participant has their own cash balance and shares
- **Collective dynamics:** Every buy/sell impacts the global price visible to all
- **Statistics:** Live metrics across all participants (connected clients, active trades, totals, max shares, min/max cash)
- **Responsive frontend:** Works on desktop and mobile browsers
- **Presenter friendly:** Simple reset by restarting the container

---

## Installation & Setup

### Prerequisites
- Python 3.11+ (for local development)  
- Docker & Docker Compose (for deployment)  
- Cloudflare Tunnel or Tailscale (optional, for remote/public access)

### Local Development

```bash
# Clone this repo
git clone https://github.com/<your-username>/interactive_chart_tool.git
cd interactive_chart_tool

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install fastapi "uvicorn[standard]" pydantic

# Run the server
uvicorn app.ict_backend:app --reload --host 0.0.0.0 --port 8000

```
### File structure
interactive_chart_tool/
├── app/
│   ├── ict_backend.py       # FastAPI backend (WebSocket + REST)
│   └── static/
│       └── index.html       # Frontend (Chart.js + WebSocket client)
├── Dockerfile
└── docker-compose.yml       # (if included in your selfhosted stack)

