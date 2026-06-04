# PHANTOM v2 — Deployment Guide

## Option 1: Local (Windows — Abhi Immediately)

```powershell
# Step 1: Project folder mein jao
cd C:\Users\mani0\Downloads\phantom-v2

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Demo run karo
cd src
python ..\demo.py
```

---

## Option 2: Docker — 3 Node Cluster (Local)

### Prerequisites
- Docker Desktop install karo: https://docker.com/products/docker-desktop
- Windows pe Docker Desktop open karo

```powershell
# Step 1: Project root mein jao
cd C:\Users\mani0\Downloads\phantom-v2

# Step 2: Docker folder mein jao
cd docker

# Step 3: 3 containers build + start karo
docker-compose up --build

# Output aayega:
# phantom-node-1 | [Raft] node-1 started as FOLLOWER
# phantom-node-2 | [Raft] node-2 started as FOLLOWER
# phantom-node-3 | [Raft] node-3 LEADER for term 1

# Step 4: Alag terminal mein — node kill karke test karo
docker stop phantom-node-3

# Step 5: Cluster phir bhi kaam kare (2/3 nodes = majority)
docker logs phantom-node-1

# Step 6: Sab band karo
docker-compose down
```

---

## Option 3: AWS EC2 Free Tier (Cloud Deployment)

### Step 1: EC2 Instance Launch karo

1. https://aws.amazon.com pe jao → Free Account banao
2. EC2 → Launch Instance
3. Settings:
   - AMI: **Ubuntu 22.04 LTS** (Free tier eligible)
   - Instance type: **t2.micro** (free tier)
   - Key pair: nayi banao → **phantom-key.pem** download karo
   - Security group → Add rules:
     - SSH: Port 22, Source: My IP
     - Custom TCP: Port 8001-8003, Source: My IP

### Step 2: SSH se Connect karo

```bash
# Windows PowerShell mein:
# Pehle key file permissions fix karo
icacls phantom-key.pem /inheritance:r /grant:r "%USERNAME%:R"

# Connect karo
ssh -i phantom-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

### Step 3: EC2 pe Setup karo

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Python + pip install
sudo apt install python3 python3-pip git docker.io docker-compose -y

# Docker permissions
sudo usermod -aG docker ubuntu
newgrp docker

# Project clone karo
git clone https://github.com/YOUR_USERNAME/phantom-v2.git
cd phantom-v2

# Dependencies install karo
pip3 install -r requirements.txt

# Demo run karo
cd src
python3 ../demo.py
```

### Step 4: Docker cluster on EC2

```bash
cd phantom-v2/docker
docker-compose up -d   # -d = background mein chale

# Status check
docker-compose ps
docker logs phantom-node-1

# Logs follow karo
docker-compose logs -f
```

### Step 5: Stop karo jab kaam ho jaaye (billing avoid)

```bash
# Containers stop
docker-compose down

# EC2 instance stop (AWS Console se)
# EC2 → Instances → Select → Instance State → Stop
# (Terminate mat karo — data delete hoga)
```

---

## Option 4: GitHub Actions CI/CD (Tests Auto-run)

`.github/workflows/test.yml` file banao:

```yaml
name: PHANTOM v2 Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: |
          cd src
          python -m pytest ../tests/ -v --tb=short

      - name: Show results
        run: echo "All tests passed!"
```

Iske baad har GitHub push pe tests automatically chalenge.
README pe green checkmark aayega — recruiters dekhte hain ye!

---

## Troubleshooting

### sortedcontainers not found
```powershell
pip install sortedcontainers numpy torch
```

### torch not found (PyTorch)
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Docker Desktop not running
- Windows taskbar mein Docker Desktop icon dhundho
- Right click → Start

### Permission denied on EC2
```bash
chmod 400 phantom-key.pem
```

### Port already in use
```powershell
# Windows mein port 8001 kaun use kar raha hai
netstat -ano | findstr :8001
taskkill /PID <PID_NUMBER> /F
```
