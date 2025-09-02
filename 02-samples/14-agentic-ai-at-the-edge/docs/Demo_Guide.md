# Agentic AI at the Edge - Demo Guide

## Overview

This guide provides a practical demonstration of the Agentic AI system powered by Qwen3-1.7B for edge deployment. The system features cockpit control agents and dynamic model selection capabilities.

## Prerequisites

### System Requirements
- Development Machine: macOS, Linux, or Windows with Python 3.8+
- Docker: Version 20.10+ installed and running
- Memory: Minimum 4GB RAM (8GB recommended)
- Storage: 3GB free space for Qwen3-1.7B model

### Pre-Demo Setup

```bash
# 1. Clone and setup repository
git clone <repository>
cd agentic-ai-at-the-edge

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env with configs 
.env 

# 4. Run setup script to create local docker image and container
ENABLE_API=true ./setup.sh   
```

## Demo Flow

### Part 1: Cockpit Control Demonstration (10 minutes)

**Goal**: Showcase vehicle control capabilities through natural language

#### Setup
```bash
#Execute in api mode. Endpoint url changes depending on if running locally or in EC2
python -m src.utils.audio_cli api --duration 3 --url http://localhost:8000/chat 
```

#### Demo Script - Climate Control

```
USER: Set the temperature to 72 degrees

A: Climate set to 72°F
Current cabin temperature: 68°F
Fan speed: 3/7
Mode: auto
AC: On

USER: I'm too hot

A: Climate set to 69°F
Current cabin temperature: 72°F
Fan speed: 3/7
Mode: auto
AC: On

USER: Turn on max AC

A: Maximum cooling activated
Temperature set to 60°F (minimum)
Fan speed set to maximum (7/7)
AC compressor on
Recirculation enabled
```

#### Demo Script - Window Control

```
USER: Open the driver window

A: Driver window opened to 100%
CAN signal: 0x3B2 [01 64 00 00]

USER: Close all windows

A: All windows closed
• Driver window: closed
• Passenger window: closed
• Rear left window: closed
• Rear right window: closed

USER: It's stuffy in here

A: All windows vented to 15% for fresh air
• Driver window: 15% open
• Passenger window: 15% open
• Rear windows: 15% open
```

#### Demo Script - Seat Control

```
USER: Move my seat back

A: Driver seat moved back
Position now at 40% forward

USER: Turn on seat heating

A: Driver seat heating set to level 2/3
Heating will reach target in about 2 minutes

USER: Save this as memory position 1

A: Driver seat position saved to memory 1
• Position: 40% forward, 50% height
• Recline: 30°
• Lumbar: Level 3/5
```

#### Demo Script - Lighting Control

```
USER: Turn on the headlights

A: Headlights turned on
Mode: on

USER: Set ambient lighting to blue

A: Ambient lighting set to blue at 60% intensity

USER: Turn on my reading light

A: Driver reading light turned on
```

#### Demo Script - Drive Mode Control

```
USER: Switch to sport mode

A: Sport mode engaged:
- Enhanced throttle response
- Firmer steering feel
- Tighter suspension
- Engine sound enhancement active
- Fuel economy reduced by ~15%

USER: Enable eco mode

A: Eco mode engaged:
- Optimized throttle for efficiency
- Earlier upshifts for lower RPM
- Climate control optimization
- Regenerative braking enhanced
- Fuel economy improved by ~10-15%
```

**Key Points to Highlight**:
- Direct vehicle control via natural language
- Safety interlocks enforced by Virtual ECU
- CAN bus signal generation for integration
- All controls work offline on edge device

### Part 2: Model Selection Demo (5 minutes)

**Goal**: Show dynamic model routing based on query complexity

#### Demo Script

```
USER: Turn on the AC

A: [Local model handles simple command]
Climate control: AC turned on
Temperature maintaining at 72°F

USER: Explain the thermodynamic principles of automotive climate control systems

A: [Complex query - would route to cloud model if available]
[Local model provides response based on available resources]
```

**Key Points**:
- Simple commands stay local for low latency
- Complex queries can route to cloud when available
- Fallback to local model when offline

### Part 3: Development vs Container Mode (5 minutes)

**Goal**: Show unified codebase across deployment modes

#### Development Mode
```bash
# Direct execution
python main.py

# Environment detection shows:
Deployment: Development
Memory: Unlimited
Interface: Terminal
```

#### Container Mode
```bash
# Build and run container
cd src/edge/deployment
./setup.sh

# Same main.py adapts to container environment
docker exec -it strands-edge-personal-assistant python /app/main.py
```

**Key Points**:
- Same codebase runs everywhere
- Automatic environment adaptation
- Resource constraints respected in container

## API Mode (Optional - if implemented)

### Starting API Server
```bash
ENABLE_API=true python main.py
```

### Testing API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Send command
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Set temperature to 72"}'
```