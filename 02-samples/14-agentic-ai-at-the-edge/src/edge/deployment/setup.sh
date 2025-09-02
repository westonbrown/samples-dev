#!/bin/bash
# setup.sh - Edge deployment script
# Handles complete deployment of the Personal Assistant

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$SCRIPT_DIR/models"
DATA_DIR="$SCRIPT_DIR/data"
LOGS_DIR="$SCRIPT_DIR/logs"

# Load environment variables from .env if it exists (checking multiple locations)
ENV_FILE=""
if [ -f "$PROJECT_ROOT/../../../.env" ]; then
    ENV_FILE="$PROJECT_ROOT/../../../.env"
    echo "📋 Loading environment from project root .env file..."
elif [ -f "$SCRIPT_DIR/.env" ]; then
    ENV_FILE="$SCRIPT_DIR/.env"
    echo "📋 Loading environment from deployment .env file..."
fi

# Source the .env file if found
if [ -n "$ENV_FILE" ]; then
    set -a  # Mark all new variables for export
    source "$ENV_FILE"
    set +a  # Turn off auto-export
    echo "✅ Environment variables loaded from $ENV_FILE"
else
    echo "⚠️ No .env file found, using defaults"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${CYAN}[SETUP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to check system requirements
check_requirements() {
    print_header "CHECKING SYSTEM REQUIREMENTS"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker is installed: $(docker --version)"
    
    # Docker is sufficient for single container deployment
    print_success "Docker ready for single container deployment"
    
    # Check available disk space (need at least 5GB for Qwen3 1.7B + Whisper)
    available_gb=$(df "$SCRIPT_DIR" | awk 'NR==2 {print int($4/1024/1024)}')
    if [ "$available_gb" -lt 5 ]; then
        print_warning "Available disk space is ${available_gb}GB. Recommended: 5GB+"
        echo "The Qwen3 1.7B + Whisper models require ~2.2GB of space."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Available disk space: ${available_gb}GB"
    fi
    
    # Check if audio devices are available (for voice input)
    if [ -d "/dev/snd" ]; then
        print_success "Audio devices found - voice input will be available"
    else
        print_warning "No audio devices found - voice input may not work"
    fi
}

# Function to create necessary directories
setup_directories() {
    print_header "SETTING UP DIRECTORIES"
    
    # Create directories with proper permissions
    mkdir -p "$MODELS_DIR" "$DATA_DIR" "$LOGS_DIR"
    
    # Set permissions (container runs as user 1001)
    if [ "$(id -u)" -eq 0 ]; then
        chown -R 1001:1001 "$MODELS_DIR" "$DATA_DIR" "$LOGS_DIR"
    fi
    
    # Validate directories are writable
    for dir in "$MODELS_DIR" "$DATA_DIR" "$LOGS_DIR"; do
        if [ ! -w "$dir" ]; then
            print_warning "Directory not writable: $dir"
            print_status "Trying to fix permissions..."
            chmod 755 "$dir" 2>/dev/null || print_error "Failed to fix permissions for $dir"
        fi
    done
    
    print_success "Created directories:"
    print_success "  Models: $MODELS_DIR"
    print_success "  Data:   $DATA_DIR"  
    print_success "  Logs:   $LOGS_DIR"
}

# Function to build the container
build_container() {
    print_header "BUILDING EDGE CONTAINER"
    
    cd "$SCRIPT_DIR"
    
    print_status "Building personal assistant edge container..."
    print_status "This may take 10-15 minutes for the first build..."
    
    # Build with Docker from project root as context
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
    
    # Detect platform for build
    PLATFORM=""
    if [[ "$(uname -m)" == "arm64" ]] || [[ "$(uname -m)" == "aarch64" ]]; then
        PLATFORM="--platform linux/arm64"
        print_status "Building for ARM64 architecture"
    else
        PLATFORM="--platform linux/amd64"
        print_status "Building for AMD64 architecture"
    fi
    
    docker build $PLATFORM -f "$PROJECT_ROOT/src/edge/deployment/Dockerfile.edge" -t personal-assistant:edge "$PROJECT_ROOT"
    
    if [ $? -eq 0 ]; then
        print_success "Container built successfully!"
    else
        print_error "Container build failed"
        exit 1
    fi
}

# Function to start the assistant
start_assistant() {
    print_header "STARTING PERSONAL ASSISTANT"
    
    cd "$SCRIPT_DIR"
    
    print_status "Starting edge personal assistant..."
    print_status "First startup will download models (~2.2GB for Qwen3 + Whisper) - this may take 10-15 minutes"
    print_status "Subsequent startups will be much faster (<10 seconds)"
    
    # Determine API mode
    API_MODE="${ENABLE_API:-false}"
    if [ "$API_MODE" = "true" ]; then
        echo "   🌐 API mode enabled"
    else
        echo "   💬 Interactive mode enabled"
    fi
    
    # Detect OS for audio device handling
    AUDIO_MOUNT=""
    AUDIO_DEVICE=""
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux has /dev/snd
        if [ -d "/dev/snd" ]; then
            AUDIO_MOUNT="-v /dev/snd:/dev/snd"
            AUDIO_DEVICE="--device /dev/snd"
            print_status "Linux audio devices detected"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use audio_exchange directory for file-based audio
        mkdir -p "$SCRIPT_DIR/audio_exchange"
        # Ensure directory is writable
        chmod 755 "$SCRIPT_DIR/audio_exchange" 2>/dev/null || true
        AUDIO_MOUNT="-v $SCRIPT_DIR/audio_exchange:/app/audio_exchange"
        print_warning "macOS detected - using file-based audio exchange"
        print_status "Audio files will be exchanged via: $SCRIPT_DIR/audio_exchange"
        
        # Validate audio exchange directory
        if [ ! -w "$SCRIPT_DIR/audio_exchange" ]; then
            print_warning "Audio exchange directory not writable: $SCRIPT_DIR/audio_exchange"
            print_status "Voice input may not work. Try: chmod 755 $SCRIPT_DIR/audio_exchange"
        fi
    fi
    
    # Detect platform for container run
    RUN_PLATFORM=""
    if [[ "$(uname -m)" == "arm64" ]] || [[ "$(uname -m)" == "aarch64" ]]; then
        RUN_PLATFORM="--platform linux/arm64"
    else
        RUN_PLATFORM="--platform linux/amd64"
    fi
    
    # Ensure critical variables have defaults
    QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/app/models/Qwen3-1.7B-Q4_K_M.gguf}"
    WHISPER_MODEL_PATH="${WHISPER_MODEL_PATH:-/app/models/ggml-base.bin}"
    TARGET_MODEL="${TARGET_MODEL:-qwen3-1.7b-q4}"
    CHAT_TEMPLATE="${CHAT_TEMPLATE:-chatml}"
    LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-2048}"
    LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-2048}"
    GGML_NTHREADS="${GGML_NTHREADS:-8}"
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    
    # Start container with direct docker run
    # Note: Container only runs llama-server on port 8080
    # Port 8000 is NOT needed (main.py runs on host for API mode)
    docker run -d \
        --name strands-edge-personal-assistant \
        --restart unless-stopped \
        $RUN_PLATFORM \
        -p 8080:8080 \
        -v "$MODELS_DIR:/app/models" \
        -v "$DATA_DIR:/app/data" \
        -v "$LOGS_DIR:/app/logs" \
        $AUDIO_MOUNT \
        $AUDIO_DEVICE \
        -e MODEL_PATH="${QWEN_MODEL_PATH}" \
        -e WHISPER_MODEL_PATH="${WHISPER_MODEL_PATH}" \
        -e TARGET_MODEL="${TARGET_MODEL}" \
        -e CHAT_TEMPLATE="${CHAT_TEMPLATE}" \
        -e LOG_LEVEL="${LOG_LEVEL:-INFO}" \
        -e ENABLE_API=$API_MODE \
        -e AWS_BEARER_TOKEN_BEDROCK="${AWS_BEARER_TOKEN_BEDROCK}" \
        -e AWS_REGION="${AWS_REGION:-us-east-1}" \
        -e FFMPEG_PATH="${FFMPEG_PATH:-ffmpeg}" \
        -e LLAMACPP_URL="${LLAMACPP_URL:-http://localhost:8080}" \
        -e CONTEXT_WINDOW="${CONTEXT_WINDOW:-20}" \
        -e MAX_TOKENS="${MAX_TOKENS:-1024}" \
        -e LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE}" \
        -e LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE}" \
        -e LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-512}" \
        -e LLAMA_N_PARALLEL="${LLAMA_N_PARALLEL:-1}" \
        -e GGML_NTHREADS="${GGML_NTHREADS}" \
        -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        -e LLAMA_THREADS_HTTP="${LLAMA_THREADS_HTTP:-8}" \
        --memory=4g \
        --cpus=8.0 \
        personal-assistant:edge
    
    print_status "Waiting for assistant to be ready..."
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker inspect strands-edge-personal-assistant --format='{{.State.Health.Status}}' 2>/dev/null | grep -q "healthy"; then
            print_success "Personal Assistant is ready!"
            break
        fi
        
        if [ $((attempt % 10)) -eq 0 ]; then
            print_status "Still waiting... (attempt $attempt/$max_attempts)"
            print_status "Check logs: docker logs strands-edge-personal-assistant"
        fi
        
        sleep 30
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        print_error "Assistant failed to start within expected time"
        exit 1
    fi
}

# Function to show usage information
show_usage() {
    print_header "PERSONAL ASSISTANT IS READY!"
    
    echo ""
    echo -e "${GREEN}🤖 Your Edge Personal Assistant is now running!${NC}"
    echo ""
    echo -e "${CYAN}📋 Quick Commands:${NC}"
    echo "  Container provides:     llama-server on port 8080"
    echo ""
    if [ "$API_MODE" = "true" ]; then
    echo "  For API Mode:"
    echo "  1. Open new terminal"
    echo "  2. Run: ENABLE_API=true python main.py"
    echo "  3. API Health Check:    curl http://localhost:8000/health"
    echo "  4. API Documentation:   http://localhost:8000/docs"
else
    echo "  For Interactive Mode:"
    echo "  1. Open new terminal"
    echo "  2. Run: python main.py"
fi
    echo "  View logs:              docker logs -f strands-edge-personal-assistant"
    echo "  Check status:           docker ps"
    echo "  Stop assistant:         docker stop strands-edge-personal-assistant"
    echo "  Restart assistant:      docker restart strands-edge-personal-assistant"
    echo ""
    echo -e "${CYAN}🎤 Voice Features:${NC}"
    echo "  • Say 'voice' to activate speech input (10 seconds)"
    echo "  • Supports natural conversation with follow-ups"
    echo "  • Powered by Qwen3 1.7B + FFmpeg Whisper"
    echo ""
    echo -e "${CYAN}🛠️  Capabilities:${NC}"
    echo "  • 📅 Calendar management and scheduling"
    echo "  • 💻 Code assistance and programming help"
    echo "  • 🔍 Search and research assistance"
    echo "  • 💬 Natural conversation with memory"
    echo ""
    echo -e "${CYAN}📊 Resource Usage:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" strands-edge-personal-assistant 2>/dev/null || echo "  Run 'docker stats' to monitor resource usage"
    echo ""
    echo -e "${CYAN}🌐 Access:${NC}"
    echo "  • Direct connection: Connect using the docker exec command above"
    echo "  • Server endpoint: http://localhost:8080 (llama.cpp server)"
    echo ""
    echo -e "${YELLOW}💡 Tips:${NC}"
    echo "  • First run downloads models (~800MB) - be patient!"
    echo "  • Voice input requires microphone access"
    echo "  • Type 'exit' in the assistant to quit gracefully" 
    echo "  • Container auto-restarts unless manually stopped"
    echo ""
}

# Function to connect to the assistant
connect_assistant() {
    print_status "Run main.py locally:"
    echo "  Interactive: python main.py"
    echo "  API mode: ENABLE_API=true python main.py"
}

# Main setup function
main() {
    print_header "PERSONAL ASSISTANT SETUP"
    echo "This script will set up your voice-enabled personal assistant"
    echo "with Qwen3 1.7B (optimized for edge) in a single container deployment."
    echo ""
    
    # Parse arguments
    local action="${1:-setup}"
    
    case "$action" in
        "setup"|"install"|"deploy")
            check_requirements
            setup_directories
            build_container
            start_assistant
            show_usage
            
            # Ask if user wants to connect immediately
            echo ""
            echo -e "${CYAN}Container is providing llama-server on port 8080${NC}"
            echo ""
            if [ "$API_MODE" = "true" ]; then
                echo -e "${CYAN}To run the API server:${NC}"
                echo "  1. Open a new terminal"
                echo "  2. Navigate to: $(dirname $SCRIPT_DIR)"
                echo "  3. Run: ENABLE_API=true python main.py"
                echo ""
                echo "The API will be available at http://localhost:8000"
            else
                echo -e "${CYAN}To run interactive mode:${NC}"
                echo "  1. Open a new terminal"
                echo "  2. Navigate to: $(dirname $SCRIPT_DIR)"
                echo "  3. Run: python main.py"
            fi
            ;;
            
        "connect"|"start"|"run")
            print_status "Connecting to existing personal assistant..."
            connect_assistant
            ;;
            
        "status"|"info")
            print_header "PERSONAL ASSISTANT STATUS"
            docker ps --filter name=strands-edge-personal-assistant
            echo ""
            docker stats --no-stream strands-edge-personal-assistant 2>/dev/null || echo "Assistant is not running"
            ;;
            
        "logs")
            print_status "Showing personal assistant logs..."
            docker logs -f strands-edge-personal-assistant
            ;;
            
        "stop")
            print_status "Stopping personal assistant..."
            docker stop strands-edge-personal-assistant
            docker rm strands-edge-personal-assistant
            print_success "Personal assistant stopped"
            ;;
            
        "restart")
            print_status "Restarting personal assistant..."
            docker restart strands-edge-personal-assistant
            print_success "Personal assistant restarted"
            ;;
            
        "rebuild")
            print_status "Rebuilding and restarting personal assistant..."
            docker stop strands-edge-personal-assistant 2>/dev/null || true
            docker rm strands-edge-personal-assistant 2>/dev/null || true
            PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
            
            # Detect platform for rebuild
            PLATFORM=""
            if [[ "$(uname -m)" == "arm64" ]] || [[ "$(uname -m)" == "aarch64" ]]; then
                PLATFORM="--platform linux/arm64"
            else
                PLATFORM="--platform linux/amd64"
            fi
            
            docker build $PLATFORM -f "$PROJECT_ROOT/src/edge/deployment/Dockerfile.edge" -t personal-assistant:edge "$PROJECT_ROOT" --no-cache
            start_assistant
            print_success "Personal assistant rebuilt and restarted"
            ;;
            
        "help"|"--help"|"-h")
            echo "Personal Assistant Setup"
            echo ""
            echo "Usage: $0 [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  setup, install, deploy  Complete setup and deployment (default)"
            echo "  connect, start, run     Connect to running assistant"
            echo "  status, info           Show assistant status"
            echo "  logs                   Show assistant logs"
            echo "  stop                   Stop the assistant"
            echo "  restart                Restart the assistant"
            echo "  rebuild                Rebuild and restart the assistant"
            echo "  help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                     # Complete setup and deployment"
            echo "  $0 connect             # Connect to running assistant"
            echo "  $0 logs                # Monitor assistant logs"
            echo "  $0 stop                # Stop the assistant"
            ;;
            
        *)
            print_error "Unknown command: $action"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"