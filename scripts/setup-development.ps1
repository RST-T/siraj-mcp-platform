# SIRAJ v6.1 Development Environment Setup Script (PowerShell)
# This script sets up the complete development environment on Windows

param(
    [switch]$SkipTests,
    [switch]$Force,
    [switch]$Help
)

# Color functions
function Write-Info {
    param($Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param($Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param($Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param($Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Help message
if ($Help) {
    Write-Host "SIRAJ v6.1 Development Environment Setup Script"
    Write-Host ""
    Write-Host "Usage: .\setup-development.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -SkipTests     Skip running tests after setup"
    Write-Host "  -Force         Force reinstall of dependencies"
    Write-Host "  -Help          Show this help message"
    Write-Host ""
    Write-Host "This script will:"
    Write-Host "  1. Check and install required tools"
    Write-Host "  2. Set up Python virtual environment"
    Write-Host "  3. Install Python dependencies"
    Write-Host "  4. Install Node.js dependencies"
    Write-Host "  5. Set up database"
    Write-Host "  6. Configure environment variables"
    Write-Host "  7. Run initial tests"
    exit 0
}

# Function to check if command exists
function Test-Command {
    param($CommandName)
    try {
        Get-Command $CommandName -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Function to install Chocolatey package manager
function Install-Chocolatey {
    if (-not (Test-Command "choco")) {
        Write-Info "Installing Chocolatey package manager..."
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        
        # Refresh environment
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        Write-Success "Chocolatey installed successfully"
    } else {
        Write-Info "Chocolatey is already installed"
    }
}

# Function to check and install prerequisites
function Install-Prerequisites {
    Write-Info "Checking and installing prerequisites..."
    
    # Install Chocolatey if not present
    Install-Chocolatey
    
    # Check Python
    if (-not (Test-Command "python")) {
        Write-Info "Installing Python 3.11..."
        choco install python311 -y
        refreshenv
    } else {
        $pythonVersion = python --version
        Write-Info "Python is installed: $pythonVersion"
    }
    
    # Check Node.js
    if (-not (Test-Command "node")) {
        Write-Info "Installing Node.js..."
        choco install nodejs -y
        refreshenv
    } else {
        $nodeVersion = node --version
        Write-Info "Node.js is installed: $nodeVersion"
    }
    
    # Check Git
    if (-not (Test-Command "git")) {
        Write-Info "Installing Git..."
        choco install git -y
        refreshenv
    } else {
        $gitVersion = git --version
        Write-Info "Git is installed: $gitVersion"
    }
    
    # Check PostgreSQL (optional for local development)
    if (-not (Test-Command "psql")) {
        Write-Warning "PostgreSQL not found. Installing..."
        choco install postgresql -y
        refreshenv
        Write-Info "PostgreSQL installed. You may need to configure it manually."
    }
    
    # Check Redis (optional for local development)
    if (-not (Test-Command "redis-server")) {
        Write-Warning "Redis not found. You can install it with: choco install redis-64"
        Write-Info "For development, you can use a cloud Redis instance"
    }
    
    Write-Success "Prerequisites check completed"
}

# Function to setup Python environment
function Setup-PythonEnvironment {
    Write-Info "Setting up Python virtual environment..."
    
    # Create virtual environment
    if ($Force -or -not (Test-Path "venv")) {
        if (Test-Path "venv") {
            Remove-Item -Recurse -Force "venv"
        }
        python -m venv venv
        Write-Success "Virtual environment created"
    } else {
        Write-Info "Virtual environment already exists"
    }
    
    # Activate virtual environment
    & "venv\Scripts\Activate.ps1"
    Write-Info "Virtual environment activated"
    
    # Upgrade pip
    python -m pip install --upgrade pip
    
    # Install Python dependencies
    Write-Info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Install development dependencies
    if (Test-Path "requirements-dev.txt") {
        pip install -r requirements-dev.txt
    }
    
    Write-Success "Python environment setup completed"
}

# Function to setup Node.js environment
function Setup-NodeEnvironment {
    Write-Info "Setting up Node.js environment..."
    
    # Install root dependencies (for testing)
    if ($Force -or -not (Test-Path "node_modules")) {
        Write-Info "Installing root Node.js dependencies..."
        npm install
    } else {
        Write-Info "Root Node.js dependencies already installed"
    }
    
    # Install frontend dependencies
    Write-Info "Installing frontend dependencies..."
    Set-Location "frontend"
    if ($Force -or -not (Test-Path "node_modules")) {
        npm install
    } else {
        npm update
    }
    Set-Location ".."
    
    Write-Success "Node.js environment setup completed"
}

# Function to setup database
function Setup-Database {
    Write-Info "Setting up database..."
    
    # Check if we're using SQLite for development
    if (-not $env:DATABASE_URL) {
        Write-Info "Using SQLite for local development"
        $env:DATABASE_URL = "sqlite:///./siraj_dev.db"
    }
    
    # Run database migrations
    Write-Info "Running database migrations..."
    Set-Location "backend"
    
    # Initialize Alembic if not done
    if (-not (Test-Path "alembic")) {
        alembic init alembic
        Write-Success "Alembic initialized"
    }
    
    # Run migrations
    try {
        alembic upgrade head
        Write-Success "Database migrations completed"
    } catch {
        Write-Warning "Database migrations failed. You may need to configure your database connection."
        Write-Info "Current DATABASE_URL: $env:DATABASE_URL"
    }
    
    Set-Location ".."
}

# Function to setup environment variables
function Setup-EnvironmentVariables {
    Write-Info "Setting up environment variables..."
    
    # Create .env file if it doesn't exist
    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        Write-Success ".env file created from .env.example"
        Write-Warning "Please edit .env file with your actual configuration values"
    } else {
        Write-Info ".env file already exists"
    }
    
    # Create frontend .env.local if it doesn't exist
    if (-not (Test-Path "frontend\.env.local")) {
        if (Test-Path "frontend\.env.example") {
            Copy-Item "frontend\.env.example" "frontend\.env.local"
            Write-Success "frontend/.env.local file created"
        } else {
            Write-Warning "frontend/.env.example not found"
        }
    }
    
    # Load environment variables for current session
    if (Test-Path ".env") {
        Get-Content ".env" | ForEach-Object {
            if ($_ -match "^([^#][^=]+)=(.*)$") {
                [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
            }
        }
        Write-Info "Environment variables loaded"
    }
}

# Function to build applications
function Build-Applications {
    Write-Info "Building applications..."
    
    # Build frontend
    Write-Info "Building frontend..."
    Set-Location "frontend"
    npm run build
    Set-Location ".."
    
    Write-Success "Applications built successfully"
}

# Function to run tests
function Run-Tests {
    if ($SkipTests) {
        Write-Warning "Skipping tests"
        return
    }
    
    Write-Info "Running tests..."
    
    # Python tests
    Write-Info "Running Python tests..."
    Set-Location "backend"
    try {
        python -m pytest tests/ --cov=. --cov-report=term-missing
        Write-Success "Python tests passed"
    } catch {
        Write-Warning "Some Python tests failed"
    }
    Set-Location ".."
    
    # Frontend tests
    Write-Info "Running frontend tests..."
    Set-Location "frontend"
    try {
        npm test -- --watchAll=false
        Write-Success "Frontend tests passed"
    } catch {
        Write-Warning "Some frontend tests failed"
    }
    Set-Location ".."
    
    # E2E tests
    Write-Info "Running E2E tests..."
    try {
        npx playwright test
        Write-Success "E2E tests passed"
    } catch {
        Write-Warning "Some E2E tests failed"
    }
}

# Function to create development scripts
function Create-DevScripts {
    Write-Info "Creating development scripts..."
    
    # Create start-dev.ps1
    @"
# Development startup script
Write-Host "Starting SIRAJ v6.1 Development Environment..." -ForegroundColor Green

# Activate virtual environment
& "venv\Scripts\Activate.ps1"

# Start Redis if available
if (Get-Process "redis-server" -ErrorAction SilentlyContinue) {
    Write-Host "Redis is already running" -ForegroundColor Yellow
} else {
    Start-Process "redis-server" -WindowStyle Minimized -ErrorAction SilentlyContinue
}

# Start backend in background
Write-Host "Starting backend server..."
Start-Process powershell -ArgumentList "-Command", "cd backend; python -m uvicorn main:app --reload --port 8000" -WindowStyle Minimized

# Start MCP server in background
Write-Host "Starting MCP server..."
Start-Process powershell -ArgumentList "-Command", "cd src\server; python enhanced_mcp_server.py" -WindowStyle Minimized

# Start frontend
Write-Host "Starting frontend server..."
cd frontend
npm run dev
"@ | Out-File -FilePath "start-dev.ps1" -Encoding utf8
    
    # Create stop-dev.ps1
    @"
# Development shutdown script
Write-Host "Stopping SIRAJ v6.1 Development Environment..." -ForegroundColor Red

# Stop Node.js processes
Get-Process -Name "node" -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "Stopped Node.js processes"

# Stop Python processes (be careful not to stop system Python)
Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { `$_.Path -like "*venv*" -or `$_.CommandLine -like "*uvicorn*" -or `$_.CommandLine -like "*mcp*" } | Stop-Process -Force
Write-Host "Stopped Python development processes"

Write-Host "Development environment stopped" -ForegroundColor Green
"@ | Out-File -FilePath "stop-dev.ps1" -Encoding utf8
    
    # Create test-all.ps1
    @"
# Run all tests script
Write-Host "Running all SIRAJ v6.1 tests..." -ForegroundColor Blue

# Activate virtual environment
& "venv\Scripts\Activate.ps1"

# Run Python tests
Write-Host "Running Python tests..." -ForegroundColor Yellow
cd backend
python -m pytest tests/ --cov=. --cov-report=html
cd ..

# Run frontend tests
Write-Host "Running frontend tests..." -ForegroundColor Yellow
cd frontend
npm test -- --watchAll=false --coverage
cd ..

# Run E2E tests
Write-Host "Running E2E tests..." -ForegroundColor Yellow
npx playwright test

Write-Host "All tests completed!" -ForegroundColor Green
"@ | Out-File -FilePath "test-all.ps1" -Encoding utf8
    
    Write-Success "Development scripts created"
    Write-Info "Available scripts:"
    Write-Info "  - start-dev.ps1: Start all development servers"
    Write-Info "  - stop-dev.ps1: Stop all development servers"
    Write-Info "  - test-all.ps1: Run all tests"
}

# Function to print setup summary
function Show-SetupSummary {
    Write-Success "========================================"
    Write-Success "SIRAJ v6.1 Development Setup Complete!"
    Write-Success "========================================"
    Write-Host ""
    Write-Info "What was set up:"
    Write-Info "✅ Prerequisites (Python, Node.js, Git)"
    Write-Info "✅ Python virtual environment"
    Write-Info "✅ Python dependencies"
    Write-Info "✅ Node.js dependencies"
    Write-Info "✅ Database setup"
    Write-Info "✅ Environment variables"
    Write-Info "✅ Development scripts"
    Write-Host ""
    Write-Success "Next Steps:"
    Write-Success "1. Edit .env and frontend/.env.local with your configuration"
    Write-Success "2. Run: .\start-dev.ps1 to start development servers"
    Write-Success "3. Visit: http://localhost:3000 for frontend"
    Write-Success "4. Visit: http://localhost:8000/docs for API documentation"
    Write-Host ""
    Write-Warning "Important Notes:"
    Write-Warning "- Configure your Auth0 and Stripe accounts"
    Write-Warning "- Set up your database connection"
    Write-Warning "- Review security settings before deploying"
    Write-Host ""
    Write-Info "For help, see: README.md and DEPLOYMENT_GUIDE.md"
}

# Main setup function
function Main {
    try {
        Write-Info "Starting SIRAJ v6.1 Development Environment Setup"
        Write-Info "=================================================="
        
        Install-Prerequisites
        Setup-PythonEnvironment
        Setup-NodeEnvironment
        Setup-Database
        Setup-EnvironmentVariables
        Build-Applications
        Run-Tests
        Create-DevScripts
        
        Show-SetupSummary
        
    } catch {
        Write-Error "Setup failed: $_"
        Write-Error $_.ScriptStackTrace
        exit 1
    }
}

# Run main function
Main