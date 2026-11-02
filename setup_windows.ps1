# GPU Utilization Optimizer - Windows Setup Script
# This script automatically sets up the project for Windows users

Write-Host "🚀 GPU Utilization Optimizer - Windows Setup" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Function to check GPU
function Test-GPU {
    Write-Host "🔍 Checking GPU..." -ForegroundColor Yellow
    
    if (Test-Command "nvidia-smi") {
        Write-Host "✅ NVIDIA GPU detected!" -ForegroundColor Green
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
        return $true
    } else {
        Write-Host "❌ NVIDIA GPU not found or drivers not installed" -ForegroundColor Red
        Write-Host "Please install NVIDIA drivers from: https://www.nvidia.com/drivers/" -ForegroundColor Yellow
        return $false
    }
}

# Function to check Python
function Test-Python {
    Write-Host "🐍 Checking Python..." -ForegroundColor Yellow
    
    if (Test-Command "python") {
        $version = python --version 2>&1
        Write-Host "✅ Python found: $version" -ForegroundColor Green
        
        # Check if it's Python 3.8+
        $versionNumber = [regex]::Match($version, "(\d+\.\d+)").Groups[1].Value
        $majorMinor = [float]$versionNumber
        if ($majorMinor -ge 3.8) {
            return $true
        } else {
            Write-Host "❌ Python 3.8+ required. Found: $version" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "❌ Python not found" -ForegroundColor Red
        return $false
    }
}

# Function to install Python
function Install-Python {
    Write-Host "📥 Installing Python..." -ForegroundColor Yellow
    
    if (Test-Command "winget") {
        Write-Host "Using winget to install Python..." -ForegroundColor Cyan
        winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    } else {
        Write-Host "Please install Python manually:" -ForegroundColor Yellow
        Write-Host "1. Go to https://python.org/downloads/" -ForegroundColor Cyan
        Write-Host "2. Download Python 3.11+ for Windows" -ForegroundColor Cyan
        Write-Host "3. Install with 'Add Python to PATH' checked" -ForegroundColor Cyan
        Write-Host "4. Restart PowerShell and run this script again" -ForegroundColor Cyan
        return $false
    }
    return $true
}

# Function to setup virtual environment
function Setup-VirtualEnv {
    Write-Host "🔧 Setting up virtual environment..." -ForegroundColor Yellow
    
    if (Test-Path "venv") {
        Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
    } else {
        Write-Host "Creating virtual environment..." -ForegroundColor Cyan
        python -m venv venv
    }
    
    # Activate virtual environment
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & ".\venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    Write-Host "Upgrading pip..." -ForegroundColor Cyan
    python -m pip install --upgrade pip
    
    return $true
}

# Function to install dependencies
function Install-Dependencies {
    Write-Host "📦 Installing Python dependencies..." -ForegroundColor Yellow
    
    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt
        Write-Host "✅ Dependencies installed successfully!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ requirements.txt not found!" -ForegroundColor Red
        return $false
    }
}

# Function to test installation
function Test-Installation {
    Write-Host "🧪 Testing installation..." -ForegroundColor Yellow
    
    try {
        Write-Host "Testing GPU monitoring..." -ForegroundColor Cyan
        python simple_gpu_monitor.py
        Write-Host "✅ GPU monitoring test passed!" -ForegroundColor Green
        
        return $true
    } catch {
        Write-Host "❌ Installation test failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to launch dashboard
function Launch-Dashboard {
    Write-Host "🌐 Launching GPU dashboard..." -ForegroundColor Yellow
    Write-Host "Dashboard will open at: http://localhost:8050" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Yellow
    Write-Host ""
    
    python simple_dashboard.py
}

# Main setup process
try {
    # Check GPU
    $gpuOk = Test-GPU
    if (-not $gpuOk) {
        Write-Host "❌ GPU check failed. Please install NVIDIA drivers first." -ForegroundColor Red
        exit 1
    }
    
    # Check/Install Python
    $pythonOk = Test-Python
    if (-not $pythonOk) {
        $installOk = Install-Python
        if (-not $installOk) {
            Write-Host "❌ Python installation failed or cancelled." -ForegroundColor Red
            exit 1
        }
        
        # Recheck Python after installation
        Write-Host "Please restart PowerShell and run this script again." -ForegroundColor Yellow
        exit 0
    }
    
    # Setup virtual environment
    $venvOk = Setup-VirtualEnv
    if (-not $venvOk) {
        Write-Host "❌ Virtual environment setup failed." -ForegroundColor Red
        exit 1
    }
    
    # Install dependencies
    $depsOk = Install-Dependencies
    if (-not $depsOk) {
        Write-Host "❌ Dependency installation failed." -ForegroundColor Red
        exit 1
    }
    
    # Test installation
    $testOk = Test-Installation
    if (-not $testOk) {
        Write-Host "❌ Installation test failed." -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "🎉 Setup completed successfully!" -ForegroundColor Green
    Write-Host "=" * 50 -ForegroundColor Green
    
    # Ask if user wants to launch dashboard
    $launch = Read-Host "Would you like to launch the GPU dashboard now? (y/n)"
    if ($launch -eq "y" -or $launch -eq "Y" -or $launch -eq "yes") {
        Launch-Dashboard
    } else {
        Write-Host ""
        Write-Host "📋 Next steps:" -ForegroundColor Cyan
        Write-Host "1. Run: python simple_gpu_monitor.py" -ForegroundColor White
        Write-Host "2. Run: python simple_dashboard.py" -ForegroundColor White
        Write-Host "3. Open: http://localhost:8050" -ForegroundColor White
        Write-Host ""
        Write-Host "🚀 Enjoy your GPU monitoring dashboard!" -ForegroundColor Green
    }
    
} catch {
    Write-Host "❌ Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please check the error and try again, or install manually." -ForegroundColor Yellow
    exit 1
}
