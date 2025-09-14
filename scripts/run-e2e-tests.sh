#!/bin/bash

# SIRAJ v6.1 End-to-End Testing Script
# This script runs comprehensive E2E tests using Playwright

set -e

# Configuration
TEST_TIMEOUT=300000  # 5 minutes
RETRY_ATTEMPTS=2
PARALLEL_WORKERS=4

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is required but not installed"
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        print_error "npm is required but not installed"
        exit 1
    fi
    
    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python is required but not installed"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to setup test environment
setup_test_environment() {
    print_status "Setting up test environment..."
    
    # Install Playwright if not installed
    if ! npx playwright --version &> /dev/null; then
        print_status "Installing Playwright..."
        npx playwright install
    fi
    
    # Install dependencies
    if [ ! -d "node_modules" ]; then
        print_status "Installing Node.js dependencies..."
        npm install
    fi
    
    # Setup Python environment for backend
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate || . venv/Scripts/activate
    
    # Install Python dependencies
    if [ -f "requirements.txt" ]; then
        print_status "Installing Python dependencies..."
        pip install -r requirements.txt
    fi
    
    # Setup environment variables for testing
    export NODE_ENV=test
    export ENVIRONMENT=test
    export DATABASE_URL="sqlite:///test.db"
    export REDIS_URL="redis://localhost:6379/1"
    export STRIPE_TEST_MODE=true
    export MOCK_AUTH=true
    
    print_success "Test environment setup completed"
}

# Function to start backend services
start_backend_services() {
    print_status "Starting backend services..."
    
    # Start backend API
    cd backend
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    cd ..
    
    # Start MCP server
    cd src/server
    python enhanced_mcp_server.py &
    MCP_PID=$!
    cd ../..
    
    # Wait for services to be ready
    print_status "Waiting for backend services to be ready..."
    sleep 10
    
    # Check if services are running
    if ! curl -s http://localhost:8000/health > /dev/null; then
        print_error "Backend API failed to start"
        cleanup_services
        exit 1
    fi
    
    print_success "Backend services started"
}

# Function to start frontend service
start_frontend_service() {
    print_status "Starting frontend service..."
    
    cd frontend
    npm run build
    npm run start &
    FRONTEND_PID=$!
    cd ..
    
    # Wait for frontend to be ready
    print_status "Waiting for frontend to be ready..."
    sleep 15
    
    # Check if frontend is running
    if ! curl -s http://localhost:3000 > /dev/null; then
        print_error "Frontend failed to start"
        cleanup_services
        exit 1
    fi
    
    print_success "Frontend service started"
}

# Function to run specific test suites
run_test_suite() {
    local suite_name=$1
    local test_file=$2
    
    print_status "Running $suite_name tests..."
    
    npx playwright test "$test_file" \
        --timeout=$TEST_TIMEOUT \
        --retries=$RETRY_ATTEMPTS \
        --workers=$PARALLEL_WORKERS \
        --reporter=html,json,junit || {
        print_error "$suite_name tests failed"
        return 1
    }
    
    print_success "$suite_name tests passed"
    return 0
}

# Function to run all E2E tests
run_all_tests() {
    print_status "Running complete E2E test suite..."
    
    local failed_tests=()
    
    # Run deployment validation tests
    if ! run_test_suite "Deployment Validation" "tests/e2e/deployment-validation.spec.ts"; then
        failed_tests+=("Deployment Validation")
    fi
    
    # Run authentication flow tests
    if ! run_test_suite "Authentication Flow" "tests/e2e/authentication-flow.spec.ts"; then
        failed_tests+=("Authentication Flow")
    fi
    
    # Run API endpoint tests
    if ! run_test_suite "API Endpoints" "tests/e2e/api-endpoints.spec.ts"; then
        failed_tests+=("API Endpoints")
    fi
    
    # Run payment processing tests
    if ! run_test_suite "Payment Processing" "tests/e2e/payment-processing.spec.ts"; then
        failed_tests+=("Payment Processing")
    fi
    
    # Run MCP server integration tests
    if ! run_test_suite "MCP Server Integration" "tests/e2e/mcp-server-integration.spec.ts"; then
        failed_tests+=("MCP Server Integration")
    fi
    
    # Report results
    if [ ${#failed_tests[@]} -eq 0 ]; then
        print_success "All E2E test suites passed!"
        return 0
    else
        print_error "The following test suites failed:"
        printf '%s\n' "${failed_tests[@]}"
        return 1
    fi
}

# Function to generate test report
generate_test_report() {
    print_status "Generating test report..."
    
    # Create test report directory
    mkdir -p test-reports
    
    # Generate HTML report
    if [ -f "playwright-report/index.html" ]; then
        cp -r playwright-report test-reports/html-report
        print_success "HTML report generated: test-reports/html-report/index.html"
    fi
    
    # Generate summary report
    cat > test-reports/summary.md << EOF
# SIRAJ v6.1 E2E Test Report

**Test Date:** $(date)
**Environment:** Test
**Base URL:** http://localhost:3000
**API URL:** http://localhost:8000

## Test Results

$(if [ -f "test-results/results.json" ]; then
    node -e "
    const fs = require('fs');
    const results = JSON.parse(fs.readFileSync('test-results/results.json', 'utf8'));
    const total = results.suites.reduce((acc, suite) => acc + suite.specs.length, 0);
    const passed = results.suites.reduce((acc, suite) => 
        acc + suite.specs.filter(spec => spec.tests.every(test => test.outcome === 'passed')).length, 0);
    const failed = total - passed;
    
    console.log(\`### Summary\`);
    console.log(\`- **Total Tests:** \${total}\`);
    console.log(\`- **Passed:** \${passed}\`);
    console.log(\`- **Failed:** \${failed}\`);
    console.log(\`- **Success Rate:** \${Math.round((passed / total) * 100)}%\`);
    console.log();
    
    results.suites.forEach(suite => {
        console.log(\`### \${suite.title}\`);
        suite.specs.forEach(spec => {
            const status = spec.tests.every(test => test.outcome === 'passed') ? '✅' : '❌';
            console.log(\`- \${status} \${spec.title}\`);
        });
        console.log();
    });
    "
else
    echo "### Summary"
    echo "Test results not available"
fi)

## Test Coverage

$(if [ -f "coverage/coverage-summary.json" ]; then
    echo "Code coverage report available in test-reports/coverage/"
else
    echo "Code coverage not available"
fi)

## Performance Metrics

$(if [ -f "test-results/performance.json" ]; then
    echo "Performance metrics available in test-results/performance.json"
else
    echo "Performance metrics not collected"
fi)

## Next Steps

1. Review failed tests (if any)
2. Check performance bottlenecks
3. Update test cases as needed
4. Run tests on different browsers/devices

EOF

    print_success "Test report generated: test-reports/summary.md"
}

# Function to cleanup services
cleanup_services() {
    print_status "Cleaning up services..."
    
    # Kill backend processes
    if [ -n "${BACKEND_PID:-}" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    if [ -n "${MCP_PID:-}" ]; then
        kill $MCP_PID 2>/dev/null || true
    fi
    
    if [ -n "${FRONTEND_PID:-}" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    # Kill any remaining processes
    pkill -f "uvicorn main:app" 2>/dev/null || true
    pkill -f "enhanced_mcp_server" 2>/dev/null || true
    pkill -f "next" 2>/dev/null || true
    
    # Clean up test database
    rm -f backend/test.db 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Function to show usage
show_usage() {
    echo "SIRAJ v6.1 E2E Testing Script"
    echo ""
    echo "Usage: $0 [options] [test-suite]"
    echo ""
    echo "Options:"
    echo "  --help, -h              Show this help message"
    echo "  --headless              Run tests in headless mode"
    echo "  --headed                Run tests in headed mode (default)"
    echo "  --timeout SECONDS       Set test timeout (default: 300)"
    echo "  --retries NUM           Set retry attempts (default: 2)"
    echo "  --workers NUM           Set parallel workers (default: 4)"
    echo "  --browser BROWSER       Run tests on specific browser (chromium, firefox, webkit)"
    echo "  --debug                 Enable debug mode"
    echo "  --report-only           Only generate reports from existing results"
    echo ""
    echo "Test Suites:"
    echo "  deployment              Run deployment validation tests"
    echo "  auth                    Run authentication flow tests"
    echo "  api                     Run API endpoint tests"
    echo "  payment                 Run payment processing tests"
    echo "  mcp                     Run MCP server integration tests"
    echo "  all                     Run all test suites (default)"
    echo ""
    echo "Examples:"
    echo "  $0                      Run all tests with default settings"
    echo "  $0 auth                 Run only authentication tests"
    echo "  $0 --headless --workers 2 all"
    echo "  $0 --browser firefox deployment"
}

# Parse command line arguments
HEADLESS=false
DEBUG=false
REPORT_ONLY=false
BROWSER=""
TEST_SUITE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --headed)
            HEADLESS=false
            shift
            ;;
        --timeout)
            TEST_TIMEOUT="$2"
            shift 2
            ;;
        --retries)
            RETRY_ATTEMPTS="$2"
            shift 2
            ;;
        --workers)
            PARALLEL_WORKERS="$2"
            shift 2
            ;;
        --browser)
            BROWSER="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --report-only)
            REPORT_ONLY=true
            shift
            ;;
        deployment|auth|api|payment|mcp|all)
            TEST_SUITE="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set Playwright options based on arguments
PLAYWRIGHT_OPTS=""
if [ "$HEADLESS" = true ]; then
    PLAYWRIGHT_OPTS="$PLAYWRIGHT_OPTS --config playwright.config.ts"
fi

if [ -n "$BROWSER" ]; then
    PLAYWRIGHT_OPTS="$PLAYWRIGHT_OPTS --project $BROWSER"
fi

if [ "$DEBUG" = true ]; then
    PLAYWRIGHT_OPTS="$PLAYWRIGHT_OPTS --debug"
fi

# Main execution
main() {
    print_status "Starting SIRAJ v6.1 E2E Test Runner"
    print_status "===================================="
    
    # Trap cleanup on exit
    trap cleanup_services EXIT
    
    if [ "$REPORT_ONLY" = true ]; then
        generate_test_report
        exit 0
    fi
    
    check_prerequisites
    setup_test_environment
    start_backend_services
    start_frontend_service
    
    case $TEST_SUITE in
        deployment)
            run_test_suite "Deployment Validation" "tests/e2e/deployment-validation.spec.ts"
            ;;
        auth)
            run_test_suite "Authentication Flow" "tests/e2e/authentication-flow.spec.ts"
            ;;
        api)
            run_test_suite "API Endpoints" "tests/e2e/api-endpoints.spec.ts"
            ;;
        payment)
            run_test_suite "Payment Processing" "tests/e2e/payment-processing.spec.ts"
            ;;
        mcp)
            run_test_suite "MCP Server Integration" "tests/e2e/mcp-server-integration.spec.ts"
            ;;
        all)
            run_all_tests
            ;;
    esac
    
    generate_test_report
    
    print_success "===================================="
    print_success "E2E Testing Completed!"
    print_success "===================================="
    print_success "Test Report: test-reports/summary.md"
    print_success "HTML Report: test-reports/html-report/index.html"
}

# Run main function
main "$@"