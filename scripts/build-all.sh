#!/usr/bin/env bash
# Build script for all optimization variants of TDigest
# This script helps validate that all optimized build configurations work correctly

set -eo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect target architecture
ARCH=$(uname -m)

get_build_args() {
    local variant="$1"
    case "$variant" in
        "fast") echo "--release" ;;
        "modern") echo "--profile release-fast" ;;
        "size") echo "--profile release-size" ;;
        "native") echo "--release" ;;
        "dev-fast") echo "--profile dev-fast" ;;
        *) echo "Unknown variant: $variant" >&2; exit 1 ;;
    esac
}

get_rustflags() {
    local variant="$1"
    case "$variant" in
        "fast") echo "" ;;
        "modern") echo "" ;;
        "size") echo "-C opt-level=s" ;;
        "native") echo "-C target-cpu=native" ;;
        "dev-fast") echo "-C opt-level=1" ;;
        *) echo "" ;;
    esac
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} TDigest Optimized Build Script${NC}"
    echo -e "${BLUE}================================${NC}"
    echo
}

print_section() {
    echo -e "${YELLOW}→ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

build_variant() {
    local variant=$1
    local build_args=$(get_build_args "$variant")
    local rustflags=$(get_rustflags "$variant")

    print_section "Building $variant variant..."
    echo "  Args: $build_args"
    echo "  RUSTFLAGS: $rustflags"
    echo

    # Set RUSTFLAGS and build
    export RUSTFLAGS="$rustflags"

    if cargo build $build_args --locked; then
        print_success "$variant build completed"

        # Check binary size
        local profile_dir
        case $variant in
            "fast"|"native"|"dev-fast") profile_dir="release" ;;
            "modern") profile_dir="release-fast" ;;
            "size") profile_dir="release-size" ;;
        esac

        if [ "$variant" = "dev-fast" ]; then
            profile_dir="dev-fast"
        fi

        local binary_path="target/$profile_dir/libtdigest_core.rlib"
        if [ -f "$binary_path" ]; then
            local size=$(stat -c%s "$binary_path" 2>/dev/null || stat -f%z "$binary_path" 2>/dev/null || echo "unknown")
            echo "  Binary size: $size bytes"
        fi
        echo
        return 0
    else
        print_error "$variant build failed"
        return 1
    fi
}

run_tests() {
    local variant=$1
    local build_args=$(get_build_args "$variant")
    local rustflags=$(get_rustflags "$variant")

    print_section "Testing $variant variant..."

    export RUSTFLAGS="$rustflags"

    if cargo test $build_args --locked -- --nocapture; then
        print_success "$variant tests passed"
    else
        print_error "$variant tests failed"
        return 1
    fi
    echo
}

run_performance_test() {
    local variant=$1
    local build_args=$(get_build_args "$variant")
    local rustflags=$(get_rustflags "$variant")

    # Only run performance tests on release builds
    if [[ $build_args == *"--release"* ]] || [[ $build_args == *"release-fast"* ]]; then
        print_section "Performance testing $variant variant..."

        export RUSTFLAGS="$rustflags"

        if cargo test $build_args --locked test_performance_regression_large_dataset -- --nocapture; then
            print_success "$variant performance test completed"
        else
            print_error "$variant performance test failed"
        fi
        echo
    fi
}

check_prerequisites() {
    print_section "Checking prerequisites..."

    # Check Rust version
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo not found. Please install Rust."
        exit 1
    fi

    local rust_version=$(rustc --version)
    print_success "Rust: $rust_version"

    # Check CPU features
    if command -v grep &> /dev/null && [ -f /proc/cpuinfo ]; then
        local cpu_features=$(grep -o -E "(avx2|sse2|fma)" /proc/cpuinfo | sort -u | tr '\n' ' ')
        echo "  CPU features: ${cpu_features:-none detected}"
    elif command -v sysctl &> /dev/null; then
        local cpu_features=$(sysctl -n machdep.cpu.features 2>/dev/null | tr ' ' '\n' | grep -E "(AVX2|SSE2|FMA)" | tr '\n' ' ' || echo "")
        echo "  CPU features: ${cpu_features:-none detected}"
    fi

    echo
}

main() {
    print_header
    check_prerequisites

    local variants_to_build=("$@")
    if [ ${#variants_to_build[@]} -eq 0 ]; then
        variants_to_build=("fast" "modern" "size")
        echo "No variants specified. Building default variants: ${variants_to_build[*]}"
        echo
    fi

    local build_failures=0
    local test_failures=0

    # Clean previous builds
    print_section "Cleaning previous builds..."
    cargo clean
    print_success "Clean completed"
    echo

    # Build all variants
    for variant in "${variants_to_build[@]}"; do
        if ! build_variant "$variant"; then
            ((build_failures++))
        fi
    done

    # Run tests for each variant
    for variant in "${variants_to_build[@]}"; do
        if ! run_tests "$variant"; then
            ((test_failures++))
        fi
    done

    # Run performance tests
    for variant in "${variants_to_build[@]}"; do
        run_performance_test "$variant"
    done

    # Summary
    print_section "Build Summary"
    if [ $build_failures -eq 0 ]; then
        print_success "All builds completed successfully!"
    else
        print_error "$build_failures build(s) failed"
    fi

    if [ $test_failures -eq 0 ]; then
        print_success "All tests passed!"
    else
        print_error "$test_failures test suite(s) failed"
    fi

    echo
    print_section "Usage Examples:"
    echo "  Recommended: RUSTFLAGS=\"$(get_rustflags fast)\" cargo build $(get_build_args fast)"
    echo "  High Performance: RUSTFLAGS=\"$(get_rustflags modern)\" cargo build $(get_build_args modern)"
    echo "  Size Optimized: RUSTFLAGS=\"$(get_rustflags size)\" cargo build $(get_build_args size)"
    echo

    if [ $build_failures -eq 0 ] && [ $test_failures -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Show help
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    echo "TDigest Optimized Build Script"
    echo
    echo "Usage: $0 [variant1] [variant2] ..."
    echo
    echo "Available variants:"
    for variant in fast modern size native dev-fast; do
        echo "  $variant: $(get_build_args $variant)"
    done
    echo
    echo "Examples:"
    echo "  $0                    # Build default variants (fast, modern, size)"
    echo "  $0 fast              # Build only fast variant"
    echo "  $0 modern native     # Build modern and native variants"
    exit 0
fi

main "$@"