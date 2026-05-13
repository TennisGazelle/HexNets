print_line() {
    echo "===> $1"
}

# Override for shorter local runs, e.g. E2E_EPOCHS=20 ./e2e_test.sh
E2E_EPOCHS="${E2E_EPOCHS:-10}"

e2e_train() {
    local msg="$1"
    shift
    echo "================================================"
    print_line "$msg"
    "$@"
    echo "================================================"
}

# GLOBAL VARS
DEBUG=0
export MPLBACKEND=Agg # to turn off all interactive plots

### install the dependencies
if [ ! -d ".venv" ]; then
    make install
fi
source .venv/bin/activate