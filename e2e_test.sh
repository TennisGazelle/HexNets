#!/bin/bash
set -euo pipefail

print_line() {
    echo "===> $1"
}

# Override for shorter local runs, e.g. E2E_EPOCHS=20 ./e2e_test.sh
E2E_EPOCHS="${E2E_EPOCHS:-100}"

e2e_train() {
    local msg="$1"
    shift
    echo "================================================"
    print_line "$msg"
    "$@"
    echo "================================================"
}

DEBUG=0

### install the dependencies
if [ ! -d ".venv" ]; then
    make install
fi
source .venv/bin/activate

# 1. Creating all the reference graphs
print_line '1. Creating all the reference graphs...'
if [ ! -d "reference" ]; then
    hexnet ref --all
fi
# ensure the reference graphs are created
if [ ! -d "reference" ]; then
    print_line "1. Reference graphs not found"
    exit 1
else
    expected_files_templates=(
        "reference/hexnet_n{n}_r{r}_structure.png"
        "reference/hexnet_n{n}_r{r}_Activation_Structure.png"
        "reference/hexnet_n{n}_r{r}_Weight_Matrix.png"
        "reference/hexnet_n{n}_multi_activation.png"
    )
    for file in "${expected_files_templates[@]}"; do
        for n in {2..8}; do
            for r in {0..5}; do
                file_path=$(echo "$file" | sed "s/{n}/${n}/g" | sed "s/{r}/${r}/g")
                if [ $DEBUG -eq 1 ]; then
                    print_line "Checking $file_path"
                fi
                if [ ! -f "$file_path" ]; then
                    print_line "1. Reference graph $file_path not found"
                    exit 1
                fi
            done
        done
    done
    print_line "1. Found all reference graphs"
fi

# 2. Smoke: hex (new + resume) and MLP under runs/e2etest-smoke/
print_line "2. Smoke: hex net train (n=3, r=0) and resume (r=1)..."
rm -rf runs/e2etest-smoke
e2e_train "2a. Hex train new run" \
    hexnet train -m hex -n 3 -r 0 -e "${E2E_EPOCHS}" -t identity \
    -rn e2etest-smoke/hex-n3-r0 \
    --run-tags e2e,smoke \
    --run-note "e2e smoke hex new"
e2e_train "2b. Hex resume same run dir with r=1" \
    hexnet train -m hex -n 3 -r 1 -e "${E2E_EPOCHS}" -t identity \
    -rd runs/e2etest-smoke/hex-n3-r0

exit 0; # TODO: remove this

print_line "3. Smoke: MLP train (n=2)..."
e2e_train "3. MLP train" \
    hexnet train -m mlp -n 2 -e "${E2E_EPOCHS}" -t identity \
    -rn e2etest-smoke/mlp-n2 \
    --run-tags e2e,smoke \
    --run-note "e2e smoke mlp"

# 4–9. Benchmark families A–F (non-exhaustive; see docs/math/benchmark-families.md)
print_line "Benchmark families A-F (runs under runs/e2etest-fam*/ )..."
rm -rf runs/e2etest-famA runs/e2etest-famB runs/e2etest-famC runs/e2etest-famD runs/e2etest-famE runs/e2etest-famF

FAM_TAG_BASE="e2e,benchmark"

# --- Family A: linear structure recovery ---
print_line "4. Family A — linear structure recovery (hex + sample MLP)"
FAM_A_TASKS=(
    identity
    linear_scale
    diagonal_scale
    diagonal_linear
    full_linear
    low_rank_linear
    affine
    orthogonal_rotation
)
for task in "${FAM_A_TASKS[@]}"; do
    e2e_train "4. Family A hex: ${task}" \
        hexnet train -m hex -n 4 -r 0 -e "${E2E_EPOCHS}" -t "${task}" \
        -a linear -l mean_squared_error -lr constant \
        -rn "e2etest-famA/hex-${task}-n4" \
        --run-tags "${FAM_TAG_BASE},famA" \
        --run-note "e2e benchmark family A (linear)"
done
for task in identity orthogonal_rotation; do
    e2e_train "4. Family A MLP: ${task}" \
        hexnet train -m mlp -n 4 -e "${E2E_EPOCHS}" -t "${task}" \
        -a linear -l mean_squared_error -lr constant \
        -rn "e2etest-famA/mlp-${task}-n4" \
        --run-tags "${FAM_TAG_BASE},famA" \
        --run-note "e2e benchmark family A (linear) MLP"
done

# --- Family B: smooth nonlinear (small activation × loss cross) ---
print_line "5. Family B — smooth nonlinear (sample grid)"
e2e_train "5. Family B: elementwise_power linear + MSE" \
    hexnet train -m hex -n 4 -r 0 -e "${E2E_EPOCHS}" -t elementwise_power \
    -a linear -l mean_squared_error -lr constant \
    -rn e2etest-famB/hex-elementwise_power-linear-mse \
    --run-tags "${FAM_TAG_BASE},famB" \
    --run-note "e2e benchmark family B"
e2e_train "5. Family B: sine sigmoid + log_cosh" \
    hexnet train -m hex -n 4 -r 0 -e "${E2E_EPOCHS}" -t sine \
    -a sigmoid -l log_cosh -lr constant \
    -rn e2etest-famB/hex-sine-sigmoid-logcosh \
    --run-tags "${FAM_TAG_BASE},famB" \
    --run-note "e2e benchmark family B"
e2e_train "5. Family B: affine leaky_relu + huber" \
    hexnet train -m hex -n 4 -r 0 -e "${E2E_EPOCHS}" -t affine \
    -a leaky_relu -l huber -lr exponential_decay \
    -rn e2etest-famB/hex-affine-leaky-huber-expdecay \
    --run-tags "${FAM_TAG_BASE},famB" \
    --run-note "e2e benchmark family B"

# --- Family C: projections ---
print_line "6. Family C — projection / constraint tasks"
FAM_C_TASKS=(
    unit_sphere_projection
    l2_ball_projection
    non_negative_projection
    simplex_projection
)
for task in "${FAM_C_TASKS[@]}"; do
    e2e_train "6. Family C: ${task}" \
        hexnet train -m hex -n 4 -r 0 -e "${E2E_EPOCHS}" -t "${task}" \
        -a linear -l huber -lr constant \
        -rn "e2etest-famC/hex-${task}-n4" \
        --run-tags "${FAM_TAG_BASE},famC" \
        --run-note "e2e benchmark family C (projection)"
done

# --- Family D: permutation / sort ---
print_line "7. Family D — order / discontinuity"
e2e_train "7. Family D: fixed_permutation" \
    hexnet train -m hex -n 4 -r 0 -e "${E2E_EPOCHS}" -t fixed_permutation \
    -a leaky_relu -l huber -lr constant \
    -rn e2etest-famD/hex-fixed_permutation-n4 \
    --run-tags "${FAM_TAG_BASE},famD" \
    --run-note "e2e benchmark family D"
e2e_train "7. Family D: sort" \
    hexnet train -m hex -n 4 -r 0 -e "${E2E_EPOCHS}" -t sort \
    -a leaky_relu -l huber -lr constant \
    -rn e2etest-famD/hex-sort-n4 \
    --run-tags "${FAM_TAG_BASE},famD" \
    --run-note "e2e benchmark family D"

# --- Family E: noise ---
print_line "8. Family E — noise robustness (sample)"
e2e_train "8. Family E: full_linear with dataset noise (both)" \
    hexnet train -m hex -n 4 -r 0 -e "${E2E_EPOCHS}" -t full_linear \
    -a linear -l huber -lr constant \
    --dataset-noise both --dataset-noise-sigma 0.05 \
    -rn e2etest-famE/hex-full_linear-noise-both-s005 \
    --run-tags "${FAM_TAG_BASE},famE" \
    --run-note "e2e benchmark family E (noise)"

# --- Family F: classification-style ---
print_line "9. Family F — classification-style vector outputs"
e2e_train "9. Family F: binary_vector_classification + sigmoid" \
    hexnet train -m hex -n 4 -r 0 -e "${E2E_EPOCHS}" -t binary_vector_classification \
    -a sigmoid -l mean_squared_error -lr constant \
    -rn e2etest-famF/hex-binary_vector_classification-sigmoid \
    --run-tags "${FAM_TAG_BASE},famF" \
    --run-note "e2e benchmark family F"
e2e_train "9. Family F: multi_label_linear + linear" \
    hexnet train -m hex -n 4 -r 0 -e "${E2E_EPOCHS}" -t multi_label_linear \
    -a linear -l mean_squared_error -lr constant \
    -rn e2etest-famF/hex-multi_label_linear-linear \
    --run-tags "${FAM_TAG_BASE},famF" \
    --run-note "e2e benchmark family F"

print_line "E2E test completed successfully"
