#!/bin/bash
set -euo pipefail

source scripts/e2e-utils.sh

#
#
# Reference Graphs
#
#
source scripts/e2e-reference.sh


#
#
# Smoke: hex (new + resume) and MLP under runs/e2etest-smoke/
#
#
# source scripts/e2e-smoke-test.sh

#
#
# Benchmark families A–F (non-exhaustive; see docs/math/benchmark-families.md)
#
#
print_line "Benchmark families A-F (runs under runs/e2etest-fam*/ )..."
# rm -rf runs/e2etest-famA runs/e2etest-famB runs/e2etest-famC runs/e2etest-famD runs/e2etest-famE runs/e2etest-famF

FAM_TAG_BASE="e2e,benchmark"

source scripts/e2e-bench-A.sh
exit 0;

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
