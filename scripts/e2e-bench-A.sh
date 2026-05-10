# NOT MEAN TO BE RUN ON ITS OWN

# --- Family A: linear structure recovery ---
print_line "Family A — linear structure recovery (hex + sample MLP)"
DATASETS=(
    identity
    linear_scale
    diagonal_scale
    diagonal_linear
    full_linear
    low_rank_linear
    affine
    orthogonal_rotation
)
ACTIVATIONS=(sigmoid relu linear leaky_relu)
LOSSES=(mean_squared_error huber)
LEARNING_RATES=(constant exponential_decay)


for dataset in "${DATASETS[@]}"; do
    for activation in "${ACTIVATIONS[@]}"; do
        for loss in "${LOSSES[@]}"; do
            combo_string="${dataset}-${activation}-${loss}"
            e2e_train "Family A hex: ${combo_string}" \
                hexnet train \
                -m hex -n 4 -r 0 \
                -e "${E2E_EPOCHS}" \
                -t "${dataset}" \
                -a "${activation}" \
                -l "${loss}" \
                -lr "constant" \
                -rn "e2etest-famA/hex-${combo_string}-n4" \
                --run-tags "${FAM_TAG_BASE},famA,${dataset},${activation},${loss}" \
                --run-note "e2e benchmark family A (linear)" --dry-run

            e2e_train "Family A mlp: ${combo_string}"
                hexnet train \
                -e "${E2E_EPOCHS}" \
                -t "${dataset}" \
                -a "${activation}" \
                -l "${loss}" \
                -lr "constant" \
                -rn "e2etest-famA/mlp-${combo_string}-n4" \
                --run-tags "${FAM_TAG_BASE},famA,${dataset},${activation},${loss}" \
                --run-note "e2e benchmark family A (linear)" --dry-run
        done
    done
done