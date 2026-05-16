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
N_DIMENSIONS=(3)

for dataset in "${DATASETS[@]}"; do
    for activation in "${ACTIVATIONS[@]}"; do
        for loss in "${LOSSES[@]}"; do
            for learning_rate in "${LEARNING_RATES[@]}"; do
                combo_string="${dataset}-${activation}-${loss}-${learning_rate}-n${N_DIMENSIONS}"
                run_tags="${FAM_TAG_BASE},famA,${dataset},${activation},${loss},${learning_rate}"

                rm -rf "runs/e2etest-famA/hex-${combo_string}"
                e2e_train "Family A hex: ${combo_string}" \
                    hexnet train \
                        --model hex --num_dims "${N_DIMENSIONS}" --rotation 0 \
                        --epochs "${E2E_EPOCHS}" \
                        --type "${dataset}" \
                        --activation "${activation}" \
                        --loss "${loss}" \
                        --learning-rate "${learning_rate}" \
                        --run-name "e2etest-famA/hex-${combo_string}" \
                        --run-tags "${FAM_TAG_BASE},famA,${dataset},${activation},${loss}" \
                        --run-note "e2e benchmark family A (linear)"\
                        --dry-run

                rm -rf "runs/e2etest-famA/mlp-${combo_string}"
                e2e_train "Family A mlp: ${combo_string}"
                    hexnet train \
                        --model mlp --num_dims "${N_DIMENSIONS}" \
                        --epochs "${E2E_EPOCHS}" \
                        --type "${dataset}" \
                        --activation "${activation}" \
                        --loss "${loss}" \
                        --learning-rate "${learning_rate}" \
                        --run-name "e2etest-famA/mlp-${combo_string}" \
                        --run-tags "${FAM_TAG_BASE},famA,${dataset},${activation},${loss}" \
                        --run-note "e2e benchmark family A (linear)"\
                        --dry-run
            done
        done
    done
done