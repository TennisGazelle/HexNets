# NOT MEANT TO BE RUN ON IT OWN

print_line "Smoke: hex net train (n=3, r=0) and resume (r=1)..."
rm -rf runs/e2etest-smoke

e2e_train "Hex train new run" \
    hexnet train -m hex -n 3 -r 0 -e "${E2E_EPOCHS}" -t identity \
    -rn e2etest-smoke/hex-n3-r0 \
    --run-tags e2e,smoke \
    --run-note "e2e smoke hex new"

e2e_train "Hex resume same run dir with r=1" \
    hexnet train -m hex -n 3 -r 1 -e "${E2E_EPOCHS}" -t identity \
    -rd runs/e2etest-smoke/hex-n3-r0

e2e_train "MLP train" \
    hexnet train -m mlp -n 2 -e "${E2E_EPOCHS}" -t identity \
    -rn e2etest-smoke/mlp-n2 \
    --run-tags e2e,smoke \
    --run-note "e2e smoke mlp"