#!/bin/bash

print_line() {
    echo "===> $1"
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
                file_path=$(echo $file | sed "s/{n}/${n}/g" | sed "s/{r}/${r}/g")
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

# 2. Making a hex net train run
print_line '2. Making a hex net train run... (n=3, r=0, e=100)'
rm -rf runs/e2etest-hex-train
hexnet train -m hex -n 3 -r 0 -e 100 -t identity -rn e2etest-hex-train
if [ $? -ne 0 ]; then
    print_line "2. Hex net train run failed"
    exit 1
fi
print_line "2. Hex net train run completed"

print_line "2. Training same hex net with r=1..."
hexnet train -m hex -n 3 -r 1 -e 100 -t identity -rd runs/e2etest-hex-train
if [ $? -ne 0 ]; then
    print_line "2. Hex net train run with r=1 failed"
    exit 1
fi
print_line "2. Hex net train run with r=1 completed"

# 3. Making a mlp net train run
print_line '3. Making a mlp net train run... (n=2, e=100)'
rm -rf runs/e2etest-mlp-train
hexnet train -m mlp -n 2 -e 100 -t identity -rn e2etest-mlp-train
if [ $? -ne 0 ]; then
    print_line "3. Mlp net train run failed"
    exit 1
fi
print_line "3. Mlp net train run completed"

print_line "E2E test completed successfully"