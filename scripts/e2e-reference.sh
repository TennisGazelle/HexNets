# NOT MEAN TO BE RUN ON ITS OWN

print_line 'Creating all the reference graphs...'
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
    print_line "Found all reference graphs"
fi