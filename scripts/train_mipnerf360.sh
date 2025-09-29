DATASET_PATH="./dataset/mipnerf360"
BASE_OUTPUT="./output_mipnerf360"
mkdir -p "$BASE_OUTPUT"

declare -A DATASET_CONFIGS=(
    ["bicycle"]="0.15:0.85:0.05:0.5:20_1.0"
    ["bonsai"]="0.7:0.3:0.1:0.3:5_0.5"
    ["counter"]="0.15:0.85:0.05:0.5:5_0.5"
    ["garden"]="0.3:0.7:0.05:0.7:15_0.5"
    ["kitchen"]="0.1:0.9:0.05:0.5:10_0.5"
    ["room"]="0.3:0.7:0.05:0.7:5_0.5"
    ["stump"]="0.3:0.7:0.05:0.7:10_1.0"
)

for dataset in "${!DATASET_CONFIGS[@]}"; do
    config_value="${DATASET_CONFIGS[$dataset]}"
    IFS=':' read -r depth_w density_w drop_min drop_max mask_pair <<< "$config_value"
    IFS='_' read -r mask_param lambda_far <<< "$mask_pair"
    drop_id="dw${depth_w}_dnw${density_w}_dmin${drop_min}_dmax${drop_max}"
    
    for run in {1..3}; do
        run_output="${BASE_OUTPUT}/${dataset}/${drop_id}/mask${mask_param}_lambda${lambda_far}/run_${run}"
        mkdir -p "$run_output"

        python train.py \
            -s "${DATASET_PATH}/${dataset}" \
            -m "$run_output" \
            --depth_weight "$depth_w" \
            --density_weight "$density_w" \
            --drop_min "$drop_min" \
            --drop_max "$drop_max" \
            --mask_param "$mask_param" \
            --lambda_far "$lambda_far" \
            --eval -r 8 --n_views 12

        python render.py \
            -m "$run_output" --eval -r 8

    done
done
