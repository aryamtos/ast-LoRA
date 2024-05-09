

data_train=/metadados_complete_spotify_train.csv
data_val=/metadados_complete_spotify_dev.csv
data_test=/metadados_complete_spotify_test.csv
seed=10
device=cuda
num_workers=2
output_path=/ast_lora/checkpoints/
dataset_name=SPY
method=LoRA



python3 main.py --data_train ${data_train} --data_val ${data_val} --data_test ${data_test} --seed $seed \
--device ${device} --num_workers $num_workers --output_path ${output_path} --dataset_name ${dataset_name} --method ${method}


# python3 main.py  --data_path '/path_to_your_dataset' --dataset_name 'FSC' --method 'adapter' --seq_or_par 'parallel' --reduction_rate_adapter 64 --adapter_type 'Pfeiffer' --apply_residual False --adapter_block 'bottleneck'