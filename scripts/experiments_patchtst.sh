#!/bin/bash




#################### original ###################



for pred_len in 24 36 48 60
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/illness/ \
      --data_path national_illness.csv \
      --task_id ili \
      --model PatchTST \
      --data custom_aa \
      --features M \
      --seq_len 36 \
      --label_len 18\
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 10\
      --itr 3\
      --batch_size 16\
      --seasonality 52\
      --learning_rate 0.0025
done



seq_len=96
model_name=PatchTST


random_seed=2021
for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --task_id traffic \
      --model PatchTST \
      --data custom_aa \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 10\
      --patience 3\
      --pct_start 0.2\
      --itr 3\
      --batch_size 24\
      --seasonality 24\
      --learning_rate 0.0001
done




for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --task_id ECL \
      --model PatchTST \
      --data custom_aa \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 10\
      --patience 3\
      --pct_start 0.2\
      --itr 3\
      --batch_size 32\
      --seasonality 96\
      --learning_rate 0.0001
done





for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm2.csv \
      --task_id ETTm2 \
      --model PatchTST \
      --data ETTm2_aa \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 10\
      --patience 3\
      --pct_start 0.4 \
      --itr 3\
      --seasonality 96\
      --batch_size 128\
      --learning_rate 0.0001
done

for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --task_id weather \
      --model PatchTST \
      --data custom_aa \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 10\
      --patience 3\
      --itr 3\
      --batch_size 128\
      --seasonality 144 \
      --learning_rate 0.0001
done




for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/exchange_rate/ \
      --data_path exchange_rate.csv \
      --task_id Exchange \
      --model PatchTST \
      --data custom_aa \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 10\
      --patience 3\
      --itr 3\
      --batch_size 128\
      --seasonality 30\
      --learning_rate 0.0001
done


###