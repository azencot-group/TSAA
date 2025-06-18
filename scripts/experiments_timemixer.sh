#!/bin/bash


model_name=TimeMixer




for preLen in 24 36 48 60
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --task_id ili \
  --model $model_name \
  --data custom_aa \
  --features M \
  --seq_len 36\
  --label_len 18 \
  --pred_len $preLen \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 3 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size 16 \
  --learning_rate 0.01 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --seasonality 52
done

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=32
batch_size=16

for preLen in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTm2.csv \
  --task_id ETTm2 \
  --model $model_name \
  --data ETTm2_aa \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 3 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window


seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=32
train_epochs=20
patience=10

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --task_id ECL \
  --model $model_name \
  --data custom_aa \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 3 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window\
  --seasonality 96


seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=64
batch_size=8


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --task_id Traffic \
  --model $model_name \
  --data custom_aa \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 3 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window\
  --seasonality 24


seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=16
train_epochs=20
patience=10

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --task_id weather \
  --model $model_name \
  --data custom_aa \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 3 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --seasonality 144 


python -u run.py \
 --is_training 1 \
 --root_path ./dataset/exchange_rate/ \
 --data_path exchange_rate.csv \
 --task_id Exchange \
 --model $model_name \
 --data custom_aa \
 --features M \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers $e_layers \
 --d_layers 1 \
 --factor 3 \
 --enc_in 8 \
 --dec_in 8 \
 --c_out 8 \
 --des 'Exp' \
 --itr 3\
 --d_model $d_model \
 --d_ff $d_ff \
 --batch_size 128 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --down_sampling_method avg \
 --down_sampling_window $down_sampling_window  \
 --seasonality 30 
done

