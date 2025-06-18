#!/bin/bash



##########################
###### Univariate ########
##########################



################
### formers ########
################
for model in FEDformer Autoformer Informer 
do

for preLen in 24 36 48 60
do
# illness
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/illness/ \
 --data_path national_illness.csv \
 --task_id ili \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --seasonality 52 \
 --itr 3
done


for preLen in 96 192 336 720
do


# ETTm2
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --task_id ETTm2 \
  --model $model \
  --data ETTm2_aa \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 512 \
  --seasonality 96 \
  --itr 3


## electricity
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/electricity/ \
 --data_path electricity.csv \
 --task_id ECL \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --seasonality 96 \
 --itr 3

# exchange
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/exchange_rate/ \
 --data_path exchange_rate.csv \
 --task_id Exchange \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --seasonality 30 \
 --itr 3

# traffic
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/traffic/ \
 --data_path traffic.csv \
 --task_id traffic \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --itr 3 \
 --seasonality 24 \
 --train_epochs 3

# weather
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/weather/ \
 --data_path weather.csv \
 --task_id weather \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --seasonality 144\
 --itr 3
done
done


###################
### NBEATS ########
###################
for model in NBEATS_I NBEATS_G
do

for preLen in 24 36 48 60
do
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/illness/ \
 --data_path national_illness.csv \
 --task_id ili \
 --model $model  \
 --data custom_aa \
 --features S \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --learning_rate 0.001 \
 --des 'Exp' \
 --itr 3\
 --seasonality 52

done

for preLen in 96 192 336 720
do

# ETTm2
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --task_id ETTm2 \
  --model $model \
  --data ETTm2_aa \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --des 'Exp' \
  --seasonality 96 \
  --learning_rate 0.001 \
  --itr 3

## electricity
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/electricity/ \
 --data_path electricity.csv \
 --task_id ECL \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --des 'Exp' \
 --seasonality 96 \
 --learning_rate 0.001 \
 --itr 3


## exchange
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/exchange_rate/ \
 --data_path exchange_rate.csv \
 --task_id Exchange \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --des 'Exp' \
 --seasonality 30 \
 --learning_rate 0.001 \
 --itr 3

## traffic
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/traffic/ \
 --data_path traffic.csv \
 --task_id traffic \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --des 'Exp' \
 --itr 3 \
 --seasonality 24 \
 --learning_rate 0.001 \
 --train_epochs 3

## weather
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/weather/ \
 --data_path weather.csv \
 --task_id weather \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --des 'Exp' \
 --itr 3 \
 --learning_rate 0.001 \
 --seasonality 144 

done
done


##################
##### wavelet ####
##################

for model in FEDformer 
do

for preLen in 24 36 48 60
do
# illness
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/illness/ \
 --data_path national_illness.csv \
 --task_id ili \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --seasonality 52 \
 --version Wavelets \
 --itr 3
done


for preLen in 96 192 336 720
do


# ETTm2
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --task_id ETTm2 \
  --model $model \
  --data ETTm2_aa \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $preLen \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 512 \
  --seasonality 96 \
  --version Wavelets \
  --itr 3


## electricity
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/electricity/ \
 --data_path electricity.csv \
 --task_id ECL \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --seasonality 96 \
 --version Wavelets \
 --itr 3

# exchange
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/exchange_rate/ \
 --data_path exchange_rate.csv \
 --task_id Exchange \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --seasonality 30 \
 --version Wavelets \
 --itr 3

# traffic
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/traffic/ \
 --data_path traffic.csv \
 --task_id traffic \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --itr 3 \
 --seasonality 24 \
 --version Wavelets \
 --train_epochs 3

# weather
python -u run.py \
 --is_training 1 \
 --root_path ./dataset/weather/ \
 --data_path weather.csv \
 --task_id weather \
 --model $model \
 --data custom_aa \
 --features S \
 --seq_len 96 \
 --label_len 48 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --des 'Exp' \
 --seasonality 144\
 --version Wavelets \
 --itr 3
done
done

