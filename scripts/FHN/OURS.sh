#--------------------------------FHN-------------------------------- 
model=ours
submodel=neural_ode
enc_net=MLP
e1_layer_n=2
delta1=0.1
delta2=$delta1
du=0.5
channel_num=2
xdim=30
data_dim=$xdim
system=FHN_xdim${xdim}_noise${delta1}_du${du}
obs_dim=$data_dim
trace_num=500
total_t=20.01
start_t=0.0
end_t=$total_t
step_t=$end_t
dt=0.0001
lr=0.001
batch_size=128
id_epoch=200
learn_epoch=30
seed_num=3
sliding_length=10.0
train_horizon=10.0
test_horizon=6.0
learn_n=10
predict_n=100
tau_unit=0.1
stride_t=0.01
num_heads=4
tau_1=0.1
tau_N=0.1
tau_s=2.0
embedding_dim=64
slow_dim=2
fast=1
sync=1
rho=250
inter_p=nearest_neighbour
redundant_dim=$((64-${slow_dim}))
koopman_dim=$((${slow_dim}+${redundant_dim}))
device=cpu
memory_size=5000
cpu_num=1
data_dir=Data/${system}_trace_num${trace_num}_t${total_t}/data/
id_log_dir=logs/${system}/${enc_net}/TimeSelection/
learn_log_dir=logs/${system}/${enc_net}/${submodel}-new_fast/SFS-slow${slow_dim}-tau_s${tau_s}/fast${fast}-sync${sync}-rho${rho}/${inter_p}/


python ./run.py \
--model $model \
--submodel $submodel \
--system $system \
--enc_net $enc_net \
--e1_layer_n $e1_layer_n \
--channel_num $channel_num \
--obs_dim $obs_dim \
--data_dim $data_dim \
--trace_num $trace_num \
--total_t $total_t \
--memory_size $memory_size \
--dt $dt \
--lr $lr \
--batch_size $batch_size \
--id_epoch $id_epoch \
--learn_epoch $learn_epoch \
--seed_num $seed_num \
--tau_unit $tau_unit \
--sliding_length $sliding_length \
--train_horizon $train_horizon \
--test_horizon $test_horizon \
--learn_n $learn_n \
--predict_n $predict_n \
--stride_t $stride_t \
--num_heads $num_heads \
--tau_1 $tau_1 \
--tau_N $tau_N \
--tau_list 0.001 0.01 0.1 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 \
--tau_s $tau_s \
--embedding_dim $embedding_dim \
--slow_dim $slow_dim \
--fast $fast \
--sync $sync \
--rho $rho \
--inter_p $inter_p \
--koopman_dim $koopman_dim \
--device $device \
--cpu_num $cpu_num \
--data_dir $data_dir \
--id_log_dir $id_log_dir \
--learn_log_dir $learn_log_dir \
--delta1 $delta1 \
--delta2 $delta2 \
--du $du \
--xdim $xdim \
--start_t $start_t \
--end_t $end_t \
--step_t $step_t \
--parallel \
