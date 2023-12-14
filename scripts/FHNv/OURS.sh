#--------------------------------FHNv-------------------------------- 
model=ours
enc_net=MLP
# submodel=lstm
submodel=neural_ode
e1_layer_n=2
delta1=0.2
delta2=0.2
du=0.5
channel_num=2
xdim=101
data_dim=$xdim
system=FHNv_xdim${xdim}_noise${delta1}_du${du}
obs_dim=$data_dim
trace_num=5
total_t=8000.0
start_t=0.0
end_t=$total_t
step_t=$end_t
dt=0.1
lr=0.001
batch_size=128
id_epoch=300
learn_epoch=30
seed_num=3
sliding_length=2000.0
train_horizon=7500.0
test_horizon=3000.0
learn_n=50
predict_n=500
# tau_unit=0.1
# stride_t=0.1
tau_unit=10.0
stride_t=1.0
num_heads=2
tau_s=0.5
embedding_dim=64
auto=1
slow_dim=3
fast=1
mask_slow=0
rho=1
inter_p=nearest_neighbour
redundant_dim=$((64-${slow_dim}))
koopman_dim=$((${slow_dim}+${redundant_dim}))
device=cuda
memory_size=5000
cpu_num=1
data_dir=Data/${system}_trace_num${trace_num}_t${total_t}/data/
id_log_dir=logs/${system}/${enc_net}/TimeSelection/
learn_log_dir=logs/${system}/${enc_net}/${submodel}/no-detach-no_adiab-LearnDynamics-slow${slow_dim}-tau_s${tau_s}/fast${fast}-mask_slow${mask_slow}-rho${rho}/${inter_p}/

python ./run.py \
--model $model \
--system $system \
--enc_net $enc_net \
--e1_layer_n $e1_layer_n \
--channel_num $channel_num \
--obs_dim $obs_dim \
--data_dim $data_dim \
--trace_num $trace_num \
--total_t $total_t \
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
--tau_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 2.5 5.0 7.5 10.0 \
--tau_s $tau_s \
--embedding_dim $embedding_dim \
--auto $auto \
--slow_dim $slow_dim \
--fast $fast \
--mask_slow $mask_slow \
--rho $rho \
--inter_p $inter_p \
--koopman_dim $koopman_dim \
--device $device \
--memory_size $memory_size \
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
