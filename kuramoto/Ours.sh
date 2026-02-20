#--------------------------------Kuramoto-------------------------------- 
model=ours
system=Kuramoto
channel_num=1
xdim=50
data_dim=$xdim
obs_dim=$data_dim
trace_num=200
total_t=200.0
start_t=0.0
end_t=$total_t
step_t=$end_t
dt=0.05
lr=0.01
batch_size=128
id_epoch=50
learn_epoch=25
seed_num=1
sliding_length=150.0
train_horizon=150.0
test_horizon=50.0
learn_n=10
predict_n=100
tau_unit=0.05
stride_t=0.5
# tau_1=0.1
# tau_N=0.1

# Slow-Fast parameters
tau_s=0.8
slow_dim=3

# Model parameters
submodel=neural_ode
enc_net=MLP
e1_layer_n=2
num_heads=2
embedding_dim=64
fast=1
sync=1
rho=1
inter_p=nearest_neighbour
auto=1
redundant_dim=$((64-${slow_dim}))
koopman_dim=$((${slow_dim}+${redundant_dim}))
device=cuda
memory_size=2000
cpu_num=1

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
--tau_list 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 \
--tau_s $tau_s \
--embedding_dim $embedding_dim \
--slow_dim $slow_dim \
--fast $fast \
--sync $sync \
--rho $rho \
--inter_p $inter_p \
--auto $auto \
--koopman_dim $koopman_dim \
--device $device \
--memory_size $memory_size \
--cpu_num $cpu_num \
--xdim $xdim \
--start_t $start_t \
--end_t $end_t \
--step_t $step_t \
--parallel
