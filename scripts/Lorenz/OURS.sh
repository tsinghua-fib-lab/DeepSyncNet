#--------------------------------Coupled_Lorenz-------------------------------- 
model=ours
submodel=neural_ode
enc_net=MLP
e1_layer_n=2
delta1=0.0
delta2=$delta1
du=0.1
channel_num=3
xdim=2
data_dim=$xdim
system=Coupled_Lorenz_0.05
obs_dim=$data_dim
trace_num=100
total_t=500.0
start_t=0.0
end_t=$total_t
step_t=$end_t
dt=0.01
lr=0.01
batch_size=128
id_epoch=30
learn_epoch=50
seed_num=5
sliding_length=400.0
train_horizon=400.0
test_horizon=100.0
learn_n=100
predict_n=5000
tau_unit=0.01
stride_t=0.05
num_heads=1
tau_1=0.1
tau_N=0.1
tau_s=2.0
embedding_dim=16
slow_dim=2
fast=1
sync=1
rho=1
inter_p=nearest_neighbour
auto=1
redundant_dim=$((64-${slow_dim}))
koopman_dim=$((${slow_dim}+${redundant_dim}))
device=cpu
memory_size=3000
cpu_num=1
data_dir=Data/${system}/data/
id_log_dir=logs/${system}/${enc_net}/TimeSelection-auto${auto}/
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
--tau_list 0.01 0.02 0.05 0.06 0.1 0.2 0.5 0.6 1.0 2.0 3.0 4.0 5.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0 \
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
--parallel
