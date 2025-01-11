#--------------------------------HalfMoon 2D--------------------------------
model=ours
submodel=neural_ode
enc_net=MLP
e1_layer_n=3
channel_num=1
xdim=1
data_dim=4
obs_dim=2
slow_dim=1
system=HalfMoon_2D
trace_num=50
total_t=31400.0
start_t=0.0
end_t=$total_t
step_t=$end_t
dt=1.0
lr=0.001
batch_size=128
id_epoch=20
learn_epoch=25
seed_num=5
sliding_length=20000.0
train_horizon=20000.0
test_horizon=10000.0
learn_n=100
predict_n=1000
tau_unit=1.0
# stride_t=1.0
stride_t=100.0
num_heads=1
tau_s=30.0
embedding_dim=64
bi_info=1
auto=1
redundant_dim=$((64-${slow_dim}))
koopman_dim=$((${slow_dim}+${redundant_dim}))
device=cpu
memory_size=3000
cpu_num=1
data_dir=Data/${system}_trace_num${trace_num}/data/
id_log_dir=logs/${system}/${enc_net}/TimeSelection/auto_${auto}/
learn_log_dir=logs/${system}/${enc_net}/${submodel}/no-detach-no_adiab-LearnDynamics-slow${slow_dim}-tau_s${tau_s}/


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
--tau_list 1.0 3.0 5.0 7.0 9.0 10.0 20.0 30.0 40.0 50.0 75.0 100.0 \
--tau_s $tau_s \
--embedding_dim $embedding_dim \
--bi_info $bi_info \
--auto $auto \
--slow_dim $slow_dim \
--koopman_dim $koopman_dim \
--device $device \
--memory_size $memory_size \
--cpu_num $cpu_num \
--data_dir $data_dir \
--id_log_dir $id_log_dir \
--learn_log_dir $learn_log_dir \
--xdim $xdim \
--start_t $start_t \
--end_t $end_t \
--step_t $step_t \
--parallel \
