#--------------------------------HalfMoon-------------------------------- 
slow_dim=1
submodel=MLP
model=led-${submodel}-${slow_dim}
channel_num=1
xdim=1
data_dim=4
obs_dim=2
system=HalfMoon_2D
trace_num=50
total_t=31400.0
start_t=0.0
end_t=$total_t
step_t=$end_t
train_horizon=20000.0
test_horizon=10000.0
learn_n=100
predict_n=1000
# tau_unit=1.0
tau_unit=1.0
dt=1.0
lr=0.01
batch_size=128
baseline_epoch=10
seed_num=3
tau_1=0.1
tau_s=20.0
device=cpu
memory_size=3000
cpu_num=1
data_dir=Data/${system}_trace_num${trace_num}/data/
baseline_log_dir=logs/${system}/$model/


python ./run.py \
--model $model \
--submodel $submodel \
--system $system \
--channel_num $channel_num \
--obs_dim $obs_dim \
--data_dim $data_dim \
--trace_num $trace_num \
--total_t $total_t \
--dt $dt \
--lr $lr \
--batch_size $batch_size \
--baseline_epoch $baseline_epoch \
--seed_num $seed_num \
--train_horizon $train_horizon \
--test_horizon $test_horizon \
--learn_n $learn_n \
--memory_size $memory_size \
--predict_n $predict_n \
--tau_unit $tau_unit \
--tau_1 $tau_1 \
--tau_s $tau_s \
--device $device \
--cpu_num $cpu_num \
--data_dir $data_dir \
--baseline_log_dir $baseline_log_dir \
--xdim $xdim \
--start_t $start_t \
--end_t $end_t \
--step_t $step_t \
--slow_dim $slow_dim \
--parallel
