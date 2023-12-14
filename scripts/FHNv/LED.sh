#--------------------------------FHNv-------------------------------- 
slow_dim=2
submodel=MLP
model=led-${submodel}-${slow_dim}
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
train_horizon=7500.0
test_horizon=3000.0
learn_n=50
predict_n=500
# tau_unit=0.1
tau_unit=10.0
dt=0.1
lr=0.001
batch_size=128
baseline_epoch=50
seed_num=5
tau_1=0.0
tau_s=0.5
device=cpu
memory_size=5000
cpu_num=1
data_dir=Data/${system}_trace_num${trace_num}_t${total_t}/data/
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
--predict_n $predict_n \
--tau_unit $tau_unit \
--tau_1 $tau_1 \
--memory_size $memory_size \
--tau_s $tau_s \
--device $device \
--cpu_num $cpu_num \
--data_dir $data_dir \
--baseline_log_dir $baseline_log_dir \
--delta1 $delta1 \
--delta2 $delta2 \
--du $du \
--xdim $xdim \
--start_t $start_t \
--end_t $end_t \
--step_t $step_t \
--slow_dim $slow_dim \
--parallel
