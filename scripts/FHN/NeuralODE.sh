#--------------------------------FHN-------------------------------- 
submodel=MLP
model=neural_ode-${submodel}
delta1=0.1
delta2=$delta1
channel_num=2
xdim=30
du=0.5
data_dim=$xdim
system_name=FHN
system=${system_name}_xdim${xdim}_noise${delta1}_du${du}
obs_dim=$data_dim
trace_num=500
total_t=20.01
start_t=0.0
end_t=$total_t
step_t=$end_t
dt=0.0001
lr=0.001
batch_size=128
baseline_epoch=20
seed_num=3
train_horizon=10.0
test_horizon=6.0
learn_n=10
predict_n=100
tau_unit=0.1
stride_t=0.01
tau_1=0.0
tau_s=2.0
redundant_dim=0
device=cpu
memory_size=5000
cpu_num=1
data_dir=Data/${system}_trace_num${trace_num}_t${total_t}/data/
baseline_log_dir=logs/${system}/$model-tau_s${tau_s}/


python ./run.py \
--model $model \
--submodel $submodel \
--system_name $system_name \
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
--memory_size $memory_size \
--stride_t $stride_t \
--tau_1 $tau_1 \
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
--parallel \
