#--------------------------------Coupled_Lorenz-------------------------------- 
slow_dim=2
submodel=MLP
model=neural_ode-${submodel}
channel_num=3
data_dim=2
obs_dim=$data_dim

system=Coupled_Lorenz_0.05
trace_num=100
total_t=500.0
start_t=0.0
end_t=$total_t
step_t=$end_t
train_horizon=100.0
test_horizon=100.0
learn_n=50
predict_n=5000
dt=0.01
tau_unit=$dt
lr=0.01
batch_size=128
baseline_epoch=50
seed_num=5
tau_1=0.0
tau_s=2.0
device=cpu
memory_size=3000
cpu_num=1
data_dir=Data/${system}/data/
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
--tau_s $tau_s \
--slow_dim $slow_dim \
--device $device \
--memory_size $memory_size \
--cpu_num $cpu_num \
--data_dir $data_dir \
--baseline_log_dir $baseline_log_dir \
--start_t $start_t \
--end_t $end_t \
--step_t $step_t \
--parallel
