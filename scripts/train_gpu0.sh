lrs=(1e-2 1e-3 1e-4 1e-5)
flags=(1 0)

# shellcheck disable=SC2068
# shellcheck disable=SC2154
for flag in ${flags[@]}
do
  for lr in ${lrs[@]}
  do
    python main.py\
    --work_dir="flag_${flag}_lr_${lr}"\
    --device="cuda:0"\
    --options \
    "lr=${lr}" \
    "flag=${flag}"
  done
done


