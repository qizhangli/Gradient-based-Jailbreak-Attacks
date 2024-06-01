

result_name="exp_${model}_${method}_topk4_bs20_seed${seed}"
savepath="./results/${result_name}"


# Create results folder if it doesn't exist
if [ ! -d ${savepath} ]; then
    mkdir ${savepath}
    echo "Folder ${savepath} created."
else
    echo "Folder ${savepath} already exists."
fi

if [[ "${method}" == *"gcgens"* ]]; then
    end=0
else
    end=99
fi


for ind in $(seq 0 $end)
do
savedir="${savepath}/${result_name}_ind${ind}"
python3 attack.py --model_name ${model} --method ${method} --data_index ${ind} --savedir ${savedir} --seed ${seed}
done
