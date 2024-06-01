

if [[ "${logdir}" == *"gcgens"* ]]; then
    python3 eval_generate.py --log_dir ${logdir}
else
    python3 collect_response.py --log_dir ${logdir}
fi

python3 eval_harmbench.py --log_dir ${logdir}

