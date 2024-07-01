

if [[ "${logdir}" == *"gcgens"* ]]; then
    python3 eval_generate.py --log_path ${logdir}
else
    python3 collect_response.py --log_path ${logdir}
fi

python3 eval_harmbench.py --log_path ${logdir}

