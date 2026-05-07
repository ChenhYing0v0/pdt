from __future__ import annotations


def print_args(args) -> None:
    print("Basic Config")
    print(f"  task_name={args.task_name} is_training={args.is_training} model_id={args.model_id} model={args.model}")
    print("Data")
    print(
        f"  data={args.data} root_path={args.root_path} data_path={args.data_path} "
        f"features={args.features} target={args.target} freq={args.freq}"
    )
    print("Forecast")
    print(
        f"  seq_len={args.seq_len} label_len={args.label_len} pred_len={args.pred_len} "
        f"enc_in={args.enc_in} d_model={args.d_model} d_ff={args.d_ff} e_layers={args.e_layers}"
    )
    print("PDT")
    print(
        f"  q_mat={args.q_mat_file} r_mat={args.r_mat_file} rk_mat={args.rk_mat_file} "
        f"embed_size={args.embed_size} r_rank={args.r_rank} k_top={args.k_top} "
        f"alpha_init={args.alpha_init} mask_threshold={args.mask_threshold}"
    )
    print("Run")
    print(
        f"  batch_size={args.batch_size} train_epochs={args.train_epochs} patience={args.patience} "
        f"lr={args.learning_rate} lradj={args.lradj} use_amp={args.use_amp}"
    )
    print("Output")
    print(
        f"  checkpoints={args.checkpoints} results={args.results} "
        f"test_results={args.test_results} log_path={args.log_path}"
    )
