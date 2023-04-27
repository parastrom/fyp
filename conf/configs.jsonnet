{
     general: {
        mode: 'train',
        batch_size: 32,
        use_cuda: true,
        is_cloud: false,
        is_debug: false,
        use_fp16: 0,
        device: "cuda:0"
    },
    model: {
        pretrain_model_type: 'BERT',
        pretrain_model: 'bert-large-uncased-whole-word-masking',
        init_model_params: null,
        init_model_optim: null,
        model_name: 'seq2tree_v2',
        grammar_type: 'spider',
        rat_layers: 8,
        rat_heads: 8,
        enc_value_with_col: true,
        num_value_col_type: 'q_num', # cls|col_0|q_num
        value_memory: true,
        predict_value: true,
        max_seq_len: 512,
        max_question_len: 120,
        max_column_num: 500,
        max_table_num: 50,
        max_column_tokens: 50,  # useless
        max_table_tokens: 20,   # useless
    },
    data: {
        db: 'conf/cache/test_db_file.pkl',
        grammar: 'conf/label_vocabs',
        train_set: 'conf/cache/train_data_file.pkl',
        dev_set: 'conf/cache/validation_data_file.pkl',
        test_set: null,
        eval_file: null,
        output: 'output',
        is_cached: false,
    },
    train: {
        epochs: 30,
        log_steps: 10,
        trainer_num: 1,
        # [begin] config for optimizer
        learning_rate: 3e-05,
        lr_scheduler: "linear_warmup_decay",
        warmup_steps: 0,
        warmup_proportion: 0.1,
        weight_decay: 0.01,
        use_dynamic_loss_scaling: false,
        init_loss_scaling: 128,
        incr_every_n_steps: 100,
        decr_every_n_nan_or_inf: 2,
        incr_ratio: 2.0,
        decr_ratio: 0.8,
        grad_clip: 1.0,
        # [end] optimizer
        random_seed: null,
        use_data_parallel: false,
    }
}