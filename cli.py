def train(dataset, csv_path, csv_text, csv_label, nconcepts, out_dir, embed_dim, vocab_size, nnegs, lam, rho, eta,
          window_size, lr, batch, gpu, inductive, dropout, nepochs, concept_dist, normalization):
    """Train GDCM

    DATASET is the name of the dataset to be used. It must be one of the datasets defined in `gdcm_cli/src/dataset/`
    which is a subclass of BaseDataset.

    OUT-DIR is the path to the output directory where the model, results, and visualization will be saved
    """
    def pruning_callback(study, trial):
    # Pruning callback function to stop unpromising trials
        if study.best_trial is None:
            return False
        current_best = study.best_trial.value
        if trial.value is None:
            return True
        # Prune trial if it's worse than the current best by a certain margin
        return trial.value >= current_best * 1.05   

    def objective(trial):
    # Defining the search space for hyperparameters
        embed_dim = trial.suggest_int('embed_dim', 45, 55)
        nnegs = trial.suggest_int('nnegs', 10, 20)
        nconcepts = trial.suggest_int('nconcepts', 3, 7)
        lam = trial.suggest_int('lam', 1, 20)
        rho = trial.suggest_int('rho', 1, 100)  
        eta = trial.suggest_int('eta', 1, 100)   
        lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
        min_df = trial.suggest_float('min_df', 0.01, 0.1)
        max_df = trial.suggest_float('max_df', 0.9, 1)
        batch = trial.suggest_categorical('batch_size', [ 1024, 2048, 4096])
        dropout = trial.suggest_float('dropout', 0, 0.05)
        
        if dataset == "csv":
            ds = CSVDataset(csv_path, csv_text, csv_label)
        else:
            dataset_class = get_dataset(dataset)
            ds = dataset_class()
        print("Loading data...")
        data_attr = ds.load_data({"vocab_size": vocab_size, "window_size": window_size})
        # remove gensim keys which are only used for visualization
        del data_attr["gensim_corpus"]
        del data_attr["gensim_dictionary"]

        

        gdcminer = GuidedDiverseConceptMiner(out_dir, embed_dim=embed_dim, nnegs=nnegs, nconcepts=nconcepts,
                                  lam=lam, rho=rho, eta=eta, gpu=gpu, file_log=True, inductive=inductive, norm=normalization, **data_attr)
    
        if gdcminer.device == "cuda" and torch.cuda.device_count() > 1:
                gdcminer = nn.DataParallel(gdcminer)
                print("using cuda")
        elif gdcminer.device == "mps":
                gdcminer = nn.DataParallel(gdcminer)   
                print("using mps")
        else:
             print("using cpu")
        print("Starts training")
        if isinstance(gdcminer, nn.DataParallel):
           
            res = gdcminer.module.fit(lr=lr, nepochs=nepochs, batch_size=batch, concept_dist=concept_dist)
            total_final_loss = res[-1, 0]
            gdcminer.module.visualize()
        else:
            
            res = gdcminer.fit(lr=lr, nepochs=nepochs, batch_size=batch, concept_dist=concept_dist)
            total_final_loss = res[-1, 0]
            gdcminer.visualize()
        print("Training finished. See results in " + out_dir)

        return total_final_loss
    
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.SuccessiveHalvingPruner())
    study.optimize(objective, n_trials=20, callbacks=[pruning_callback])
    
    num_hyperparameter_searches = len(study.trials)
    
    print('Best trial:')
    print('  Value: {}'.format(study.best_value))
    print('  Params: ')
    for key, value in study.best_params.items():
        print('    {}: {}'.format(key, value))

    print(f"Number of Hyperparameter Searches: {num_hyperparameter_searches}")
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()