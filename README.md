Code supplement for
"Measuring and Optimizing for Rank-Based Compatibility When Updating 
Risk Stratification Models"

Introduces a new rank-based compatibility metric: 
	rbc_score in compatibility_metrics.py
Intoduces rank-based imcompatibility loss:
	make_approx_incompatibility_loss_fx in model_functions.py
	note the hyperparameter s=100
Builds updated models that are compatible with original models
	make_f_u in model_functions.py
Runs updating experiments
	utils_experiments.py
Experiment analysis functions
	utils_analysis.py


Overview of Contents
	compatibility_metrics.py 
		contains implementations of existing and new compatibility metrics
	example_analysis_heatmap.ipynb
		analysis notebook for heatmap and \alpha \beta sweep figures	example_analysis.ipynb
		analysis notebook for central tendency and model performance overview figures	example_experiment.ipynb
		notebook to kick off the experiment	model_functions.py
		helper module with tensorflow loss functions and model creation functions			utils_analysis.py
		functions to help with analysis	utils_experiment.py
		functions to run the primary updating experiment


How to run
	call setup_and_run (utils_experiments.py)
		example_experiment.ipynb does this 
		you will need to supply your own data though
		we can't directly share MIMIC-III data
		this will set up a experiment directory with data, models, and results
	once complete you can access the res.csv file in the specified experiment directory
		example_analysis.ipynb and example_analysis_heatmap.ipynb both point
		at a res.csv file

Some setup_and_run details:
	this runs the main experiment, it generates splits data, generates original models,
	then generates a bunch of updated models
	this process is then replicated
	this involves lots of model training
	so we parallelized the process
	in order to do this we have to store some information (namely dataset splits and
	models) to disk

	parameters
	experiment_dir: the directory you want the experiment to be stored to (string)
	X: the feature matrix (2D numpy array)
	y: label vector (1D numpy array)
	Cs=[0]: list of of L2 regularization weights
	alphas=[0, 0.5, 1.0]: list of alphas
	n_reps=5: number of replications
	n_jobs=5: number of parallel jobs you want
	n_engineered=1: number of ``RBC models'' you want trained
	n_engineered_resample=0: number of ``RBC models'' you want trained using a resampling
				 of the dataset (this is for more variation in models)
	n_standard=1: number of ``BCE models''
	n_standard_resample=1: number of ``BCE models'' trained using resampling
	n_od=200: dataset size for the original model development
	n_oe=200: dataset size for the original model validation
	n_ud=3000: dataset size for the updated model development
	n_ue=3000: dataset size for the original model validation
        f_o_optimizer='Adam': the optimizer used for the original model
        f_u_optimizer='Adam': the optimizer used for the updated model
	f_o_fit_kwargs={'epochs': 100, 'verbose': True}: the fit kwargs for the original model
	f_u_fit_kwargs={'epochs': 100, 'verbose': True}: the fit kwargs for the updated model
	f_o_early_stopping_kwargs: the early stopping kwargs for the original model
	f_u_early_stopping_kwargs: the early stopping kwargs for the updated model


	Note: if the setup_and_run process dies you can kick if off by calling ''run'' on the 
	experiment directory

