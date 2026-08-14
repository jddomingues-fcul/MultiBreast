.PHONY: clean

PYTHON_INTERPRETER = python3

#################################################################################
# PROCESSED DATASETS                                                            #
#################################################################################

advanced-mri-lesions:
	$(PYTHON_INTERPRETER) -m data_preprocessing.preprocess --dataset advanced-mri-breast-lesions

breast-lesion-usg:
	$(PYTHON_INTERPRETER) -m data_preprocessing.preprocess --dataset breast-lesions-usg

cbis-ddsm:
	$(PYTHON_INTERPRETER) -m data_preprocessing.preprocess --dataset cbis-ddsm

cdd-cesm:
	$(PYTHON_INTERPRETER) -m data_preprocessing.preprocess --dataset cdd-cesm

embed:
	$(PYTHON_INTERPRETER) -m data_preprocessing.preprocess --dataset embed

rsna-bcd:
	$(PYTHON_INTERPRETER) -m data_preprocessing.preprocess --dataset rsna-bcd

cmmd:
	$(PYTHON_INTERPRETER) -m data_preprocessing.preprocess --dataset cmmd

breast-micro-calc:
	$(PYTHON_INTERPRETER) -m data_preprocessing.preprocess --dataset breast-micro-calc

mama-mia:
	$(PYTHON_INTERPRETER) -m data_preprocessing.preprocess --dataset mama-mia

oasbud:
	$(PYTHON_INTERPRETER) -m data_preprocessing.preprocess --dataset oasbud

inbreast:
	$(PYTHON_INTERPRETER) -m data_preprocessing.preprocess --dataset inbreast

la-breast:
	$(PYTHON_INTERPRETER) -m data_preprocessing.preprocess --dataset la-breast

processed-datasets: advanced-mri-lesions breast-lesion-usg breast-micro-calc cbis-ddsm cmmd cdd-cesm oasbud la-breast inbreast mama-mia embed rsna-bcd
	@echo "All datasets processed"

#################################################################################
# SPLITS                                                                        #
#################################################################################
report_generation_dataset:
	$(PYTHON_INTERPRETER) -m data_split.report_generation_split --processed_data_path ../data/processed --save_path ../data/report_generation_split --hold_modality_out

process_and_split: processed-datasets report_generation_dataset
	@echo "All datasets processed and split"

#################################################################################
# TRAINING                                                                      #
#################################################################################
train:
	$(PYTHON_INTERPRETER) -m models.amber.train --yaml_config configs/train/amber/baseline.yaml --seed 1
	$(PYTHON_INTERPRETER) -m models.amber.train --yaml_config configs/train/amber/baseline_us.yaml --seed 1
	$(PYTHON_INTERPRETER) -m models.amber.train --yaml_config configs/train/amber/baseline_cesm.yaml --seed 1
	$(PYTHON_INTERPRETER) -m models.amber.train --yaml_config configs/train/amber/baseline_mg.yaml --seed 1
	$(PYTHON_INTERPRETER) -m models.amber.train --yaml_config configs/train/amber/baseline_mr.yaml --seed 1
	$(PYTHON_INTERPRETER) -m models.amber_no_modality_text.train --yaml_config configs/train/amber_no_modality_text/baseline.yaml --seed 1
	$(PYTHON_INTERPRETER) -m models.amber.train --yaml_config configs/train/amber/baseline_homo_no_mg.yaml --seed 1
	$(PYTHON_INTERPRETER) -m models.amber.train --yaml_config configs/train/amber/baseline_homo_no_us.yaml --seed 1
	$(PYTHON_INTERPRETER) -m models.amber.train --yaml_config configs/train/amber/baseline_homo_no_cesm.yaml --seed 1
	$(PYTHON_INTERPRETER) -m models.amber.train --yaml_config configs/train/amber/baseline_homo_no_mr.yaml --seed 1

#################################################################################
# EVALUATING                                                                    #
#################################################################################

eval_paired:
	$(PYTHON_INTERPRETER) -m eval.eval --eval_config configs/eval_paired/us_vs_all.yaml
	$(PYTHON_INTERPRETER) -m eval.eval --eval_config configs/eval_paired/cesm_vs_all.yaml
	$(PYTHON_INTERPRETER) -m eval.eval --eval_config configs/eval_paired/mr_vs_all.yaml
	$(PYTHON_INTERPRETER) -m eval.eval --eval_config configs/eval_paired/dm_vs_all.yaml

eval_on_set:
	$(PYTHON_INTERPRETER) -m eval.eval_on_set_cls --eval_on_set_config configs/eval_on_set/amber_cls/baseline_on_all.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_on_set --eval_on_set_config configs/eval_on_set/amber/baseline_on_all.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_on_set --eval_on_set_config configs/eval_on_set/amber_no_modality_text/baseline_on_all.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_on_set --eval_on_set_config configs/eval_on_set/amber/baseline_on_cesm.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_on_set --eval_on_set_config configs/eval_on_set/amber/baseline_on_us.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_on_set --eval_on_set_config configs/eval_on_set/amber/baseline_on_mr.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_on_set --eval_on_set_config configs/eval_on_set/amber/baseline_on_mg.yaml

eval_single:
	$(PYTHON_INTERPRETER) -m eval.eval_single_cls --eval_single_config configs/eval_single/amber_cls/baseline_on_cesm.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single_cls --eval_single_config configs/eval_single/amber_cls/baseline_on_us.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single_cls --eval_single_config configs/eval_single/amber_cls/baseline_on_mr.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single_cls --eval_single_config configs/eval_single/amber_cls/baseline_on_mg.yaml

	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber/baseline_on_cesm.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber/baseline_on_us.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber/baseline_on_mr.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber/baseline_on_mg.yaml
#
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber/cesm_model_on_cesm.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber/us_model_on_us.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber/mr_model_on_mr.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber/mg_model_on_mg.yaml

	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_no_modality_text/baseline_on_cesm.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_no_modality_text/baseline_on_us.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_no_modality_text/baseline_on_mr.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_no_modality_text/baseline_on_mg.yaml

	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_cesm_on_cesm.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_cesm_on_us.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_cesm_on_mr.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_cesm_on_mg.yaml

	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_us_on_cesm.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_us_on_us.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_us_on_mr.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_us_on_mg.yaml

	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_mr_on_cesm.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_mr_on_us.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_mr_on_mr.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_mr_on_mg.yaml

	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_mg_on_cesm.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_mg_on_us.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_mg_on_mr.yaml
	$(PYTHON_INTERPRETER) -m eval.eval_single --eval_single_config configs/eval_single/amber_homo/homo_no_mg_on_mg.yaml

#################################################################################
# CLEANING
#################################################################################
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "wandb" -exec rm -rf {} +
	find . -type d -name "artifacts" -exec rm -rf {} +
	find . -type d -name "lightning_logs" -exec rm -rf {} +
	find . -type f -name "*.index" -delete
	find . -type f -name "*.index.classes" -delete
	find . -type f -name "*.DS_Store" -delete
	find . -type d -name "extracted_features" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type d -name ".model_artefacts" -exec rm -rf {} +
	rm nohup.out

clean_plots:
	rm -rf plots/*
	rm -rf models_analysis/plots/*
	rm -rf torch_matrices/*

clean_logs:
	rm -rf logs/*
