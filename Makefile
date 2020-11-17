VENV_NAME?=election
# Need to specify bash in order for conda activate to work.
SHELL=/bin/zsh
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
# single $ is Makefile variable
# double $ is shell variable
CONDA_SOURCE=source $$(conda info --base)/etc/profile.d/conda.sh ;

help:
	@echo "make setup"
	@echo "		setup conda environment, run upon only once"
	@echo "make update"
	@echo "		update environment with latest requiremetns"

setup:
	conda create -y -n $(VENV_NAME) python=3.7
	make update
venv:
	$(CONDA_SOURCE) conda activate ; conda activate $(VENV_NAME)
update:
	make venv
	conda env update -f environment.yml
	
	