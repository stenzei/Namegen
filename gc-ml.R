# Steps for using Google Cloud Machine Learning with cloudml package.
library(cloudml)

# Initializing a new project.
# Run omly if the project is new.
gcloud_init()

# Submitting the job.
# Note: Before the job can be run on Google Cloud
# the Google Cloud Machine Learning Engine API has 
# to be activated. 
# Training takes place on a standard GPU (Tesla K80 GPU)
job <- cloudml_train("namegen.R", config = "tuning.yml")

# View the status of the job.
job_status(job)

# Collect the results.
job_collect(job)

# Show training runs.
runs <- ls_runs()
str(runs)

# view the latest run
view_run()
