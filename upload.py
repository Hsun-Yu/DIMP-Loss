# %%
import wandb
# %%
run = wandb.init(name="upload_imp_diversity_checker_for_mrpc", project="DIMP-Loss", job_type="add-dataset")
artifact = wandb.Artifact(name="diversity-checker_IMP_glue_mrpc_bert_5", type="dataset")
artifact.add_dir("./artifacts/imp_loss_diversity_checker_for_mrpc")
artifact.save()

# Logs the artifact version "my_data" as a dataset with data from dataset.h5
# %%
# %%
