run = neptune.init_run(project='major/major')


#### neptune.ai API key:
set NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZjUzNzE5ZS1mMjlmLTQzNjEtYjA3NC00YTkxNGQ3YjY1ZDgifQ=="


```
eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZjUzNzE5ZS1mMjlmLTQzNjEtYjA3NC00YTkxNGQ3YjY1ZDgifQ==
```


### Integrateing with jupyter notebook
import neptune
model = neptune.init_model(
    name="Prediction model",
    key="MOD", 
    project="major/major", 
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZjUzNzE5ZS1mMjlmLTQzNjEtYjA3NC00YTkxNGQ3YjY1ZDgifQ==", # your credentials
)

```
Log notebook checkpoints automatically!
Run the model training in Jupyter Notebook, automatically log checkpoints to Neptune, and display the notebooks interactively in the app.
Step 1: Install the neptune-notebooks extension:

pip install neptune-notebooks
jupyter nbextension enable --py neptune-notebooks
For detailed installation instructions, see JupyterLab and Jupyter Notebook setup in the docs.

The following buttons appear in your Jupyter environment:



Step 2: To connect Jupyter to your Neptune account, click the Neptune icon and enter your API token.

Step 3: Upload the very first checkpoint to your Neptune project manually by clicking Upload. Enter your Neptune project name and other details.

Step 4: Run the model training.

Now, each time you run a cell containing neptune.init_run(), your notebook checkpoint will be uploaded automatically.
```


```
Step 1: Install the client library
pip install neptune
Step 2: Create a run, then log whatever model building metadata you care about.
train.py
import neptune

run = neptune.init_run(
    project="major/major",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZjUzNzE5ZS1mMjlmLTQzNjEtYjA3NC00YTkxNGQ3YjY1ZDgifQ==",
)  # your credentials

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].append(0.9 ** epoch)

run["eval/f1_score"] = 0.66

run.stop()
Step 3: Run it
python train.py
Step 4: See your metadata displayed here in Neptune!
```