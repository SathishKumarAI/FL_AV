federated_learning_project/
│
├── data/
│   ├── client_1/
│   │   ├── data.csv
│   │   └── ...
│   ├── client_2/
│   │   ├── data.csv
│   │   └── ...
│   ├── ...
│   └── global_model/
│       ├── model.pkl
│       └── ...
│
├── models/
│   ├── model.py
│   └── ...
│
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   └── ...
│
├── experiments/
│   ├── experiment_1/
│   │   ├── config.json
│   │   ├── run_experiment.py
│   │   └── ...
│   └── experiment_2/
│       ├── config.json
│       ├── run_experiment.py
│       └── ...
│
├── logs/
│   ├── client_1/
│   │   ├── training_log.txt
│   │   └── ...
│   ├── client_2/
│   │   ├── training_log.txt
│   │   └── ...
│   └── global/
│       ├── aggregation_log.txt
│       └── ...
│
├── main.py
├── requirements.txt
└── README.md
