# Test Technique Antoine Audras

## Démarage de InfluxDB

L'instance de InfluxDB utilisée est gérée grâce au fichier Docker compose résidant à la
racine du projet.

Avant de démarrer le serveur il faut définir 3 fichier contenant les différants secrets
nécéssaire au setup. Il vous faudra créer les fichiers suivant:
 - `.env.influxdb2-admin-password` : contenant le mdp admin
 - `.env.influxdb2-admin-token` : contenant le token admin
 - `.env.influxdb2-admin-username` : contenant le nom d'utilisateur admin

Une fois les fichiers crées il vous suffira de lancer l'instance à l'aide de la commande
`docker compose up`.

Si tout s'est déroulé normalement, vous devriez avoir accès à un dashboard à l'adresse
suivante : `http://localhost:8086`.


# Installation des requirements

Un fichier `requirements.txt` est fourni à la racine du projet, vous pouvez installer
les dépendences python avec la commande `pip install -r requirements.txt`.


## Chargement du csv train dans InfluxDB

Pour charger les données d'entrainement dans InfluxDB vous diposez d'un script dans
le dossier `scripts/`.

Vous pouvez l'appeler avec la commande suivante : `python scripts/import_alki_csv.py chemin/fichier.csv`
pour charger les données du fichier train.


## Export des variables d'environnement

Afin de communiquer avec InfluxDB, vous aurez besoin d'exporter quelques variables d'environnement :
 - `INFLUXDB_V2_URL` : contenant l'adresse du serveur (http://localhost:8086 par défaut)
 - `INFLUXDB_V2_ORG` : "alki" par défaut ou toute autre valeur spécifiée dans le setup du Docker compose
 - `INFLUXDB_V2_TOKEN` : la valeur du token défini en secret pour le Docker compose dans le fichier `.env.influxdb2-admin-token`.


## Entrainement d'un modèle TFT

Pour lancer l'entrainement d'un modèle TFT il vous faudra utiliser le point dentrée
`run.py`.

Grâce à Hydra vous pouvez utiliser les configs par défauts et remplacer des paramètres
au moment de lancer l'experience en utilisant le méchanisme de surcharge.

Par exemple si vous souhaitez entrainer un modèle TFT avec la config de base mais une
`hidden_size` plus grande et 3 couches de LSTM au lieu de 1, vous pouvez lancer la
commande suivante : `python run.py mode=train +experiments=tft model.hidden_size=128 model.lstm_layers=3 checkpoint.work_dir=outputs/64_lstm_3`.

Il est impératif de définir un `mode` "train" ou "eval", le reste est libre à
l'utilsateur.

Les artecfacts seront sauvegardés dans le dossier `outputs/64_lstm_3/` précisé par le
paramètre `checkpoint.work_dir`.

Vous pouvez lire la configuration de base de l'[experience]](conf/experiments/tft.yaml)
ainsi que la configuration du [modèle](conf/models/tft.yaml) pour comprendre les
principaux paramètres.

Si vous souhaitez afficher la configuration entière afin de mieux en comprendre la
topologie, vous pouvez le faire grâce en passant les arguments `--cfg job` et
`--resolve` au script.

exemple : `python run.py mode=train +experiments=tft model.hidden_size=128 model.lstm_layers=3 checkpoint.work_dir=outputs/64_lstm_3 --cfg job --resolve`

<details>
  <summary>Output de la commande</summary>

  ```bash
    mode: train
    checkpoint:
      save_checkpoints: true
      work_dir: outputs/64_lstm_3
      model_name: model_tft
      file_name: null
    model:
      _target_: darts.models.TFTModel
      input_chunk_length: 24
      output_chunk_length: 12
      hidden_size: 128
      lstm_layers: 3
      num_attention_heads: 4
      full_attention: false
      dropout: 0.1
      loss_fn: null
      likelihood:
        _target_: darts.utils.likelihood_models.QuantileRegression
        quantiles:
        - 0.01
        - 0.05
        - 0.1
        - 0.15
        - 0.2
        - 0.25
        - 0.3
        - 0.4
        - 0.5
        - 0.6
        - 0.7
        - 0.75
        - 0.8
        - 0.85
        - 0.9
        - 0.95
        - 0.99
      batch_size: 512
      add_relative_index: false
      add_encoders:
        cyclic:
          future:
          - month
        datetime_attribute:
          future:
          - dayofweek
        transformer:
          _target_: darts.dataprocessing.transformers.Scaler
      random_state: 42
      model_name: model_tft
      work_dir: outputs/64_lstm_3
      save_checkpoints: true
      log_tensorboard: true
      torch_metrics:
        _target_: torchmetrics.MetricCollection
        metrics:
        - _target_: torchmetrics.MeanAbsolutePercentageError
      optimizer_kwargs:
        lr: 0.001
      pl_trainer_kwargs:
        callbacks:
        - _target_: pytorch_lightning.callbacks.EarlyStopping
          monitor: val_MeanAbsolutePercentageError
          patience: 150
          min_delta: 0.005
          verbose: true
          mode: min
        - _target_: pytorch_lightning.callbacks.ModelCheckpoint
          monitor: val_MeanAbsolutePercentageError
          verbose: true
    data:
      filters:
        customer_values:
        - ARGALYS
        - LES MIRACULEUX
        - MINCI DELICE
        - NUTRAVANCE
        builder_arguments:
          CUSTOMER:
          - ARGALYS
          - LES MIRACULEUX
          - MINCI DELICE
          - NUTRAVANCE
          _measurement:
          - customer_quantity
          _field:
          - QUANTITY
    train_parameters:
      epochs: 600
      random_state: 42
    test_parameters:
      'n': 23
      test_size: 0.2
      num_samples: 100
    torch_metrics:
      _target_: torchmetrics.MetricCollection
      metrics:
      - _target_: torchmetrics.MeanAbsolutePercentageError
    pl_trainer_kwargs:
      callbacks:
      - _target_: pytorch_lightning.callbacks.EarlyStopping
        monitor: val_MeanAbsolutePercentageError
        patience: 150
        min_delta: 0.005
        verbose: true
        mode: min
      - _target_: pytorch_lightning.callbacks.ModelCheckpoint
        monitor: val_MeanAbsolutePercentageError
        verbose: true

  ```

</details>


## Finetuning

Pout finetuner un modèle il suffit de lancer la procédure d'entrainement en spécifiant
le nom du modèle et le dossier d'un checkpoint préalablement sauvegardé.

Par exemple pour reprendre l'entrainement du modèle précédement entrainé il suffit de
lancer la commande `python run.py mode=train +experiments=tft checkpoint.work_dir=outputs/64_lstm_3`.


## Prediction

De même que pour l'eval et le fine-tuning il suffit d'appeler le script `run.py` avec
`mode` mis en "predict" et de spécifier le dossier du checkpoint du modèle à utiliser
avec la variable `checkpoint.work_dir`.

Par exemple pour prédire avec le modèle entrainé précédement il suffit de lancer la
commande ` python run.py mode=predict +experiments=tft checkpoint.work_dir=outputs/64_lstm_3`
