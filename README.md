# audio_recognition_bot

## Step1. Collect Dataset

To start collecting the dataset, run

```
python3 audio_digits_dataset_bot.py
```

Telegram bot will issue 5 random numbers that need to be dictated with pauses

## Step 2. Training model

In order to train the module and get ```model.pkl```, you need to run the ```ml.ipynb``` in jupyter notebook

```
jupyter notebook ml.ipynb
```

## Step 3. Run telegram bot
```
python3 prog.py
```
