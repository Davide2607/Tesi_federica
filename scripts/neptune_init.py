import neptune

# Inizializza Neptune
def init_neptune():
# Inizializza Neptune
    run = neptune.init_run(
        project="",  # Sostituisci con il tuo project name
        api_token=""            # Sostituisci con il tuo API token
    )
    return run
