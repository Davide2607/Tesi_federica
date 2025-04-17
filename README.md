# Final Scripts - Facial Expression Recognition

Questa cartella contiene gli script finali utilizzati per il training, il fine-tuning e la generazione delle mappe di attenzione nel progetto di tesi.

## 📁 Struttura della cartella

---

## 📂 `scripts/`: Script Python principali

Questa cartella contiene tutti gli script organizzati per funzione:

### 📊 Analisi e visualizzazione
- `cm_images.py`: genera le confusion matrix a partire dai risultati dei modelli.
- `neptune_init.py`:  tracciamento degli esperimenti con Neptune.ai.

### 🧠 Training e fine-tuning dei modelli
- `train_final_layers.py`: script principale per il training finale dei modelli.
- `finetuning.py`: script per il fine-tuning di modelli pre-addestrati.
- `optimizer_final_layers.py`: definisce l'ottimizzazione standard per la ricerca dei migliori parametri per l'addestramento
dei final layers.
- `optimizer_finetuning.py`: definisce ottimizzatori specifici per il fine-tuning.
- `find_unfreeze.py`: sblocca layer specifici di modelli pre-addestrati per il fine-tuning e testa qual è il numero migliore di layer da scongelare.

### 🛠️ Generazione e gestione dei dati
- `data_generators.py`: contiene le classi per la generazione dinamica dei dati durante il training.
- `loading_data.py`: gestisce il caricamento delle immagini.

### 🧪 Loss e metriche
- `losses.py`: definizione delle funzioni di loss personalizzate.

### 🔁 Test e sperimentazioni varie
- `find_truncated_layers.py`: script sperimentale, probabilmente legato a test o saliency map.
- `pattlite_prova.py`: versione di prova dello script precedente.

---

## 📂 `sbatch/`
Contiene gli script per il lancio dei job su cluster HPC tramite SLURM.
---

## ℹ️ Note
- Tutti gli script sono scritti in Python e organizzati per una pipeline modulare e riutilizzabile.
- Gli script relativi a ottimizzazione, fine-tuning e data loading sono separati per facilitare il controllo delle sperimentazioni.

