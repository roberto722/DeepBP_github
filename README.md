# DeepBP Transformer

Questo repository contiene l'implementazione del modello **DelayAndSumTransformer** (precedentemente conosciuto come BPTransformer) per la ricostruzione di immagini a partire da sinogrammi.

## Script di backprojection lineare

Per verificare rapidamente la qualità della backprojection sui sinogrammi HDF5 è
disponibile lo script CLI `tools/backprojection_linear.py`, che replica in
NumPy/SciPy le funzioni MATLAB `backproject_waveeq_linear` e `bp_kernel` e
riutilizza la geometria della sonda definita in `TrainConfig`.

Esempio di utilizzo (con i percorsi di default di `TrainConfig`):

```bash
python tools/backprojection_linear.py \
    --input sample.hdf5 \
    --data-root /path/to/datasets_transformer_proj \
    --sino-dir Forearm2000_hdf5/train_val_tst \
    --recs-dir Forearm2000_recs/L1_Shearlet \
    --split val \
    --summary
```

Lo script accetta molteplici override della geometria (`--n_det`, `--dt_s`,
`--nx`, ecc.), consente di memorizzare su disco il kernel di
backprojection (`--kernel-cache`) e, se necessario, applica un filtro PSF
(`--psf-path` oppure `--psf-size/--psf-sigma`). Il risultato viene salvato in
formato `.npy` (ed eventualmente in `.png`), mentre l'opzione `--summary`
stampa un riepilogo JSON con statistiche basilari dell'immagine ricostruita.

> **Dipendenze.** È richiesta l'installazione di `scipy` per il calcolo della
> funzione di Bessel di Hankel; opzionalmente, la presenza di PyTorch permette
> di riutilizzare direttamente il loader del dataset già presente nel training
> loop.

## Script di inference sul dataset

Per eseguire l'inferenza su un intero split del dataset e salvare metriche e
output su disco è disponibile lo script `tools/inference_dataset.py`.

Esempio (usando il checkpoint migliore e i percorsi di default in `TrainConfig`):

```bash
python tools/inference_dataset.py \
    --checkpoint /path/to/best.pt \
    --split val \
    --output-dir ./inference_outputs \
    --save-png \
    --intermediate-indices 0,1,2
```

Lo script salva un file `metrics.json` con le metriche aggregate e per-sample,
oltre alle predizioni in formato `.npy` e, se richiesto, PNG affiancati
(`Initial | intermediates | Pred | GT`).

Per usare la configurazione salvata nella run di training (utile in PyCharm),
puoi passare `--run-dir` oppure `--config-path`:

```bash
python tools/inference_dataset.py \
    --checkpoint /path/to/best.pt \
    --run-dir /path/to/run_dir \
    --split val
```

## Dashboard interattiva

È disponibile anche una piccola dashboard Streamlit per visualizzare sinogrammi, ricostruzioni intermedie e output finali del modello:

```bash
streamlit run tools/dashboard_app.py
```

La dashboard utilizza i percorsi definiti in `TrainConfig` (in `main.py`) per individuare dataset e checkpoint. Se nella cartella `work_dir` configurata sono presenti i file `best.pt` o `last.pt`, verranno proposti per il caricamento automatico dei pesi.

Puoi selezionare un file sinogramma già presente nei percorsi configurati oppure caricare un nuovo file `.hdf5`. Il parsing dei dati riusa la logica del dataset (`HDF5Dataset`) e mostra sia il sinogramma normalizzato sia la ricostruzione Delay-and-Sum e l'output del ViT.

Assicurati di installare le dipendenze necessarie (Streamlit, PyTorch, nibabel, h5py, matplotlib, ecc.) prima di avviare la dashboard.
