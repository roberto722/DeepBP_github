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

## Dashboard interattiva

È disponibile anche una piccola dashboard Streamlit per visualizzare sinogrammi, ricostruzioni intermedie e output finali del modello:

```bash
streamlit run tools/dashboard_app.py
```

La dashboard utilizza i percorsi definiti in `TrainConfig` (in `main.py`) per individuare dataset e checkpoint. Se nella cartella `work_dir` configurata sono presenti i file `best.pt` o `last.pt`, verranno proposti per il caricamento automatico dei pesi.

Puoi selezionare un file sinogramma già presente nei percorsi configurati oppure caricare un nuovo file `.hdf5`. Il parsing dei dati riusa la logica del dataset (`HDF5Dataset`) e mostra sia il sinogramma normalizzato sia la ricostruzione Delay-and-Sum e l'output del ViT.

Assicurati di installare le dipendenze necessarie (Streamlit, PyTorch, nibabel, h5py, matplotlib, ecc.) prima di avviare la dashboard.

## Normalizzazione dei target

Di default il dataset scala sinogrammi e immagini di riferimento nell'intervallo
[0, 1] utilizzando le statistiche globali configurate (`sino_min/sino_max` e
`img_min/img_max`). È ora possibile disattivare la normalizzazione delle sole
immagini impostando `TrainConfig.normalize_targets = False`. In questo modo i
target vengono restituiti nel loro range originale (ad esempio 0–320) mentre i
sinogrammi continuano a essere normalizzati.

Disattivare la normalizzazione modifica la scala della loss L1 (i valori
risulteranno più grandi in proporzione al nuovo range) e dei pesi basati
sull'intensità. Le soglie `weight_threshold` e `ssim_mask_threshold` operano
sempre sui valori prodotti dal dataset: se si usano target non normalizzati è
quindi consigliabile aggiornarle esplicitamente in base alle nuove unità.
