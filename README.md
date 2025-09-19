# DeepBP Transformer

Questo repository contiene l'implementazione del modello **DelayAndSumTransformer** (precedentemente conosciuto come BPTransformer) per la ricostruzione di immagini a partire da sinogrammi.

## Dashboard interattiva

È disponibile una piccola dashboard Streamlit per visualizzare sinogrammi, ricostruzioni intermedie e output finali del modello:

```bash
streamlit run tools/dashboard_app.py
```

La dashboard utilizza i percorsi definiti in `TrainConfig` (in `main.py`) per individuare dataset e checkpoint. Se nella cartella `work_dir` configurata sono presenti i file `best.pt` o `last.pt`, verranno proposti per il caricamento automatico dei pesi.

Puoi selezionare un file sinogramma già presente nei percorsi configurati oppure caricare un nuovo file `.hdf5`. Il parsing dei dati riusa la logica del dataset (`HDF5Dataset`) e mostra sia il sinogramma normalizzato sia la ricostruzione Delay-and-Sum e l'output del ViT.

Assicurati di installare le dipendenze necessarie (Streamlit, PyTorch, nibabel, h5py, matplotlib, ecc.) prima di avviare la dashboard.
