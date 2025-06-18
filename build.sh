#!/bin/bash
BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS_DIR="$BASE_PATH/utils/datasets"
MODEL_FILE="$BASE_PATH/model.pkl"
ZIP_FILE="$BASE_PATH/zipped_data/data.zip"
FULL_ZIP="$BASE_PATH/zipped_data/data_unified.zip"
SWMM_BIN="$BASE_PATH/SWMM/swmm5-ubuntu-cli/swmm5"
SWMM_SRC="$BASE_PATH/SWMM/swmm5-ubuntu-cli/Stormwater-Management-Model/src"
MAKEFILE="../../Makefile"

if [[ -z "$ZIP_FILE" || -z "$FULL_ZIP" ]]; then
    echo "Errore nei path ZIP: controlla le variabili ZIP_FILE o FULL_ZIP"
    exit 1
fi

if [[ -d "$DATASETS_DIR" && -f "$MODEL_FILE" ]]; then
    echo "Tutto già presente. Nessuna estrazione necessaria."
else
    echo "File o cartella mancanti. Procedo con l'estrazione da ZIP multi-volume..."

    if [[ -f "$ZIP_FILE" ]]; then
        echo "Ricombinazione dei volumi ZIP in $FULL_ZIP..."
        zip -s 0 --out "$FULL_ZIP" "$ZIP_FILE"

        if [[ $? -ne 0 || ! -f "$FULL_ZIP" ]]; then
            echo "Errore durante la ricombinazione dei volumi ZIP"
            exit 1
        fi

        echo "Estrazione da $FULL_ZIP..."
        unzip -o "$FULL_ZIP" -d "$BASE_PATH"

        if [[ $? -eq 0 ]]; then
            echo "Estrazione completata con successo."

            echo "Pulizia: rimozione $FULL_ZIP..."
            rm -f "$FULL_ZIP"

            # Pulizia ZIP indesiderato
            if [[ -f "$BASE_PATH/.zip" ]]; then
                echo "Rimozione di uno ZIP anomalo trovato in $BASE_PATH/.zip"
                rm -f "$BASE_PATH/.zip"
            fi
        else
            echo "Estrazione fallita."
            exit 1
        fi
    else
        echo "Archivio ZIP non trovato: $ZIP_FILE"
        exit 1
    fi
fi

if [ -f "$SWMM_BIN" ]; then
    echo "SWMM già compilato: $SWMM_BIN"
else
    echo "Compilazione di SWMM in corso..."
    (cd "$SWMM_SRC" && make -f "$MAKEFILE")

    if [ -f "$SWMM_BIN" ]; then
        echo "Compilazione riuscita: swmm5 creato."
    else
        echo "Errore: compilazione fallita o binario mancante."
        exit 1
    fi

    cd $BASE_PATH
fi