import mne
import numpy as np
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
import os


def apply_cwt(subject, session):
    # Ajuste de ruta (usando string formateado)
    data_path = f'C:\\Users\\aadel\\Desktop\\GCID\\Cuarto\\Segundo Cuatrimestre\\TFG\\python\\preprocessing\\session_{session}\\{subject}_0{session}_epochs_ica_a2-epo.fif'

    if not os.path.exists(data_path):
        print(f"⚠️ Salto: El archivo para el sujeto {subject} en sesión {session} no existe.")
        return False

    try:
        # 1. Cargar Epochs
        epochs = mne.read_epochs(data_path, preload=True, verbose=False)

        # 2. Definir objetivos y analizar canales disponibles
        target_channels = ['A25', 'A31', 'B30', 'B6', 'EXG1']
        all_channels = epochs.ch_names
        bad_channels = epochs.info['bads']

        # Canales que están físicamente en el archivo y NO son "bads"
        good_channels_available = ['A13', 'A11', 'A21', 'A12', 'B26']

        final_selection = []

        # 3. Lógica de Sustitución Dinámica
        for ch in target_channels:
            if ch in good_channels_available:
                # Si el canal objetivo está sano, se queda
                final_selection.append(ch)
            else:
                # Si es "bad" o no existe, buscamos un sustituto
                # El sustituto debe estar sano y no estar ya en nuestra lista de objetivos
                substitutes = [c for c in good_channels_available
                               if c not in target_channels and c not in final_selection]

                if substitutes:
                    new_ch = substitutes[0]  # Tomamos el primero disponible
                    final_selection.append(new_ch)
                    print(f"🔄 Sujeto {subject}: Sustituyendo {ch} (malo) por {new_ch} (sano)")
                else:
                    print(f"❌ Error crítico: No hay suficientes canales sanos para el sujeto {subject}")
                    return False

        # 4. Aplicar la selección final de 5 canales
        epochs.pick(final_selection)
        epochs.reorder_channels(final_selection)

        # 5. Procesamiento CWT
        freqs = np.linspace(4, 40, num=40)
        n_cycles = freqs / 2.

        tfr = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                         return_itc=False, average=False, verbose=False)

        # Extraer potencia y aplicar log-transform
        x_data = np.abs(tfr.data) ** 2
        x_data = np.log10(x_data + 1e-10)

        # 6. Normalización (Z-score por canal y frecuencia)
        mean = x_data.mean(axis=(0, 3), keepdims=True)
        std = x_data.std(axis=(0, 3), keepdims=True)
        x_data = (x_data - mean) / (std + 1e-10)

        y_labels = epochs.events[:, 2]

        # 7. Guardar resultados
        save_name = f'../ml_data/{subject}_0{session}_cwt_data.npz'
        np.savez_compressed(save_name,
                            power=x_data.astype(np.float32),
                            label=y_labels,
                            frex=freqs,
                            times=epochs.times,
                            ch_names=final_selection)

        print(f"✅ {subject}_0{session} guardado. Canales usados: {final_selection}")
        return True

    except Exception as e:
        print(f"❌ Error procesando Sujeto {subject}: {e}")
        return False

def plot_subject_scalograms(file_path, subject_id):
    # 1. Cargar los datos guardados
    data = np.load(file_path)
    x_data = data['power']  # Shape: (epochs, channels, freqs, times)
    y_labels = data['label']
    freqs = data['frex']
    times = data['times']
    ch_names = list(data['ch_names'])

    # 2. Separar por clases y promediar (Grand Average del Sujeto)
    class_0_mean = x_data[y_labels == 0].mean(axis=0)  # On-Task
    class_1_mean = x_data[y_labels == 1].mean(axis=0)  # Mind-Wandering

    # Seleccionamos un canal clave (Pz es el índice 1 si usaste ['PO7', 'Pz', 'PO8'])
    ch_idx = ch_names.index('A19') if 'A19' in ch_names else 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Definir límites de color para que la comparación sea justa
    vmin, vmax = np.percentile(x_data, [5, 95])

    # Plot Clase 0
    im0 = axes[0].imshow(class_0_mean[ch_idx], aspect='auto', origin='lower',
                         extent=[times[0], times[-1], freqs[0], freqs[-1]],
                         cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Sujeto {subject_id} - ON-TASK (Promedio)")
    axes[0].set_ylabel("Frecuencia (Hz)")
    axes[0].set_xlabel("Tiempo (s)")

    # Plot Clase 1
    im1 = axes[1].imshow(class_1_mean[ch_idx], aspect='auto', origin='lower',
                         extent=[times[0], times[-1], freqs[0], freqs[-1]],
                         cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Sujeto {subject_id} - MIND-WANDERING (Promedio)")
    axes[1].set_xlabel("Tiempo (s)")

    # Añadir barra de color
    fig.colorbar(im1, ax=axes.ravel().tolist(), label='Potencia (dB)')
    plt.suptitle(f"Análisis Tiempo-Frecuencia en Canal {ch_names[ch_idx]}", fontsize=16)
    plt.show()
