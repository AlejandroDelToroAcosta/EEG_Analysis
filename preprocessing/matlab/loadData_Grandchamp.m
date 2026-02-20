function eeg = loadData_Grandchamp_MOD(sub, session)
% Carga el archivo final limpio (ica_a2.set) para el análisis de features.
% Divide los ensayos en On-Task (128) y Mind Wandering (4).

% --- Configuración de la ruta ---
f_dat = fullfile('preprocessing'); 

% --- 1. Cargar el EEG Limpio Final ---
EEG_raw = pop_loadset(fullfile(f_dat, [num2str(sub),'_', num2str(session), '_epochs_ica_a2.set']));

% Inicializar etiquetas
labels = [];
dat = [];

% --- 2. Epocar Condición On-Task (OT / Focused) ---
% Se alinea con el Estímulo (128). Ventana: [0 8] segundos.
% Nota: El nombre '128' debe coincidir con el tipo del trigger en EEGLAB.
EEG_ot = pop_epoch(EEG_raw, {'128'}, [0 8]); 
dat = EEG_ot.data;
labels = [labels repelem([0], EEG_ot.trials)];

% --- 3. Epocar Condición Mind Wandering (MW) ---
% Se alinea con la Respuesta de MW (4). Ventana: [-10 -2] segundos.
EEG_mw = pop_epoch(EEG_raw, {'4'}, [-10 -2]); 

% Concatenar los datos de MW a los datos de OT
dat(:,:, end+1:end+size(EEG_mw.data,3)) = EEG_mw.data;
labels = [labels repelem([1], EEG_mw.trials)];

% --- 4. Crear la estructura de salida 'eeg' ---
eeg.data    = dat;
eeg.labels  = labels;
eeg.chanlocs = EEG_ot.chanlocs; % Usamos la estructura de un set
eeg.srate   = EEG_ot.srate;
eeg.times = EEG_ot.times;
end