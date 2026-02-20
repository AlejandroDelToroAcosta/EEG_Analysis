function all_features_out = extractConnectivityFeatures_MOCK(eeg, chans_connect, bands)
% MOCK: SIMULA la extracción de características de Power e ISPC.
% Genera un vector de 40 características aleatorias por ensayo.

    % --- 1. CONFIGURACIÓN ---
    n_trials = size(eeg.data, 3);
    
    % Contamos el número de características esperadas (igual que la versión real)
    n_chans = length(chans_connect); % 4 Canales
    n_bands = length(bands);         % 2 Bandas (Theta, Alpha)
    n_windows = 2;                   % 2 Ventanas (Baseline, Post-stim)
    
    % Número de pares (ISPC) = n * (n - 1) / 2 -> 4 * 3 / 2 = 6 pares
    n_pairs = n_chans * (n_chans - 1) / 2;
    
    % Cálculo del número total de características por ensayo:
    % Power (4 Canales * 2 Bandas * 2 Ventanas) = 16
    n_power_feats = n_chans * n_bands * n_windows;
    % ISPC (6 Pares * 2 Bandas * 2 Ventanas) = 24
    n_ispc_feats = n_pairs * n_bands * n_windows;
    
    features_per_trial = n_power_feats + n_ispc_feats; % Total: 40 características
    
    % --- 2. GENERACIÓN DE DATOS MOCK ---

    fprintf('\n[MOCK MODE] Generando datos de Power/ISPC simulados (%d caracteristicas/ensayo)...\n', features_per_trial);
    
    % Generar una matriz de datos aleatorios [n_trials x features_per_trial]
    % Usamos rand() para que los valores estén entre 0 y 1 (similar a ISPC).
    mock_data = rand(n_trials, features_per_trial);
    
    % --- 3. CONSTRUCCIÓN DE LA SALIDA ---
    
    % La salida debe ser [n_trials x (1 + N_Features)]
    all_features_out = zeros(n_trials, features_per_trial + 1);
    
    % Columna 1: Etiquetas de condición
    all_features_out(:, 1) = eeg.labels;
    
    % Columnas 2 en adelante: Datos simulados
    all_features_out(:, 2:end) = mock_data;
    
    fprintf('MOCK completado. Datos listos para guardado.\n');

end % Fin función MOCK