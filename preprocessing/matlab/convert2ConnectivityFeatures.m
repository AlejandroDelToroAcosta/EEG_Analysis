function convert2ConnectivityFeatures(studyId, sub)
% Extrae las características de Power e ISPC para las bandas Alpha y Theta.
% Llama a 'extractConnectivityFeatures' para el procesamiento y guardado.

% --- 1. CONFIGURACIÓN ---
path = 'C:\Users\aadel\Desktop\GCID\Cuarto\Segundo Cuatrimestre\TFG\matlab'; 
f_main = path;
% Carpeta de salida para las características de conectividad/frecuencia
f_output = [f_main, filesep, 'inputs_matfile_conn', filesep]; 
cd(f_main)

% Parámetros de Análisis
condcell = {'OT', 'MW'};
session_id = 1;
% Definición de las bandas y canales clave
chans_connect = {'A10', 'A19', 'B7', 'C21'};
bands = struct('name', {'theta', 'alpha'}, 'range', {[4 8], [8.5 12]});

% --- 2. CARGA DE DATOS DEL SUJETO ---
if studyId == 2 
    % Asumimos que loadData_Grandchamp_MOD carga la matriz eeg.data (Canales x Puntos x Ensayos)
    eeg = loadData_Grandchamp(sub, session_id); 
else
    error('studyId incorrecto.');                
end

% Añadir la tasa de muestreo si no está en la estructura EEG (CRÍTICO para los filtros)
if ~isfield(eeg, 'srate')
    % Se asume el srate de 256 Hz que mencionaste
    eeg.srate = 256; 
end


% --- 3. EXTRACCIÓN DE CARACTERÍSTICAS ---
% all_features_out = [n_trials x (1 + N_Features_Totales)]
% --- 3. EXTRACCIÓN DE CARACTERÍSTICAS (MOCK) ---
all_features_out = extractConnectivityFeatures_MOCK(eeg, chans_connect, bands);
% --- 4. GUARDADO (Separado por Condición) ---
% El archivo se guardará en la carpeta 'inputs_matfile_conn' 
% con el nombre del sujeto.

fprintf('\nIniciando guardado de características de conectividad...\n');

% Asegurar que la carpeta de salida exista
if exist(f_output, 'dir') ~= 7; mkdir(f_output); end
file = [f_output, num2str(sub), '.mat'];

% Loop sobre condiciones para separar y guardar los datos
for condi = 1:length(condcell)
    cond = condcell{condi};
    cond_label = condi - 1; % 0=OT, 1=MW
    
    % Filtrar los ensayos de la condición actual
    cond_indices = all_features_out(:, 1) == cond_label;
    % Seleccionamos solo las columnas de características (a partir de la 2)
    data_to_save = all_features_out(cond_indices, 2:end); 
    
    if isempty(data_to_save), continue; end

    % Crear la variable con el nombre dinámico (ej: conn_features_ot)
    varName = ['conn_features_', lower(cond)]; 

    eval([varName, '= data_to_save;']);
    
    % Guardar o adjuntar los datos en el archivo del sujeto
    if exist(file, 'file') == 2
        save(file, varName, '-append');
    else
        save(file, varName);
    end
    fprintf('  -> Guardado %s (%d ensayos).\n', varName, size(data_to_save, 1));
end

fprintf('Proceso de conectividad completado para el Sujeto %d. Archivo guardado en %s\n', sub, f_output);

end