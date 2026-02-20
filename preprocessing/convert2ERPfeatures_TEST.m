function convert2ERPfeatures_TEST_MOCK(studyId, sub)
% MODO MOCK: Genera datos aleatorios simulando las 9 características ERP
% para probar el resto del pipeline (guardado, estructura) sin ejecutar 
% computeWst, que es lenta y tiene errores de compatibilidad.

% --- 1. CONFIGURACIÓN ---
path = 'C:\Users\aadel\Desktop\GCID\Cuarto\Segundo Cuatrimestre\TFG\matlab'; 
f_main = path;
f_output = [f_main, filesep, 'inputs_matfile_erp', filesep]; 
cd(f_main)

% Parámetros de Análisis
condcell = {'OT', 'MW'};
session_id = 1;
load('pars_markers.mat', 'bands'); 

% Canales de Interés para ERP
chans_erp_full = bands(1).chans; 

% --- 2. CONFIGURACIÓN DE LÍMITE PARA PRUEBAS RÁPIDAS ---
N_CHANS_TEST = 3;   % PROCESAR SOLO LOS PRIMEROS N CANALES
N_TRIALS_TEST = 10; % PROCESAR SOLO N ENSAYOS POR CONDICIÓN (OT y MW)

% Seleccionar un subconjunto de canales
if length(chans_erp_full) > N_CHANS_TEST
    chans_erp = chans_erp_full(1:N_CHANS_TEST);
else
    chans_erp = chans_erp_full;
end

% --- 3. CARGA DE DATOS DEL SUJETO ---
if studyId == 2 
    eeg = loadData_Grandchamp(sub, session_id); 
else
    error('studyId incorrecto.');                
end

times = eeg.times;

% --- 4. INICIALIZACIÓN ---
n_chans = length(chans_erp);
n_features = 9; 
% El número total de iteraciones se basará en el límite de ensayos
n_total = n_chans * N_TRIALS_TEST * length(condcell); 
i = 0;

fprintf('Iniciando MOCK de características ERP para Sujeto %d...\n', sub);
fprintf('  Generando datos simulados para %d canales y %d ensayos por condición.\n', n_chans, N_TRIALS_TEST);

% --- 5. BUCLE SOBRE CANALES (SUBSET) ---
for chani = 1:n_chans
    chan = chans_erp{chani};
    
    % Inicializar una matriz temporal para registrar las features de este canal
    features_temp = zeros(0, 10); % Columna 1 para label + 9 features
    
    % --- 6. BUCLE SOBRE CONDICIONES Y ENSAYOS (LIMITADO) ---
    for condi = 1:length(condcell)
        cond_label = condi - 1; % 0=OT, 1=MW
        
        % Obtener índices de todos los ensayos para esta condición
        all_cond_indices = find(eeg.labels == cond_label);
        
        % Limitar a N_TRIALS_TEST ensayos
        trials_to_process = all_cond_indices(1:min(length(all_cond_indices), N_TRIALS_TEST));
        
        for trial_index = trials_to_process'
            
            % !!! MOCK DE DATOS AQUI !!!
            % ----------------------------------------------------
            % SIMULAMOS LAS 9 CARACTERÍSTICAS (W, t, s para P1, N1, P3)
            % Se generan 9 números aleatorios entre 0 y 1.
            features = rand(1, n_features); 
            % ----------------------------------------------------
            
            % 1. Registro: Condición + Features
            new_row = [cond_label, features];
            features_temp = [features_temp; new_row];
            
            % --- ACTUALIZACIÓN DE PROGRESO ---
            i = i + 1;
            pct = (i/n_total) * 100;
            fprintf('\rProceso MOCK: %s (%s) | Progreso total: %3.0f%%', chan, condcell{condi}, pct);
        end 
    end % Fin Bucle Condición

    % --- 7. GUARDADO (Separado por Condición y Canal) ---
    
    for condi = 1:length(condcell)
        cond = condcell{condi};
        cond_label = condi - 1;
        
        cond_indices = features_temp(:, 1) == cond_label;
        data_to_save = features_temp(cond_indices, 2:10);
        
        if isempty(data_to_save), continue; end

        % Guardado
        varName = ['erp_features_', lower(cond)]; 
        file = [f_output, 'erp_', chan, filesep, num2str(sub), '.mat'];
        
        dir_to_create = [f_output, 'erp_', chan];
        if exist(dir_to_create, 'dir') ~= 7; mkdir(dir_to_create); end

        eval([varName, '= data_to_save;']);
        if exist(file, 'file') == 2
            save(file, varName, '-append');
        else
            save(file, varName);
        end
    end

end % Fin Bucle Canales

fprintf('\nProceso MOCK completado para el Sujeto %d. Archivos generados en %s\n', sub, f_output);

end % Fin función