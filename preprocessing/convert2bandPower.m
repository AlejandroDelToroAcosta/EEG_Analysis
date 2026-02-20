function convert2bandPower(studyId, sub) 
% Esta función calcula POTENCIA y ISPC simultáneamente para un sujeto.

% --- 1. CONFIGURACIÓN DE RUTA Y PARÁMETROS FIJOS ---
path = 'C:\Users\aadel\Desktop\GCID\Cuarto\Segundo Cuatrimestre\TFG\matlab'; 
f_main = path;
% El output de ISPC se guardará en una subcarpeta diferente a la de potencia para evitar mezclas
f_output_power = [f_main, filesep, 'inputs_matfile', filesep]; 
f_output_ispc = [f_main, filesep, 'inputs_matfile_ispc', filesep]; 
cd(f_main)

% Parámetros generales
condcell = {'OT', 'MW'};
session_id = 1; 
baseline  = [-1000 0];
stimOn    = [0 1000]; 
checkOn = 0; % Parámetro para waveletConv

% Parámetros específicos para ISPC
winlenRange = [1.5, 3.5]; % Valor por defecto para computeISPC
ISPC_chan_names = {'A10', 'B7'}; % Par de canales para calcular ISPC (ej: Occipital Bilateral)

% --- 2. CARGA DE PARÁMETROS y SUJETO ---
load('pars_markers.mat', 'bands'); 

if studyId == 2 
    eeg = loadData_Grandchamp(sub, session_id);
else
    error('studyId incorrecto.');                
end

times = eeg.times;
srate = eeg.srate; 
baseIdx   = dsearchn(times', baseline');
stimOnIdx = dsearchn(times', stimOn');

% --- 4. BUCLE SOBRE BANDAS ---
for id = 1:length(bands)
    measure = bands(id).name;
    bandrange = bands(id).range;
    chans = bands(id).chans;
    

    % --- 5. CÁLCULO DE WAVELET Y ALMACENAMIENTO DE COEFICIENTES ---
    
    % Parámetros Wavelet
    nFreq = 20;            
    nCycleRange = [4 8];   
    scaling = 'log';       
    
    % Inicializar matrices de salida
    powerAllChans = zeros(length(chans), size(eeg.data, 2), size(eeg.data, 3)); 
    % ¡ALMACENAR COEFICIENTES COMPLEJOS PARA ISPC!
    all_complex_coeffs = cell(1, length(chans));

    % Bucle sobre los canales para convolución (para Potencia y para ISPC)
    for chani_idx = 1:length(chans)
        
        % 1. Extraer y Convolucionar
        data_single_chan = squeeze(eeg.data(chani_idx, :, :)); % nPnt x nTrial
        convres = waveletConv(data_single_chan, srate, bandrange, nFreq, nCycleRange, scaling, checkOn, 0, 0, 0, 0); 
        
        % 2. Almacenar coeficientes complejos para uso posterior en ISPC
        all_complex_coeffs{chani_idx} = convres.data; % Freq x Pnt x Trial
        
        % 3. Cálculo de la Potencia y Promediado en Frecuencia
        power_map_tf = abs(convres.data).^2; 
        power_band_avg = squeeze(mean(power_map_tf, 1)); % 1 x Pnt x Trial
        
        % 4. Registrar el resultado para el canal actual
        powerAllChans(chani_idx, :, :) = power_band_avg;
        
    end % Fin Bucle de Canales para convolución
    
    % -----------------------------------------------------------------
    % 6. ANÁLISIS 1: POTENCIA (Guardado por canal y condición)
    % -----------------------------------------------------------------
    for chani = 1:length(chans)
        chan = chans{chani};
        power_chan = squeeze(powerAllChans(chani, :, :)); % Tiempo x Ensayo
        
        for condi = 1:length(condcell)
            cond = condcell{condi};
            cond_label = condi - 1; 

            trial_indices = find(eeg.labels == cond_label);
            power_cond_trials = power_chan(:, trial_indices); 

            if isempty(power_cond_trials), continue; end
            
            % Promediado Temporal (Base y Estímulo)
            base_power = mean(power_cond_trials(baseIdx(1):baseIdx(end), :), 1); 
            stimOn_power = mean(power_cond_trials(stimOnIdx(1):stimOnIdx(end), :), 1); 

            data_to_save = [base_power(:), stimOn_power(:)]; 
            
            % Guardado
            varName = [measure, '_', lower(cond)]; 
            file = [f_output_power, measure, '_', chan, filesep, num2str(sub), '.mat'];
            
            dir_to_create = [f_output_power, measure, '_', chan];
            if exist(dir_to_create, 'dir') ~= 7; mkdir(dir_to_create); end

            eval([varName, '= data_to_save;']);
            if exist(file, 'file') == 2
                save(file, varName, '-append');
            else
                save(file, varName);
            end
            

        end % End condi loop (Potencia)
    end  % End chani loop (Potencia)
    
    % -----------------------------------------------------------------
    % 7. ANÁLISIS 2: ISPC (Coherencia)
    % -----------------------------------------------------------------
    
    % 1. Encontrar los índices de los canales A10 y B7 en el array 'chans'
    idx1 = find(strcmpi(chans, ISPC_chan_names{1}), 1);
    idx2 = find(strcmpi(chans, ISPC_chan_names{2}), 1);
    
    if isempty(idx1) || isempty(idx2)
        warning(['ISPC: Uno o ambos canales (', ISPC_chan_names{1}, '/', ISPC_chan_names{2}, ') no están en la lista de análisis. Omitiendo ISPC.']);
        continue;
    end
    
    % 2. Recuperar coeficientes
    data1 = all_complex_coeffs{idx1}; % A10
    data2 = all_complex_coeffs{idx2}; % B7
    frex_wav = convres.frex; % El vector de frecuencias es el mismo

    % 3. CÁLCULO DE ISPC
    % computeISPC(data1, data2, frex, times, winlenRange, byTrialOn, baseline, plotOn, pbOn, parsPrintOn, gpu2use)
    ISPC_tf = computeISPC(data1, data2, frex_wav, times, winlenRange, 0, baseline, 0, 1, 0, 0); 
    
    % 4. Promediado en Frecuencia (en las 20 frecuencias de la banda)
    ISPC_band_avg = squeeze(mean(ISPC_tf, 1)); % Resultado: Tiempo x Ensayo
    
    % 5. Bucle sobre condiciones y Guardado de ISPC
    for condi = 1:length(condcell)
        cond = condcell{condi};
        cond_label = condi - 1; 

        trial_indices = find(eeg.labels == cond_label);
        ISPC_cond_trials = ISPC_band_avg(:, trial_indices); % Tiempo x Ensayos_Cond
        
        if isempty(ISPC_cond_trials), continue; end
        
        % Promediado Temporal (Estímulo)
        stimOn_ispc = mean(ISPC_cond_trials(stimOnIdx(1):stimOnIdx(end), :), 1); % 1 x Ensayos_Cond
        
        % Guardado de ISPC
        varName = [measure, '_', lower(cond), '_ispc']; 
        file_name = [num2str(sub), '_', ISPC_chan_names{1}, '_', ISPC_chan_names{2}, '.mat'];
        file = [f_output_ispc, measure, filesep, file_name];
        
        dir_to_create = [f_output_ispc, measure];
        if exist(dir_to_create, 'dir') ~= 7; mkdir(dir_to_create); end

        eval([varName, '= stimOn_ispc(:);']);
        if exist(file, 'file') == 2
            save(file, varName, '-append');
        else
            save(file, varName);
        end
        
                
    end % End condi loop (ISPC)
    
end % End bands loop

end % func