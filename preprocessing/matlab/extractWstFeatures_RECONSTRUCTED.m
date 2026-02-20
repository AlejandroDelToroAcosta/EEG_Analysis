function features = extractWstFeatures_RECONSTRUCTED(Wst, tlag, scale)
% Reconstrucción de la función de extracción de picos P1, N1, P3 de la WST.
% Utiliza rangos de tiempo proporcionados: P1[50-150], N1[100-200], P3[250-600].

    % RANGOS DE BÚSQUEDA (RANGOS DE TIEMPO AJUSTADOS)
    % NOTA: Los rangos de Escala (Scale) se han estimado con base en la literatura 
    % de WST en ERP (P1: Escala baja/rápida, P3: Escala alta/lenta).
    componentRange = struct(...
        'P1', struct('Time', [ 50, 150], 'Scale', [ 10,  50], 'Type', 'Max'), ...
        'N1', struct('Time', [100, 200], 'Scale', [ 40, 120], 'Type', 'Min'), ...
        'P3', struct('Time', [250, 600], 'Scale', [ 80, 200], 'Type', 'Max'));

    componentNames = {'P1', 'N1', 'P3'};
    features = zeros(1, 9);
    feat_idx = 1;

    for i = 1:length(componentNames)
        comp = componentNames{i};
        range = componentRange.(comp);

        % 1. Encontrar índices de la ventana de tiempo
        [~, t_start_idx] = min(abs(tlag - range.Time(1)));
        [~, t_end_idx] = min(abs(tlag - range.Time(2)));
        
        % 2. Encontrar índices de la ventana de escala (asumida)
        [~, s_start_idx] = min(abs(scale - range.Scale(1)));
        [~, s_end_idx] = min(abs(scale - range.Scale(2)));

        % Asegurar que los índices sean válidos
        t_indices = t_start_idx:t_end_idx;
        s_indices = s_start_idx:s_end_idx;

        % Manejo de errores si la ventana es vacía
        if isempty(t_indices) || isempty(s_indices)
             warning(['Ventana vacía para ', comp, '. Asignando ceros.']);
             features(feat_idx:feat_idx+2) = [0, 0, 0];
             feat_idx = feat_idx + 3;
             continue;
        end
        
        % 3. Extraer el sub-mapa de interés
        Wst_window = Wst(s_indices, t_indices);

        % 4. Buscar el Pico (Máximo o Mínimo)
        if strcmpi(range.Type, 'Max')
            [W_peak, max_loc] = max(Wst_window(:));
        else % 'Min' para N1
            [W_peak, max_loc] = min(Wst_window(:));
        end

        % 5. Convertir el índice lineal (max_loc) a coordenadas (s_idx, t_idx)
        [s_rel_idx, t_rel_idx] = ind2sub(size(Wst_window), max_loc);

        % 6. Mapear a índices globales y obtener los valores finales
        t_global_idx = t_indices(t_rel_idx);
        s_global_idx = s_indices(s_rel_idx);

        % Los 3 valores del componente: Amplitud (W), Latencia (t), Escala (s)
        t_peak = tlag(t_global_idx);
        s_peak = scale(s_global_idx);

        % 7. Almacenar los resultados
        features(feat_idx:feat_idx+2) = [W_peak, t_peak, s_peak];
        feat_idx = feat_idx + 3;
    end
end