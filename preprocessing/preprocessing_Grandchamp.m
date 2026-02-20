function EEG = preprocessing_Grandchamp(sub, stage, session)
% preprocessing(data, stage, session)
% data can be multiple in some stages
% stage 1: import data, filtering, downsampling, interpolation, epoching
% stage 2: visually inspect epochs before ICA
% stage 3: remove marked epochs and run ICA
% stage 4: visually inspect components
% stage 5: remove marked comps and visually inspect epochs again
% stage 6: remove marked epochs 
% Correspondance: Christina Jin (cyj.sci@gmail.com)

p_root = fullfile('data', 'ds001787-download');
p_eeg = fullfile(p_root ,['sub-', pad(num2str(sub),3,'left','0')], ['ses-', pad(num2str(session),2,'left','0')], 'eeg');
p_prepro = char(fullfile("preprocessing"));

bandpass      = [0.1 42]; 
epoch         = [-10 8];%in seconds
baseline      = [-0.2 0];
triggers      = {'128'};
cmps2plot     = 30:-1:1;  

%%        
if stage == 1

    % import raw
    f_eeg = ['sub-', pad(num2str(sub),3,'left','0'), '_ses-', pad(num2str(session),2,'left','0'), '_task-meditation_eeg.bdf'];    disp(f_eeg);
    EEG = pop_biosig(fullfile(p_eeg, f_eeg), 'channels', 1:64); 
    
    % add channel locations
    eeglab_path = fileparts(which('eeglab.m'));
    locs_path = fullfile(eeglab_path, 'plugins', 'dipfit5.6', 'dipfit', 'standard_BEM', 'elec', 'standard_1005.elc');    
    EEG = pop_chanedit(EEG, 'lookup', locs_path); 
    % reref
    EEG = pop_reref( EEG, [16 53] ,'keepref','on');

    % band-pass filtering
    EEG = pop_eegfilt(EEG, min(bandpass), max(bandpass), 0, 0, 0, 0);

    % down-sampling
    %EEG = pop_resample(EEG, srate);

    % extract epochs
    EEG = pop_epoch(EEG, triggers, epoch); 

    % baseline correction
    EEG = pop_rmbase(EEG, baseline*1000);
    %if EEG.trials ~= 825
    %    error('Incorrect trigger number!')
    %end
    EEG = pop_rmbase(EEG, baseline*1000);
    
    f_out = [num2str(sub), '_', num2str(session), '_epochs.set'];
    
    EEG.filename = f_out;
    EEG.filepath = p_prepro;
    
    EEG.datfile = strrep(f_out, '.set', '.fdt'); 
   
    pop_saveset(EEG, 'filename', f_out, 'filepath', p_prepro);
    % save 
end

%%
% run ICA direclty since not many trials with these datasets
if stage == 2
    
    disp('Skip the mannual inspection before the ICA.')
            % load data
    EEG = pop_loadset(fullfile(p_prepro,[num2str(sub), '_', num2str(session), '_epochs.set']));
    if 0

        % mannual processing
        disp ('Call the following:')
        disp ('pop_eegplot(EEG);')  % visual inpection

        % save markers
        disp (['pop_saveset(EEG, fullfile(p_prepro, ''', num2str(sub), '', num2str(session), '_epochs.set''))']);
    end

end

%%

if stage == 3

    % load 
    EEG = pop_loadset(fullfile(p_prepro, [num2str(sub), '_', num2str(session), '_epochs.set']));

    % reject the marked epoches
    %EEG = pop_rejepoch(EEG, EEG.reject.rejmanual);

    % run ica 
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'maxsteps', 50);

    % overwrite 
    pop_saveset(EEG, fullfile(p_prepro, [num2str(sub), '_', num2str(session), '_epochs_ica.set']));

end

%%

if stage == 4

    % load 
    EEG = pop_loadset(fullfile(p_prepro, [num2str(sub), '_', num2str(session),  '_epochs_ica.set']));

    % plot cmps
    disp('Call the following:')
    disp(['pop_prop(EEG, 0,[', num2str(cmps2plot),'], NaN,{''freqrange'' [0.1 42]});'])

    % save markers
    disp (['pop_saveset(EEG, fullfile(p_prepro, ''', num2str(sub), '_', num2str(session), '_epochs_ica.set''))']);

end

%%

if stage == 5

    % load 
    EEG = pop_loadset(fullfile(p_prepro, [num2str(sub),'_', num2str(session), '_epochs_ica.set']));

    % remove comps
    EEG = pop_subcomp(EEG);

    % baseline correction
    EEG = pop_rmbase(EEG, baseline*1000);

    % save
    EEG = pop_saveset(EEG, fullfile(p_prepro, [num2str(sub),'_', num2str(session), '_epochs_ica_a.set']));

    % visually inspect epochs again
    disp ('Call the following:')
    disp ('pop_eegplot(EEG);')

    % save markers
    disp (['pop_saveset(EEG, fullfile(p_prepro, ''', num2str(sub), '_', num2str(session), '_epochs_ica_a.set''))']);

end

%%

if stage == 6

    % load
    EEG = pop_loadset(fullfile(p_prepro, [num2str(sub),'_', num2str(session), '_epochs_ica_a.set']));

    % reject the marked epoches
    EEG = pop_rejepoch(EEG, EEG.reject.rejmanual);
    
    % save 
    pop_saveset(EEG, fullfile(p_prepro, [num2str(sub), '_', num2str(session), '_epochs_ica_a2.set']));

end

end %func

