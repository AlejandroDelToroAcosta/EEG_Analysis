
import numpy as np
import mne
from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids
import pandas as pd
mne.viz.set_browser_backend("qt")


def extract_labeled_events(events, event_id_stim=128, event_ids_response=[2, 4, 8],
                           min_responses=2, max_response_delay=10.0, sfreq=256):
    """
    Extract stimulus events with response-based labels

    CRITICAL: Returns STIMULUS events (trigger 128), not response events

    Parameters
    ----------
    events : array
        MNE events array
    event_id_stim : int
        Stimulus event ID (128)
    event_ids_response : list
        Valid response codes [2, 4, 8]
    min_responses : int
        Minimum responses required (default: 3 for Q1+Q2+Q3)
    max_response_delay : float
        Max time (seconds) to find responses
    sfreq : float
        Sampling frequency

    Returns
    -------
    events_labeled : array
        Stimulus events with response-based labels
        Shape: (n_trials, 3) where col 3 is label (0=on-task, 1=MW)
    metadata : DataFrame
        Trial information including responses and RTs
    report : dict
        Extraction statistics
    """

    print("\n" + "="*70)
    print("EXTRACTING LABELED EVENTS (Stimulus-Locked Approach)")
    print("="*70)
    print("✓ Epoch center: Trigger 128 (stimulus onset)")
    print("✓ Epoch window: [-10, 0] seconds (mental state BEFORE question)")
    print("✓ Labels: Based on Q2 response (4 or 8 = Mind Wandering)")
    print("="*70)

    # Find all stimuli
    stim_indices = np.where(events[:, 2] == event_id_stim)[0]
    print(f"\nFound {len(stim_indices)} stimulus events (trigger 128)")

    events_labeled = []
    metadata_list = []
    max_delay_samples = int(max_response_delay * sfreq)

    trials_included = 0
    trials_excluded = 0
    excluded_details = []

    for trial_idx, stim_idx in enumerate(stim_indices):
        stim_sample = events[stim_idx, 0]

        # Find responses after this stimulus
        if stim_idx + 1 < len(events):
            subsequent = events[stim_idx + 1:]
            within_window = subsequent[
                (subsequent[:, 0] - stim_sample) <= max_delay_samples
            ]
        else:
            within_window = np.array([])

        # Collect responses
        responses = []
        for evt in within_window:
            if evt[2] in event_ids_response:
                responses.append(evt)
                if len(responses) == 3:
                    break
            if evt[2] == event_id_stim:  # Next stimulus
                break

        n_resp = len(responses)

        # Prepare response data
        resp_samples = [np.nan, np.nan, np.nan]
        resp_codes = [np.nan, np.nan, np.nan]
        rts = [np.nan, np.nan, np.nan]

        for i, resp in enumerate(responses):
            resp_samples[i] = resp[0]
            resp_codes[i] = resp[2]
            rts[i] = resp[0] - stim_sample

        # Include trial if enough responses
        if n_resp >= min_responses:
            # Determine label from Q2 response
            q2 = resp_codes[1]

            if np.isnan(q2):
                label = 999  # Unknown
                label_str = 'unknown'
            elif q2 in [4, 8]:
                label = 1  # Mind Wandering
                label_str = 'mind_wandering'
            else:  # q2 == 2
                label = 0  # On-Task
                label_str = 'on_task'

            # CRITICAL: Event is STIMULUS (128), not response
            events_labeled.append([stim_sample, 0, label])

            metadata_list.append({
                'trial': trial_idx + 1,
                'stim_sample': stim_sample,
                'n_responses': n_resp,
                'label': label,
                'label_str': label_str,
                'q1_response': resp_codes[0],
                'q2_response': resp_codes[1],
                'q3_response': resp_codes[2],
                'rt_q1_samples': rts[0],
                'rt_q2_samples': rts[1],
                'rt_q3_samples': rts[2]
            })

            trials_included += 1
        else:
            trials_excluded += 1
            excluded_details.append({
                'trial': trial_idx + 1,
                'n_responses': n_resp,
                'reason': f'Only {n_resp}/{min_responses} responses'
            })

    events_labeled = np.array(events_labeled, dtype=int)
    metadata = pd.DataFrame(metadata_list)

    # Report
    report = {
        'total_stimuli': len(stim_indices),
        'trials_included': trials_included,
        'trials_excluded': trials_excluded,
        'inclusion_rate': (trials_included / len(stim_indices) * 100) if len(stim_indices) > 0 else 0,
        'excluded_trials': excluded_details
    }

    if len(metadata) > 0:
        label_counts = metadata['label_str'].value_counts()
        report['label_distribution'] = label_counts.to_dict()

    # Print summary
    print(f"\n✓ Extraction complete:")
    print(f"  Total stimuli: {report['total_stimuli']}")
    print(f"  Included: {trials_included} ({report['inclusion_rate']:.1f}%)")
    print(f"  Excluded: {trials_excluded}")

    if len(metadata) > 0:
        print(f"\n  Label distribution:")
        for lbl, cnt in metadata['label_str'].value_counts().items():
            pct = (cnt / len(metadata)) * 100
            print(f"    {lbl}: {cnt} ({pct:.1f}%)")

    if trials_excluded > 0:
        print(f"\n  ⚠️  Excluded trials (first 5):")
        for detail in excluded_details[:5]:
            print(f"    Trial {detail['trial']}: {detail['reason']}")

    return events_labeled, metadata, report


def preprocessing_grandchamp_v2(sub, stage, session, bids_root=None,
                                use_bids=True, fast_mode=False, min_responses=2):
    """
    EEG preprocessing with stimulus-locked epoching and response labels

    Parameters
    ----------
    sub : int or str
        Subject number
    stage : int
        Processing stage (1-6)
    session : int or str
        Session number
    bids_root : str or Path
        BIDS dataset root
    use_bids : bool
        Use BIDS loading (recommended)
    fast_mode : bool
        Fast settings for testing
    min_responses : int
        Minimum responses to include trial (default: 3)

    Returns
    -------
    epochs : mne.Epochs
        Preprocessed epochs centered on stimulus (trigger 128)
        With labels: 0 = on-task, 1 = mind-wandering
    """

    # Format strings
    if isinstance(sub, int):
        sub_str = f'{sub:03d}'
    else:
        sub_str = sub.zfill(3)

    if isinstance(session, int):
        ses_str = f'{session:02d}'
    else:
        ses_str = session.zfill(2)

    # Paths
    if bids_root is None:
        bids_root = Path('data') / 'ds001787-download'
    else:
        bids_root = Path(bids_root)

    p_prepro_session1 = Path('preprocessing/session_1')
    p_prepro_session2 = Path('preprocessing/session_2')

    p_prepro_session1_metadata = Path('preprocessing/session_1/metadata')
    p_prepro_session2_metadata = Path('preprocessing/session_2/metadata')

    p_prepro_session2.mkdir(exist_ok=True)
    p_prepro_session1.mkdir(exist_ok=True)

    p_prepro_session2_metadata.mkdir(exist_ok=True)
    p_prepro_session1_metadata.mkdir(exist_ok=True)

    p_session = p_prepro_session1 if ses_str == '01' else p_prepro_session2
    p_metadata = p_prepro_session1_metadata if ses_str == '01' else p_prepro_session2_metadata
    # Parameters
    l_freq = 0.1
    h_freq = 42
    tmin = -10.0  # 10 seconds BEFORE stimulus
    tmax = 0.0    # Up to stimulus onset
    baseline = (-0.5, 0)  # Baseline just before stimulus

    # ========================================================================
    # STAGE 1: Import, filter, epoch (stimulus-locked)
    # ========================================================================
    if stage == 1:
        print(f"\n=== Stage 1: Stimulus-Locked Epoching ===")
        print(f"Subject: {sub_str}, Session: {ses_str}")
        if fast_mode:
            print("⚡ FAST MODE enabled")

        # Load data
        if use_bids:
            bids_path = BIDSPath(
                subject=sub_str,
                session=ses_str,
                task="Meditation",
                root=bids_root
            )
            print(f"\nLoading: {bids_path}")
            raw = read_raw_bids(bids_path=bids_path, verbose=True)
            raw.load_data()
        else:
            p_eeg = bids_root / f'sub-{sub_str}' / f'ses-{ses_str}' / 'eeg'
            f_eeg = f'sub-{sub_str}_ses-{ses_str}_task-Meditation_eeg.bdf'
            print(f"\nLoading: {p_eeg / f_eeg}")
            raw = mne.io.read_raw_bdf(p_eeg / f_eeg, preload=True, verbose=True)
            raw.pick_channels(raw.ch_names[:64])
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, on_missing='warn')

        print(f"Channels: {len(raw.ch_names)}, Sfreq: {raw.info['sfreq']} Hz")

        # Re-reference
        if len(raw.ch_names) >= 54:
            ref_ch = [raw.ch_names[15], raw.ch_names[52]]
            print(f"Re-referencing to: {ref_ch}")
            raw, _ = mne.set_eeg_reference(raw, ref_channels=ref_ch, projection=False)
        else:
            raw, _ = mne.set_eeg_reference(raw, ref_channels='average', projection=False)

        # Filter
        print(f"Filtering: {l_freq}-{h_freq} Hz")
        raw.filter(l_freq, h_freq, fir_design='firwin', verbose=True)

        # Find events
        try:
            events = mne.find_events(raw, stim_channel='Status', verbose=True)
        except ValueError:
            events = mne.find_events(raw, verbose=True)

        print(f"\nTotal events: {len(events)}")
        print(f"Unique IDs: {np.unique(events[:, 2])}")

        # Extract labeled events
        events_labeled, metadata, report = extract_labeled_events(
            events,
            event_id_stim=128,
            event_ids_response=[2, 4, 8],
            min_responses=min_responses,
            max_response_delay=10.0,
            sfreq=raw.info['sfreq']
        )

        # Save metadata
        if ses_str == "01":
            metadata_file = p_prepro_session1_metadata / f'{sub_str}_{ses_str}_metadata.csv'
            report_file = p_prepro_session1_metadata / f'{sub_str}_{ses_str}_extraction_report.txt'

        else:
            metadata_file = p_prepro_session2_metadata / f'{sub_str}_{ses_str}_metadata.csv'
            report_file = p_prepro_session2_metadata / f'{sub_str}_{ses_str}_extraction_report.txt'



        # Add RT in seconds
        sfreq = raw.info['sfreq']
        metadata['rt_q1_sec'] = metadata['rt_q1_samples'] / sfreq
        metadata['rt_q2_sec'] = metadata['rt_q2_samples'] / sfreq
        metadata['rt_q3_sec'] = metadata['rt_q3_samples'] / sfreq

        metadata.to_csv(metadata_file, index=False)
        print(f"\n✓ Metadata saved: {metadata_file}")


        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("STIMULUS-LOCKED EXTRACTION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Subject: {sub_str}, Session: {ses_str}\n\n")
            f.write(f"Total stimuli: {report['total_stimuli']}\n")
            f.write(f"Included: {report['trials_included']} ({report['inclusion_rate']:.1f}%)\n")
            f.write(f"Excluded: {report['trials_excluded']}\n\n")

            if 'label_distribution' in report:
                f.write("Label distribution:\n")
                for lbl, cnt in report['label_distribution'].items():
                    f.write(f"  {lbl}: {cnt}\n")

            if len(report['excluded_trials']) > 0:
                f.write("\nExcluded trials:\n")
                for detail in report['excluded_trials']:
                    f.write(f"  Trial {detail['trial']}: {detail['reason']}\n")

        print(f"✓ Report saved: {report_file}")

        # Print RT stats
        print(f"\nReaction Times (mean ± std):")
        for q in ['q1', 'q2', 'q3']:
            rt = metadata[f'rt_{q}_sec'].dropna()
            if len(rt) > 0:
                print(f"  {q.upper()}: {rt.mean():.3f} ± {rt.std():.3f} s")

        # Create event dictionary
        event_dict = {
            'on_task': 0,
            'mind_wandering': 1
        }

        # Create epochs (stimulus-locked!)
        print(f"\nCreating epochs: [{tmin}, {tmax}] s around trigger 128")
        epochs = mne.Epochs(
            raw,
            events_labeled,
            event_id=event_dict,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            verbose=True,
            metadata=metadata
        )

        print(f"\nTotal epochs: {len(epochs)}")
        print(f"  On-Task: {len(epochs['on_task'])}")
        print(f"  Mind Wandering: {len(epochs['mind_wandering'])}")

        # Save
        f_out = f'{sub_str}_{ses_str}_epochs-epo.fif'
        if ses_str == "01":
            epochs.save(p_prepro_session1 / f_out, overwrite=True)
        else:
            epochs.save(p_prepro_session2 / f_out, overwrite=True)


        print(f"\n✓ Saved: {f_out}")

        return epochs
        # ========================================================================
        # STAGE 2: Visual inspection before ICA (optional)
        # ========================================================================
    elif stage == 2:
        print(f"\n=== Stage 2: Visual Inspection (Interactive) ===")

        # 1. Cargar las épocas del Stage 1
        f_in = f'{sub_str}_{ses_str}_epochs-epo.fif'
        if ses_str == "01":
            epochs = mne.read_epochs(p_prepro_session1 / f_in, preload=True)

        else:
            epochs = mne.read_epochs(p_prepro_session2 / f_in, preload=True)


        print(f"\nLoaded {len(epochs)} epochs")

        # 2. LANZAR EL PLOT INTERACTIVO
        # El programa se detendrá aquí hasta que cierres la ventana (gracias a block=True)
        print("\nAbriendo interfaz de inspección... Marca en ROJO las épocas con mucho ruido.")
        epochs.plot(n_epochs=10, n_channels=30, block=True, scalings='auto')

        # 3. APLICAR CAMBIOS
        # Una vez cierras la ventana, el código sigue ejecutándose:
        n_prev = len(epochs)
        epochs.drop_bad()  # Elimina físicamente las marcadas en rojo
        n_post = len(epochs)

        print(f"\nInspección finalizada:")
        print(f"  Épocas eliminadas: {n_prev - n_post}")
        print(f"  Épocas restantes: {n_post}")

        # 4. GUARDAR EL RESULTADO LIMPIO
        # Sobrescribimos el archivo para que el Stage 3 (ICA) use los datos sin ruido bruto
        if ses_str == "01":
            epochs.save(p_prepro_session1 / f_in, overwrite=True)

        else:
            epochs.save(p_prepro_session2 / f_in, overwrite=True)
        print(f"✓ Épocas filtradas visualmente guardadas en: {f_in}")

        return epochs
    # ========================================================================
    # STAGE 3: ICA
    # ========================================================================
    elif stage == 3:
        print(f"\n=== Stage 3: ICA ===")

        f_in = f'{sub_str}_{ses_str}_epochs-epo.fif'
        epochs = mne.read_epochs(p_session / f_in, preload=True)

        print(f"Loaded {len(epochs)} epochs")

        if fast_mode:
            n_components = min(15, len(epochs.ch_names))
            ica_method = 'fastica'
            max_iter = 200
            print("⚡ FAST MODE: FastICA with 15 components")
        else:
            n_components = min(64, len(epochs.ch_names))
            ica_method = 'infomax'
            max_iter = 50

        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=ica_method,
            fit_params=dict(extended=True) if ica_method == 'infomax' else None,
            max_iter=max_iter,
            random_state=42
        )

        print(f"Running ICA ({ica_method}, {n_components} components)...")
        ica.fit(epochs)

        f_ica = f'{sub_str}_{ses_str}_ica.fif'
        ica.save(p_session / f_ica, overwrite=True)

        f_out = f'{sub_str}_{ses_str}_epochs_ica-epo.fif'
        epochs.save(p_session / f_out, overwrite=True)

        print(f"\n✓ Saved: {f_ica}, {f_out}")

        return epochs, ica

    # ========================================================================
    # STAGE 4: Inspect ICA components
    # ========================================================================
    elif stage == 4:
        print(f"\n=== Stage 4: Inspect ICA Components ===")

        # Load epochs and ICA
        f_epochs = f'{sub_str}_{ses_str}_epochs_ica-epo.fif'
        f_ica = f'{sub_str}_{ses_str}_ica.fif'

        epochs = mne.read_epochs(p_session / f_epochs, preload=True)
        ica = mne.preprocessing.read_ica(p_session / f_ica)

        print(f"Loaded {len(epochs)} epochs and ICA with {ica.n_components_} components")

        print("\n" + "="*70)
        print("MANUAL INSPECTION REQUIRED")
        print("="*70)
        print("\nTo inspect ICA components, run the following commands:\n")
        print(">>> ica.plot_components(picks=range(30), inst=epochs)")
        print(">>> ica.plot_sources(epochs, block=True)")

        print("\nTo mark bad components (e.g., eye blinks, muscle artifacts):")
        print(">>> ica.exclude = [0, 2, 5]  # Replace with bad component indices")

        print("\nTo save ICA with marked components:")
        print(f">>> ica.save('{p_session / f_ica}', overwrite=True)")

        print("\n" + "="*70)
        print("OPTIONAL: Automatic artifact detection")
        print("="*70)

        # Automatic EOG detection
        try:
            eog_indices, eog_scores = ica.find_bads_eog(epochs, threshold=3.0)
            if len(eog_indices) > 0:
                print(f"\n✓ Automatic EOG detection found components: {eog_indices}")
                print(f"  Scores: {eog_scores[eog_indices]}")
                print("\nTo accept these automatically:")
                print(f">>> ica.exclude = {eog_indices}")
                print(f">>> ica.save('{p_session / f_ica}', overwrite=True)")
            else:
                print("\n⚠️  No EOG components detected automatically")
                print("  → Inspect components manually")
        except Exception as e:
            print(f"\n⚠️  Automatic EOG detection failed: {e}")
            print("  → Inspect components manually")

        print("\n" + "="*70)

        return epochs, ica

    # ========================================================================
    # STAGE 5: Apply ICA and inspect epochs again
    # ========================================================================
    elif stage == 5:
        print(f"\n=== Stage 5: Apply ICA ===")

        # Load epochs and ICA
        f_epochs = f'{sub_str}_{ses_str}_epochs_ica-epo.fif'
        f_ica = f'{sub_str}_{ses_str}_ica.fif'

        epochs = mne.read_epochs(p_session / f_epochs, preload=True)
        ica = mne.preprocessing.read_ica(p_session / f_ica)

        print(f"Loaded {len(epochs)} epochs")
        print(f"ICA components to remove: {ica.exclude}")

        if len(ica.exclude) == 0:
            print("\n⚠️  WARNING: No components marked for removal!")
            print("  Run Stage 4 first and mark bad components")
            print("  Or the ICA will have no effect")

        # Apply ICA (remove marked components)
        print(f"\nApplying ICA (removing {len(ica.exclude)} components)...")
        epochs_clean = ica.apply(epochs.copy())

        # Re-apply baseline correction
        print("Re-applying baseline correction...")
        epochs_clean.apply_baseline((-0.5, 0))

        # Save cleaned epochs
        f_out = f'{sub_str}_{ses_str}_epochs_ica_a-epo.fif'
        epochs_clean.save(p_session / f_out, overwrite=True)
        print(f"\n✓ Saved: {f_out}")

        print("\n" + "="*70)
        print("OPTIONAL: Visual inspection of cleaned epochs")
        print("="*70)
        print("\nTo visually inspect cleaned epochs:")
        print(">>> epochs_clean.plot(n_epochs=10, n_channels=30, block=True, scalings='auto')")

        print("\nTo mark additional bad epochs:")
        print(">>> epochs_clean.drop_bad()")

        print("\nTo save after manual inspection:")
        print(f">>> epochs_clean.save('{p_session / f_out}', overwrite=True)")
        print("="*70)

        return epochs_clean

    # ========================================================================
    # STAGE 6: Final epoch rejection
    # ========================================================================
    elif stage == 6:
        print(f"\n=== Stage 6: Final Epoch Rejection ===")

        # Load cleaned epochs
        f_in = f'{sub_str}_{ses_str}_epochs_ica_a-epo.fif'
        epochs = mne.read_epochs(p_session / f_in, preload=True)

        print(f"Loaded {len(epochs)} epochs")

        # Get initial counts
        n_initial = len(epochs)
        n_on_task_initial = len(epochs['on_task']) if 'on_task' in epochs.event_id else 0
        n_mw_initial = len(epochs['mind_wandering']) if 'mind_wandering' in epochs.event_id else 0

        # Drop bad epochs (manually marked)
        print("\nRejecting manually marked bad epochs...")
        epochs.drop_bad()

        # Final counts
        n_final = len(epochs)
        n_on_task_final = len(epochs['on_task']) if 'on_task' in epochs.event_id else 0
        n_mw_final = len(epochs['mind_wandering']) if 'mind_wandering' in epochs.event_id else 0

        n_rejected = n_initial - n_final
        rejection_rate = (n_rejected / n_initial * 100) if n_initial > 0 else 0

        print(f"\n✓ Rejection complete:")
        print(f"  Initial epochs: {n_initial}")
        print(f"  Rejected: {n_rejected} ({rejection_rate:.1f}%)")
        print(f"  Final epochs: {n_final}")

        if n_on_task_final > 0 or n_mw_final > 0:
            print(f"\n  Label distribution:")
            print(f"    On-Task: {n_on_task_initial} → {n_on_task_final}")
            print(f"    Mind Wandering: {n_mw_initial} → {n_mw_final}")

        # Save final epochs
        f_out = f'{sub_str}_{ses_str}_epochs_ica_a2-epo.fif'
        epochs.save(p_session / f_out, overwrite=True)
        print(f"\n✓ Final epochs saved: {f_out}")

        # Save summary report
        summary_file = p_session / f'{sub_str}_{ses_str}_final_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PREPROCESSING PIPELINE SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Subject: {sub_str}\n")
            f.write(f"Session: {ses_str}\n\n")
            f.write(f"Initial epochs (after Stage 1): {n_initial}\n")
            f.write(f"Rejected epochs: {n_rejected} ({rejection_rate:.1f}%)\n")
            f.write(f"Final epochs: {n_final}\n\n")

            if n_on_task_final > 0 or n_mw_final > 0:
                f.write("Label distribution:\n")
                f.write(f"  On-Task: {n_on_task_final}\n")
                f.write(f"  Mind Wandering: {n_mw_final}\n\n")

            f.write("Files generated:\n")
            f.write(f"  - {sub_str}_{ses_str}_metadata.csv\n")
            f.write(f"  - {sub_str}_{ses_str}_extraction_report.txt\n")
            f.write(f"  - {sub_str}_{ses_str}_epochs-epo.fif\n")
            f.write(f"  - {sub_str}_{ses_str}_ica.fif\n")
            f.write(f"  - {sub_str}_{ses_str}_epochs_ica-epo.fif\n")
            f.write(f"  - {sub_str}_{ses_str}_epochs_ica_a-epo.fif\n")
            f.write(f"  - {sub_str}_{ses_str}_epochs_ica_a2-epo.fif\n")

        print(f"\n✓ Summary saved: {summary_file}")

        print("\n" + "="*70)
        print("🎉 PREPROCESSING PIPELINE COMPLETE!")
        print("="*70)
        print("\nYour data is ready for analysis!")
        print(f"\nFinal cleaned epochs: {p_session / f_out}")
        print(f"Load with: epochs = mne.read_epochs('{p_session / f_out}')")
        print("="*70)

        return epochs

    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 1-6.")
