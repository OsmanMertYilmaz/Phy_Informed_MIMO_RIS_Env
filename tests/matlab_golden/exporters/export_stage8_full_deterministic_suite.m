function manifest = export_stage8_full_deterministic_suite(configCsv,outRoot)
%EXPORT_STAGE8_FULL_DETERMINISTIC_SUITE
%
% Stage 8A integration golden suite.
%
% Raw bank metadata + WIdx + Z are sufficient to rebuild the whole
% deterministic analytical environment in Python:
%
%   geometry -> LSP/K -> moments -> rho -> W -> gamma
%   -> muFeff/sigma2Feff -> Cmat -> muSNR/sigma2Wick
%
% Scope:
%   - nl = 1
%   - XP = 2
%   - fixed W represented by Type-I WIdx
%   - no stochastic pilot-channel generation yet
%
% Coverage:
%   8 isolated propagation labels:
%     Indoor-Office LOS/NLOS
%     UMi LOS/NLOS
%     UMa LOS/NLOS
%     RMa LOS/NLOS
%
% For each label, choose the lowest-complexity row satisfying:
%
%   scenario_BR == scenario_RU == label
%
% This keeps full double-precision integration validation memory-safe while
% preserving all environment/LOS-NLOS branches.
%
% Each bank evaluates 16 deterministic binary RIS candidates.

    if nargin < 1 || strlength(string(configCsv)) == 0
        configCsv = "generate_train_scenarios_data_2000.csv";
    end
    if nargin < 2 || strlength(string(outRoot)) == 0
        outRoot = "stage8_full_deterministic_golden";
    end

    configCsv = string(configCsv);
    outRoot = string(outRoot);

    assert(isfile(configCsv),'CSV bulunamadi: %s',configCsv);

    requiredFunctions = { ...
        'generate_geometry','generate_lsp', ...
        'generate_channel_moments_no_cdl', ...
        'compute_ch_rho_avg','compute_ch_eff_rho_avg_fast', ...
        'generate_codebook','generate_ris_response', ...
        'generate_eff_moments','evaluate_gamma_metric'};

    for k = 1:numel(requiredFunctions)
        assert(exist(requiredFunctions{k},'file') ~= 0, ...
            'MATLAB path''inde eksik fonksiyon: %s',requiredFunctions{k});
    end

    if isfolder(outRoot)
        rmdir(outRoot,'s');
    end
    mkdir(outRoot);

    opts = detectImportOptions(configCsv);
    opts = setvartype(opts,{'scenario_BR','scenario_RU'},'string');
    T = readtable(configCsv,opts);

    labels = [ ...
        "Indoor-Office-LOS"
        "Indoor-Office-NLOS"
        "UMi-LOS"
        "UMi-NLOS"
        "UMa-LOS"
        "UMa-NLOS"
        "RMa-LOS"
        "RMa-NLOS"];

    nCases = numel(labels);
    nCandidates = 16;
    c0 = physconst('LightSpeed');

    manifest = table( ...
        'Size',[nCases 10], ...
        'VariableTypes',{ ...
            'string','double','double','double','double', ...
            'double','double','double','double','string'}, ...
        'VariableNames',{ ...
            'caseName','tableRow','sourceIndex','fc','nT', ...
            'nR','nRIS','nCandidates','complexityScore','file'});

    fprintf('\n=================================================\n');
    fprintf(' Stage 8A full deterministic integration exporter\n');
    fprintf('=================================================\n');

    for caseIndex = 1:nCases

        label = labels(caseIndex);

        mask = T.scenario_BR == label & T.scenario_RU == label;
        idx = find(mask);
        assert(~isempty(idx), ...
            'scenario_BR==scenario_RU==%s satiri bulunamadi.',label);

        % Pick the cheapest full Stage-3 covariance case:
        % approximately nR^2 * nRIS^2.
        nRAll = 2 .* double(T.nR1(idx)) .* double(T.nR2(idx));
        nRISAll = 2 .* double(T.nRIS_x(idx)) .* double(T.nRIS_y(idx));
        cost = (nRAll.^2) .* (nRISAll.^2);

        [~,localBest] = min(cost);
        tableRow = idx(localBest);
        row = T(tableRow,:);

        if ismember('index',T.Properties.VariableNames)
            sourceIndex = double(row.index);
        else
            sourceIndex = tableRow;
        end

        fc = double(row.fc);
        lambda0 = c0/fc;

        scenarioBR = char(string(row.scenario_BR));
        scenarioRU = char(string(row.scenario_RU));

        ris = double([row.ris_x,row.ris_y,row.ris_z]);
        gnb = double([row.gnb_x,row.gnb_y,row.gnb_z]);
        ue  = double([row.ue_x,row.ue_y,row.ue_z]);

        geometry.ris = ris;
        geometry.gnb = gnb;
        geometry.ue = ue;
        geometry = generate_geometry(geometry);

        nT1 = double(row.nT1);
        nT2 = double(row.nT2);
        nR1 = double(row.nR1);
        nR2 = double(row.nR2);
        nRISx = double(row.nRIS_x);
        nRISy = double(row.nRIS_y);

        nT = 2*nT1*nT2;
        nR = 2*nR1*nR2;
        nRIS = 2*nRISx*nRISy;

        gnb2ris = nrCDLChannel;
        gnb2ris.TransmitAntennaArray.Size = [nT1 nT2 2 1 1];
        gnb2ris.TransmitAntennaArray.Element = 'isotropic';
        gnb2ris.ReceiveAntennaArray.Size = [nRISx nRISy 2 1 1];
        gnb2ris.ReceiveAntennaArray.PolarizationAngles = [45 -45];
        gnb2ris.CarrierFrequency = fc;

        ris2ue = nrCDLChannel;
        ris2ue.TransmitAntennaArray.Size = [nRISx nRISy 2 1 1];
        ris2ue.TransmitAntennaArray.Element = 'isotropic';
        ris2ue.ReceiveAntennaArray.Size = [nR1 nR2 2 1 1];
        ris2ue.ReceiveAntennaArray.PolarizationAngles = [45 -45];
        ris2ue.CarrierFrequency = fc;

        lspBR = generate_lsp( ...
            string(row.scenario_BR),fc, ...
            geometry.ris2gnb,geometry.gnb2ris);

        lspRU = generate_lsp( ...
            string(row.scenario_RU),fc, ...
            geometry.ue2ris,geometry.ris2ue);

        if lspBR.isLOS
            KBR = 10^(lspBR.mu_K/10);
        else
            KBR = 0;
        end

        if lspRU.isLOS
            KRU = 10^(lspRU.mu_K/10);
        else
            KRU = 0;
        end

        [muBR,sigma2BR,dbarTBR,dbarRBR] = ...
            generate_channel_moments_no_cdl( ...
                gnb2ris,geometry.ris2gnb,geometry.gnb2ris, ...
                c0,lambda0,KBR,lspBR);

        [muRU,sigma2RU,dbarTRU,dbarRRU] = ...
            generate_channel_moments_no_cdl( ...
                ris2ue,geometry.ue2ris,geometry.ris2ue, ...
                c0,lambda0,KRU,lspRU);

        [rhoRB,rhoBR] = compute_ch_eff_rho_avg_fast( ...
            dbarTBR,dbarRBR, ...
            geometry.ris2gnb,geometry.gnb2ris, ...
            lspBR.mu_ASA,lspBR.sigma_ASA, ...
            lspBR.mu_ZSA,lspBR.sigma_ZSA, ...
            lspBR.mu_ASD,lspBR.sigma_ASD, ...
            lspBR.mu_ZSD,lspBR.sigma_ZSD, ...
            lspBR.c_ASA,lspBR.c_ZSA, ...
            lspBR.c_ASD,lspBR.c_ZSD, ...
            lspBR.mu_offset_ZOD,lambda0);

        rhoRUhop = sigma2RU * compute_ch_rho_avg( ...
            dbarTRU,geometry.ris2ue, ...
            lspRU.mu_ASD,lspRU.sigma_ASD, ...
            lspRU.mu_ZSD,lspRU.sigma_ZSD, ...
            lspRU.c_ASD,lspRU.c_ZSD, ...
            lspRU.mu_offset_ZOD,lambda0);

        [rhoRU,rhoUR] = compute_ch_eff_rho_avg_fast( ...
            dbarTRU,dbarRRU, ...
            geometry.ue2ris,geometry.ris2ue, ...
            lspRU.mu_ASA,lspRU.sigma_ASA, ...
            lspRU.mu_ZSA,lspRU.sigma_ZSA, ...
            lspRU.mu_ASD,lspRU.sigma_ASD, ...
            lspRU.mu_ZSD,lspRU.sigma_ZSD, ...
            lspRU.c_ASA,lspRU.c_ZSA, ...
            lspRU.c_ASD,lspRU.c_ZSD, ...
            lspRU.mu_offset_ZOD,lambda0);

        % ----------------------------------------------------
        % Deterministic Type-I fixed W.
        % ----------------------------------------------------
        codebook = generate_codebook(2,nT1,nT2,1,1);

        nI2 = size(codebook,3);
        nI11 = size(codebook,4);
        nI12 = size(codebook,5);

        i11 = mod(caseIndex-1,nI11)+1;
        i12 = mod(2*(caseIndex-1),nI12)+1;
        i2  = mod(caseIndex-1,nI2)+1;

        WIdx = [i11 i12 i2];
        W = codebook(:,1,i2,i11,i12);

        % ----------------------------------------------------
        % Candidate Z set.
        % ----------------------------------------------------
        Z = zeros(nCandidates,nRIS);
        Z(1,:) = 0;
        Z(2,:) = 1;
        Z(3,:) = mod(0:nRIS-1,2);
        Z(4,:) = 1-Z(3,:);

        rng(930000 + sourceIndex,'twister');
        Z(5:end,:) = randi([0 1],nCandidates-4,nRIS);

        gammaCandidates = complex(zeros(nCandidates,nRIS));
        muFeffCandidates = complex(zeros(nCandidates,nR));
        sigma2FeffCandidates = zeros(nCandidates,nR);
        CmatCandidates = complex(zeros(nR,nR,nCandidates));
        muSNRCandidates = zeros(nCandidates,1);
        sigma2WickCandidates = zeros(nCandidates,1);

        UBR = [];

        phaseLevels = deg2rad([45 135]);

        for candidateIndex = 1:nCandidates

            z = Z(candidateIndex,:).';
            phi = phaseLevels(z+1).';
            [~,gamma] = generate_ris_response(phi);

            [muFeff,sigma2Feff,UBRcandidate] = ...
                generate_eff_moments( ...
                    W,gamma, ...
                    rhoRB,rhoBR, ...
                    lspBR.mu_XPR,lspBR.sigma_XPR, ...
                    muBR,sigma2BR, ...
                    muRU,rhoRUhop);

            [muSNR,sigma2Wick,~,Cmat] = ...
                evaluate_gamma_metric( ...
                    rhoRU,rhoUR,rhoRB,rhoBR, ...
                    gamma,W, ...
                    [],muFeff,sigma2Feff, ...
                    lspBR.mu_XPR,lspBR.sigma_XPR, ...
                    lspRU.mu_XPR,lspRU.sigma_XPR, ...
                    muBR,sigma2BR, ...
                    muRU,sigma2RU,UBRcandidate);

            if isempty(UBR)
                UBR = UBRcandidate(:,:,1);
            end

            gammaCandidates(candidateIndex,:) = gamma(:).';
            muFeffCandidates(candidateIndex,:) = muFeff(:,1).';
            sigma2FeffCandidates(candidateIndex,:) = sigma2Feff(:,1).';
            CmatCandidates(:,:,candidateIndex) = Cmat;
            muSNRCandidates(candidateIndex) = muSNR;
            sigma2WickCandidates(candidateIndex) = sigma2Wick;
        end

        caseName = regexprep(label,'[^A-Za-z0-9]+','_');
        fileName = caseName + ".mat";
        filePath = fullfile(outRoot,fileName);

        save(filePath, ...
            'scenarioBR','scenarioRU','fc', ...
            'ris','gnb','ue', ...
            'nT1','nT2','nR1','nR2','nRISx','nRISy', ...
            'WIdx','W','Z', ...
            'gammaCandidates','UBR', ...
            'muFeffCandidates','sigma2FeffCandidates', ...
            'CmatCandidates','muSNRCandidates','sigma2WickCandidates', ...
            '-v7');

        complexityScore = nR^2*nRIS^2;

        manifest.caseName(caseIndex) = caseName;
        manifest.tableRow(caseIndex) = tableRow;
        manifest.sourceIndex(caseIndex) = sourceIndex;
        manifest.fc(caseIndex) = fc;
        manifest.nT(caseIndex) = nT;
        manifest.nR(caseIndex) = nR;
        manifest.nRIS(caseIndex) = nRIS;
        manifest.nCandidates(caseIndex) = nCandidates;
        manifest.complexityScore(caseIndex) = complexityScore;
        manifest.file(caseIndex) = fileName;

        fprintf( ...
            '[%d/8] %-20s | nT=%d nR=%d nRIS=%d | C=%d\n', ...
            caseIndex,label,nT,nR,nRIS,nCandidates);
    end

    writetable(manifest,fullfile(outRoot,'manifest.csv'));

    zipFile = outRoot + ".zip";
    if isfile(zipFile)
        delete(zipFile);
    end
    zip(zipFile,outRoot);

    fprintf('\n=================================================\n');
    fprintf(' Stage 8A golden suite completed.\n');
    fprintf(' Manifest : %s\n',fullfile(outRoot,'manifest.csv'));
    fprintf(' ZIP      : %s\n',zipFile);
    fprintf('=================================================\n\n');

    disp(manifest);
end
