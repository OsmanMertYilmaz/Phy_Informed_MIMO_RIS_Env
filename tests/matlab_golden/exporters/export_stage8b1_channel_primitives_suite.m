function manifest = export_stage8b1_channel_primitives_suite(configCsv,outRoot)
%EXPORT_STAGE8B1_CHANNEL_PRIMITIVES_SUITE
%
% Stage 8B-1 stochastic channel parity.
%
% The current generate_channel_train uses MATLAB RNG internally. We do NOT
% try to reproduce MATLAB twister/randn in PyTorch. Instead this exporter:
%
%   1) resets a deterministic MATLAB RNG seed,
%   2) replays exactly the random draws used by generate_channel_train and
%      stores the resulting random primitives,
%   3) resets the same seed,
%   4) calls the real current generate_channel_train,
%   5) saves H as the golden reference.
%
% Therefore Python receives exactly the same:
%   XPR, ASA/ZSA/ASD/ZSD samples,
%   per-cluster angular offsets,
%   per-ray 2x2 polarization phase matrices.
%
% It also exports a small full-chain check:
%   H_BR,H_RU -> F -> SVD -> Type-I W -> Feff -> Y.
%
% Coverage: the same 8 isolated propagation labels used in the previous
% scenario suite. Each selected bank satisfies scenario_BR==scenario_RU.
%
% Default N=2 is deliberate: deterministic parity needs every stochastic
% branch/ray but does not need a Monte-Carlo-sized realization count.

    if nargin < 1 || strlength(string(configCsv)) == 0
        configCsv = "generate_train_scenarios_data_2000.csv";
    end
    if nargin < 2 || strlength(string(outRoot)) == 0
        outRoot = "stage8b1_channel_golden";
    end

    configCsv = string(configCsv);
    outRoot = string(outRoot);

    assert(isfile(configCsv),'CSV bulunamadi: %s',configCsv);

    requiredFunctions = { ...
        'generate_geometry','generate_lsp','generate_channel_train', ...
        'generate_cascaded_ch','generate_ris_response', ...
        'generate_codebook','selection_precoder','generate_eff_ch'};

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

    N = 2;
    c0 = physconst('LightSpeed');

    manifest = table( ...
        'Size',[numel(labels) 10], ...
        'VariableTypes',{ ...
            'string','double','double','double','double', ...
            'double','double','double','double','string'}, ...
        'VariableNames',{ ...
            'caseName','tableRow','sourceIndex','fc','nT', ...
            'nR','nRIS','N','isLOS','file'});

    fprintf('\n================================================\n');
    fprintf(' Stage 8B-1 stochastic channel parity exporter\n');
    fprintf('================================================\n');

    for caseIndex = 1:numel(labels)

        label = labels(caseIndex);

        mask = T.scenario_BR == label & T.scenario_RU == label;
        tableRow = find(mask,1,'first');

        assert(~isempty(tableRow), ...
            'scenario_BR==scenario_RU==%s bulunamadi.',label);

        row = T(tableRow,:);

        if ismember('index',T.Properties.VariableNames)
            sourceIndex = double(row.index);
        else
            sourceIndex = tableRow;
        end

        fc = double(row.fc);
        lambda0 = c0/fc;

        scenario = char(label);

        ris = double([row.ris_x,row.ris_y,row.ris_z]);
        gnb = double([row.gnb_x,row.gnb_y,row.gnb_z]);
        ue  = double([row.ue_x,row.ue_y,row.ue_z]);

        geometry.ris = ris;
        geometry.gnb = gnb;
        geometry.ue = ue;
        geometry = generate_geometry(geometry);

        gnb2ris = geometry.gnb2ris;
        ris2gnb = geometry.ris2gnb;
        ris2ue  = geometry.ris2ue;
        ue2ris  = geometry.ue2ris;

        nT1 = double(row.nT1);
        nT2 = double(row.nT2);
        nR1 = double(row.nR1);
        nR2 = double(row.nR2);
        nRISx = double(row.nRIS_x);
        nRISy = double(row.nRIS_y);

        nT = 2*nT1*nT2;
        nR = 2*nR1*nR2;
        nRIS = 2*nRISx*nRISy;

        chBR = nrCDLChannel;
        chBR.TransmitAntennaArray.Size = [nT1 nT2 2 1 1];
        chBR.TransmitAntennaArray.Element = 'isotropic';
        chBR.ReceiveAntennaArray.Size = [nRISx nRISy 2 1 1];
        chBR.ReceiveAntennaArray.PolarizationAngles = [45 -45];
        chBR.CarrierFrequency = fc;

        chRU = nrCDLChannel;
        chRU.TransmitAntennaArray.Size = [nRISx nRISy 2 1 1];
        chRU.TransmitAntennaArray.Element = 'isotropic';
        chRU.ReceiveAntennaArray.Size = [nR1 nR2 2 1 1];
        chRU.ReceiveAntennaArray.PolarizationAngles = [45 -45];
        chRU.CarrierFrequency = fc;

        lspBR = generate_lsp( ...
            row.scenario_BR,fc,ris2gnb,gnb2ris);
        lspRU = generate_lsp( ...
            row.scenario_RU,fc,ue2ris,ris2ue);

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

        seedBR = 810000 + 1000*caseIndex + 17;
        seedRU = 810000 + 1000*caseIndex + 53;

        BRP = capturePrimitives(seedBR,N,lspBR);
        RUP = capturePrimitives(seedRU,N,lspRU);

        rng(seedBR,'twister');
        [HBR,muBR,sigma2BR,dbarTBR,dbarRBR] = ...
            generate_channel_train( ...
                chBR,ris2gnb,gnb2ris,c0,lambda0,KBR,N,lspBR);

        rng(seedRU,'twister');
        [HRU,muRU,sigma2RU,dbarTRU,dbarRRU] = ...
            generate_channel_train( ...
                chRU,ue2ris,ris2ue,c0,lambda0,KRU,N,lspRU);

        % --------------------------------------------------------
        % Small full-chain integration check.
        % --------------------------------------------------------
        rng(910000 + sourceIndex,'twister');
        z = randi([0 1],nRIS,1);
        phaseLevels = deg2rad([45 135]);
        phi = phaseLevels(z+1).';
        [beta,gamma] = generate_ris_response(phi);

        F = generate_cascaded_ch(HBR,HRU,gamma);

        Hpilot = squeeze(F(1,:,:));
        [~,~,V] = svd(Hpilot);
        codebook = generate_codebook(2,nT1,nT2,1,1);
        [W,WIdx] = selection_precoder(codebook,1,V);

        % generate_eff_ch also returns analytic moments; for B1 we only
        % need its exact empirical F*W branch. Use dummy valid statistics
        % from the already-available analytical environment.
        %
        % Recompute the rho inputs exactly as in the project.
        rho_ris2ue = sigma2RU * compute_ch_rho_avg( ...
            dbarTRU,ris2ue, ...
            lspRU.mu_ASD,lspRU.sigma_ASD, ...
            lspRU.mu_ZSD,lspRU.sigma_ZSD, ...
            lspRU.c_ASD,lspRU.c_ZSD, ...
            lspRU.mu_offset_ZOD,lambda0);

        [rhoRB,rhoBR] = compute_ch_eff_rho_avg_fast( ...
            dbarTBR,dbarRBR,ris2gnb,gnb2ris, ...
            lspBR.mu_ASA,lspBR.sigma_ASA, ...
            lspBR.mu_ZSA,lspBR.sigma_ZSA, ...
            lspBR.mu_ASD,lspBR.sigma_ASD, ...
            lspBR.mu_ZSD,lspBR.sigma_ZSD, ...
            lspBR.c_ASA,lspBR.c_ZSA, ...
            lspBR.c_ASD,lspBR.c_ZSD, ...
            lspBR.mu_offset_ZOD,lambda0);

        [Feff,~,~] = generate_eff_ch( ...
            F,W,gamma,rhoRB,rhoBR, ...
            lspBR.mu_XPR,lspBR.sigma_XPR, ...
            muBR,sigma2BR,muRU,rho_ris2ue);

        Y = squeeze(sum(abs(Feff).^2,2));
        Y = real(Y(:));

        % Flatten primitive structs into explicit MAT variables.
        [ ...
            BR_XPR,BR_ASAv,BR_ZSAv,BR_ASDv,BR_ZSDv, ...
            BR_clusterOffsets,BR_Phi] = unpackPrimitive(BRP);

        [ ...
            RU_XPR,RU_ASAv,RU_ZSAv,RU_ASDv,RU_ZSDv, ...
            RU_clusterOffsets,RU_Phi] = unpackPrimitive(RUP);

        BR_K = KBR;
        BR_isLOS = logical(lspBR.isLOS);
        BR_M = lspBR.M; BR_L = lspBR.L;
        BR_mu_XPR = lspBR.mu_XPR; BR_sigma_XPR = lspBR.sigma_XPR;
        BR_c_ASA = lspBR.c_ASA; BR_c_ZSA = lspBR.c_ZSA;
        BR_c_ASD = lspBR.c_ASD; BR_c_ZSD = lspBR.c_ZSD;
        BR_mu_offset_ZOD = lspBR.mu_offset_ZOD;

        RU_K = KRU;
        RU_isLOS = logical(lspRU.isLOS);
        RU_M = lspRU.M; RU_L = lspRU.L;
        RU_mu_XPR = lspRU.mu_XPR; RU_sigma_XPR = lspRU.sigma_XPR;
        RU_c_ASA = lspRU.c_ASA; RU_c_ZSA = lspRU.c_ZSA;
        RU_c_ASD = lspRU.c_ASD; RU_c_ZSD = lspRU.c_ZSD;
        RU_mu_offset_ZOD = lspRU.mu_offset_ZOD;

        caseName = regexprep(label,'[^A-Za-z0-9]+','_');
        fileName = caseName + ".mat";
        filePath = fullfile(outRoot,fileName);

        save(filePath, ...
            'scenario','fc','N', ...
            'ris','gnb','ue','gnb2ris','ris2gnb','ris2ue','ue2ris', ...
            'nT1','nT2','nR1','nR2','nRISx','nRISy', ...
            'BR_K','BR_isLOS','BR_M','BR_L', ...
            'BR_mu_XPR','BR_sigma_XPR', ...
            'BR_c_ASA','BR_c_ZSA','BR_c_ASD','BR_c_ZSD', ...
            'BR_mu_offset_ZOD', ...
            'BR_XPR','BR_ASAv','BR_ZSAv','BR_ASDv','BR_ZSDv', ...
            'BR_clusterOffsets','BR_Phi', ...
            'RU_K','RU_isLOS','RU_M','RU_L', ...
            'RU_mu_XPR','RU_sigma_XPR', ...
            'RU_c_ASA','RU_c_ZSA','RU_c_ASD','RU_c_ZSD', ...
            'RU_mu_offset_ZOD', ...
            'RU_XPR','RU_ASAv','RU_ZSAv','RU_ASDv','RU_ZSDv', ...
            'RU_clusterOffsets','RU_Phi', ...
            'HBR','muBR','sigma2BR','dbarTBR','dbarRBR', ...
            'HRU','muRU','sigma2RU','dbarTRU','dbarRRU', ...
            'z','phi','beta','gamma','F','W','WIdx','Feff','Y', ...
            '-v7');

        manifest.caseName(caseIndex) = caseName;
        manifest.tableRow(caseIndex) = tableRow;
        manifest.sourceIndex(caseIndex) = sourceIndex;
        manifest.fc(caseIndex) = fc;
        manifest.nT(caseIndex) = nT;
        manifest.nR(caseIndex) = nR;
        manifest.nRIS(caseIndex) = nRIS;
        manifest.N(caseIndex) = N;
        manifest.isLOS(caseIndex) = double(lspBR.isLOS);
        manifest.file(caseIndex) = fileName;

        fprintf( ...
            '[%d/8] %-20s | nT=%d nR=%d nRIS=%d | N=%d\n', ...
            caseIndex,label,nT,nR,nRIS,N);
    end

    writetable(manifest,fullfile(outRoot,'manifest.csv'));

    zipFile = outRoot + ".zip";
    if isfile(zipFile), delete(zipFile); end
    zip(zipFile,outRoot);

    fprintf('\nManifest: %s\n',fullfile(outRoot,'manifest.csv'));
    fprintf('ZIP     : %s\n',zipFile);
    fprintf('================================================\n\n');

    disp(manifest);
end


function P = capturePrimitives(seed,N,lsp)
% Replay exactly the RNG calls in the current generate_channel_train.m.

    rng(seed,'twister');

    M = lsp.M;
    L = lsp.L;

    P.XPR = lsp.mu_XPR + lsp.sigma_XPR * randn(N,M*L);

    P.ASAv = 10.^(lsp.mu_ASA + lsp.sigma_ASA*randn(N,1));
    P.ASAv = min(P.ASAv,104);

    P.ZSAv = 10.^(lsp.mu_ZSA + lsp.sigma_ZSA*randn(N,1));
    P.ZSAv = min(P.ZSAv,52);

    P.ASDv = 10.^(lsp.mu_ASD + lsp.sigma_ASD*randn(N,1));
    P.ASDv = min(P.ASDv,104);

    P.ZSDv = 10.^(lsp.mu_ZSD + lsp.sigma_ZSD*randn(N,1));
    P.ZSDv = min(P.ZSDv,52);

    P.clusterOffsets = zeros(N,M,4);
    P.Phi = zeros(N,M,L,2,2);

    for n = 1:N
        ASA = P.ASAv(n);
        ZSA = P.ZSAv(n);
        ASD = P.ASDv(n);
        ZSD = P.ZSDv(n);

        for c = 1:M
            P.clusterOffsets(n,c,1) = randn * ASA/7;
            P.clusterOffsets(n,c,2) = randn * ZSA/7;
            P.clusterOffsets(n,c,3) = randn * ASD/7;
            P.clusterOffsets(n,c,4) = randn * ZSD/7;

            for p = 1:L
                Phi = -pi + 2*pi*rand(2,2);
                P.Phi(n,c,p,1,1) = Phi(1,1);
                P.Phi(n,c,p,1,2) = Phi(1,2);
                P.Phi(n,c,p,2,1) = Phi(2,1);
                P.Phi(n,c,p,2,2) = Phi(2,2);
            end
        end
    end
end


function [XPR,ASAv,ZSAv,ASDv,ZSDv,clusterOffsets,Phi] = unpackPrimitive(P)
    XPR = P.XPR;
    ASAv = P.ASAv;
    ZSAv = P.ZSAv;
    ASDv = P.ASDv;
    ZSDv = P.ZSDv;
    clusterOffsets = P.clusterOffsets;
    Phi = P.Phi;
end
