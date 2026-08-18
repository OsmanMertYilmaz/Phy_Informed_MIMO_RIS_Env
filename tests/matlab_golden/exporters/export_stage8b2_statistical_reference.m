function manifest = export_stage8b2_statistical_reference(configCsv,outRoot,N_ref)
%EXPORT_STAGE8B2_STATISTICAL_REFERENCE
%
% Stage 8B-2 MATLAB statistical reference.
%
% Unlike Stage 8B-1, random samples are NOT exported. MATLAB and Python use
% independent RNG streams. Only statistical summaries are compared.
%
% For every propagation condition:
%   H_BR, H_RU statistics are summarized around analytic muH/sigma2H.
%   A fixed deterministic W and z are used for:
%       H_BR,H_RU -> F -> Feff -> Y
%   and empirical Y quantiles are exported, including direct q05.
%
% N_ref default = 1000. This is deliberately a statistical sanity reference,
% not the final high-precision q05 label engine. Increase to 4000+ later if
% tighter tail comparisons are desired.

    if nargin < 1 || strlength(string(configCsv)) == 0
        configCsv = "generate_train_scenarios_data_2000.csv";
    end
    if nargin < 2 || strlength(string(outRoot)) == 0
        outRoot = "stage8b2_statistical_reference";
    end
    if nargin < 3 || isempty(N_ref)
        N_ref = 1000;
    end

    configCsv = string(configCsv);
    outRoot = string(outRoot);

    assert(isfile(configCsv),'CSV bulunamadi: %s',configCsv);

    requiredFunctions = { ...
        'generate_geometry','generate_lsp','generate_channel_train', ...
        'generate_cascaded_ch','generate_ris_response','generate_codebook'};

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

    c0 = physconst('LightSpeed');

    manifest = table( ...
        'Size',[numel(labels) 10], ...
        'VariableTypes',{ ...
            'string','double','double','double','double', ...
            'double','double','double','double','string'}, ...
        'VariableNames',{ ...
            'caseName','tableRow','sourceIndex','fc','nT', ...
            'nR','nRIS','N_ref','isLOS','file'});

    fprintf('\n===============================================\n');
    fprintf(' Stage 8B-2 MATLAB statistical reference\n');
    fprintf(' N_ref = %d\n',N_ref);
    fprintf('===============================================\n');

    for caseIndex = 1:numel(labels)

        label = labels(caseIndex);
        idx = find(T.scenario_BR == label & T.scenario_RU == label);
        assert(~isempty(idx),'Case bulunamadi: %s',label);

        % Stochastic reference generation is expensive. Choose the smallest
        % port-complexity row for each scenario branch.
        nTAll = 2.*double(T.nT1(idx)).*double(T.nT2(idx));
        nRAll = 2.*double(T.nR1(idx)).*double(T.nR2(idx));
        nRISAll = 2.*double(T.nRIS_x(idx)).*double(T.nRIS_y(idx));
        cost = nRISAll.*(nTAll+nRAll);

        [~,j] = min(cost);
        tableRow = idx(j);
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

        lspBR = generate_lsp(row.scenario_BR,fc,ris2gnb,gnb2ris);
        lspRU = generate_lsp(row.scenario_RU,fc,ue2ris,ris2ue);

        if lspBR.isLOS, KBR = 10^(lspBR.mu_K/10); else, KBR = 0; end
        if lspRU.isLOS, KRU = 10^(lspRU.mu_K/10); else, KRU = 0; end

        rng(820000 + 1000*caseIndex + 17,'twister');
        [HBR,muBR,sigma2BR] = generate_channel_train( ...
            chBR,ris2gnb,gnb2ris,c0,lambda0,KBR,N_ref,lspBR);

        rng(820000 + 1000*caseIndex + 53,'twister');
        [HRU,muRU,sigma2RU] = generate_channel_train( ...
            chRU,ue2ris,ris2ue,c0,lambda0,KRU,N_ref,lspRU);

        BRS = summarizeLink(HBR,muBR,sigma2BR);
        RUS = summarizeLink(HRU,muRU,sigma2RU);

        % Fixed z.
        rng(920000 + sourceIndex,'twister');
        z = randi([0 1],nRIS,1);
        phaseLevels = deg2rad([45 135]);
        phi = phaseLevels(z+1).';
        [beta,gamma] = generate_ris_response(phi);

        % Fixed Type-I rank-1 W. No random pilot/SVD in B2: B2 isolates
        % channel-distribution parity. Multi-W/pilot selection belongs to B3.
        codebook = generate_codebook(2,nT1,nT2,1,1);
        nI2 = size(codebook,3);
        nI11 = size(codebook,4);
        nI12 = size(codebook,5);

        i11 = mod(caseIndex-1,nI11)+1;
        i12 = mod(2*(caseIndex-1),nI12)+1;
        i2  = mod(caseIndex-1,nI2)+1;
        WIdx = [i11 i12 i2];
        W = codebook(:,1,i2,i11,i12);

        F = generate_cascaded_ch(HBR,HRU,gamma);

        Feff = complex(zeros(N_ref,nR));
        for n = 1:N_ref
            Feff(n,:) = (squeeze(F(n,:,:))*W).';
        end

        Y = real(sum(abs(Feff).^2,2));
        YS = summarizeY(Y);

        BR_K = KBR; BR_isLOS = logical(lspBR.isLOS);
        BR_M = lspBR.M; BR_L = lspBR.L;
        BR_mu_XPR=lspBR.mu_XPR; BR_sigma_XPR=lspBR.sigma_XPR;
        BR_mu_ASA=lspBR.mu_ASA; BR_sigma_ASA=lspBR.sigma_ASA;
        BR_mu_ZSA=lspBR.mu_ZSA; BR_sigma_ZSA=lspBR.sigma_ZSA;
        BR_mu_ASD=lspBR.mu_ASD; BR_sigma_ASD=lspBR.sigma_ASD;
        BR_mu_ZSD=lspBR.mu_ZSD; BR_sigma_ZSD=lspBR.sigma_ZSD;
        BR_c_ASA=lspBR.c_ASA; BR_c_ZSA=lspBR.c_ZSA;
        BR_c_ASD=lspBR.c_ASD; BR_c_ZSD=lspBR.c_ZSD;
        BR_mu_offset_ZOD=lspBR.mu_offset_ZOD;

        RU_K = KRU; RU_isLOS = logical(lspRU.isLOS);
        RU_M = lspRU.M; RU_L = lspRU.L;
        RU_mu_XPR=lspRU.mu_XPR; RU_sigma_XPR=lspRU.sigma_XPR;
        RU_mu_ASA=lspRU.mu_ASA; RU_sigma_ASA=lspRU.sigma_ASA;
        RU_mu_ZSA=lspRU.mu_ZSA; RU_sigma_ZSA=lspRU.sigma_ZSA;
        RU_mu_ASD=lspRU.mu_ASD; RU_sigma_ASD=lspRU.sigma_ASD;
        RU_mu_ZSD=lspRU.mu_ZSD; RU_sigma_ZSD=lspRU.sigma_ZSD;
        RU_c_ASA=lspRU.c_ASA; RU_c_ZSA=lspRU.c_ZSA;
        RU_c_ASD=lspRU.c_ASD; RU_c_ZSD=lspRU.c_ZSD;
        RU_mu_offset_ZOD=lspRU.mu_offset_ZOD;

        BR_meanLeakNorm=BRS.meanLeakNorm;
        BR_varianceRatioMean=BRS.varianceRatioMean;
        BR_fourthMomentRatio=BRS.fourthMomentRatio;
        BR_realVarianceRatio=BRS.realVarianceRatio;
        BR_imagVarianceRatio=BRS.imagVarianceRatio;
        BR_pseudoCovAbsRatio=BRS.pseudoCovAbsRatio;

        RU_meanLeakNorm=RUS.meanLeakNorm;
        RU_varianceRatioMean=RUS.varianceRatioMean;
        RU_fourthMomentRatio=RUS.fourthMomentRatio;
        RU_realVarianceRatio=RUS.realVarianceRatio;
        RU_imagVarianceRatio=RUS.imagVarianceRatio;
        RU_pseudoCovAbsRatio=RUS.pseudoCovAbsRatio;

        Y_mean=YS.mean;
        Y_var=YS.var;
        Y_q01=YS.q01;
        Y_q05=YS.q05;
        Y_q10=YS.q10;
        Y_q50=YS.q50;
        Y_q90=YS.q90;
        Y_q99=YS.q99;

        caseName = regexprep(label,'[^A-Za-z0-9]+','_');
        fileName = caseName + ".mat";
        filePath = fullfile(outRoot,fileName);

        save(filePath, ...
            'scenario','N_ref','fc', ...
            'nT1','nT2','nR1','nR2','nRISx','nRISy', ...
            'gnb2ris','ris2gnb','ris2ue','ue2ris', ...
            'BR_K','BR_isLOS','BR_M','BR_L', ...
            'BR_mu_XPR','BR_sigma_XPR', ...
            'BR_mu_ASA','BR_sigma_ASA','BR_mu_ZSA','BR_sigma_ZSA', ...
            'BR_mu_ASD','BR_sigma_ASD','BR_mu_ZSD','BR_sigma_ZSD', ...
            'BR_c_ASA','BR_c_ZSA','BR_c_ASD','BR_c_ZSD', ...
            'BR_mu_offset_ZOD', ...
            'RU_K','RU_isLOS','RU_M','RU_L', ...
            'RU_mu_XPR','RU_sigma_XPR', ...
            'RU_mu_ASA','RU_sigma_ASA','RU_mu_ZSA','RU_sigma_ZSA', ...
            'RU_mu_ASD','RU_sigma_ASD','RU_mu_ZSD','RU_sigma_ZSD', ...
            'RU_c_ASA','RU_c_ZSA','RU_c_ASD','RU_c_ZSD', ...
            'RU_mu_offset_ZOD', ...
            'z','phi','beta','gamma','W','WIdx', ...
            'BR_meanLeakNorm','BR_varianceRatioMean', ...
            'BR_fourthMomentRatio','BR_realVarianceRatio', ...
            'BR_imagVarianceRatio','BR_pseudoCovAbsRatio', ...
            'RU_meanLeakNorm','RU_varianceRatioMean', ...
            'RU_fourthMomentRatio','RU_realVarianceRatio', ...
            'RU_imagVarianceRatio','RU_pseudoCovAbsRatio', ...
            'Y_mean','Y_var','Y_q01','Y_q05','Y_q10', ...
            'Y_q50','Y_q90','Y_q99', ...
            '-v7');

        manifest.caseName(caseIndex)=caseName;
        manifest.tableRow(caseIndex)=tableRow;
        manifest.sourceIndex(caseIndex)=sourceIndex;
        manifest.fc(caseIndex)=fc;
        manifest.nT(caseIndex)=nT;
        manifest.nR(caseIndex)=nR;
        manifest.nRIS(caseIndex)=nRIS;
        manifest.N_ref(caseIndex)=N_ref;
        manifest.isLOS(caseIndex)=double(lspBR.isLOS);
        manifest.file(caseIndex)=fileName;

        fprintf( ...
            '[%d/8] %-20s | nT=%d nR=%d nRIS=%d | q05=%.6g\n', ...
            caseIndex,label,nT,nR,nRIS,Y_q05);
    end

    writetable(manifest,fullfile(outRoot,'manifest.csv'));

    zipFile = outRoot + ".zip";
    if isfile(zipFile), delete(zipFile); end
    zip(zipFile,outRoot);

    fprintf('\nZIP: %s\n',zipFile);
    fprintf('===============================================\n\n');

    disp(manifest);
end


function S = summarizeLink(H,muH,sigma2H)
    N = size(H,1);
    [U,Sz] = size(muH);
    C = H - reshape(muH,[1 U Sz]);

    meanEmp = squeeze(mean(H,1));
    sig = max(real(sigma2H),eps);

    S.meanLeakNorm = ...
        norm(meanEmp-muH,'fro') / sqrt(numel(muH)*sig);

    abs2 = abs(C).^2;

    S.varianceRatioMean = mean(abs2,'all')/sig;
    S.fourthMomentRatio = mean(abs2.^2,'all')/(sig^2);
    S.realVarianceRatio = 2*mean(real(C).^2,'all')/sig;
    S.imagVarianceRatio = 2*mean(imag(C).^2,'all')/sig;
    S.pseudoCovAbsRatio = abs(mean(C.^2,'all'))/sig;
end


function S = summarizeY(Y)
    Y = real(Y(:));
    S.mean = mean(Y);
    S.var = var(Y,1);

    q = quantile(Y,[.01 .05 .10 .50 .90 .99]);

    S.q01=q(1);
    S.q05=q(2);
    S.q10=q(3);
    S.q50=q(4);
    S.q90=q(5);
    S.q99=q(6);
end
