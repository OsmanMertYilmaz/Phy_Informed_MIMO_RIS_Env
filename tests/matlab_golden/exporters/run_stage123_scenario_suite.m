function manifest = run_stage123_scenario_suite(configCsv,outRoot)
%RUN_STAGE123_SCENARIO_SUITE
% Validate Stage 1, Stage 2 and Stage 3 golden outputs across:
%
%   Indoor-Office LOS / NLOS
%   UMi           LOS / NLOS
%   UMa           LOS / NLOS
%   RMa           LOS / NLOS
%
% Exactly 8 cases are selected.
%
% For a clean scenario-isolated test, each selected CSV row must satisfy:
%
%   scenario_BR == scenario_RU == requested scenario-condition
%
% Therefore both hops of one test bank use the same scenario-condition.
%
% Outputs per case:
%
%   <case>/stage1_br.mat
%   <case>/stage1_ru.mat
%   <case>/stage2.mat
%   <case>/stage3.mat
%
% Plus:
%
%   manifest.csv
%   <outRoot>.zip
%
% Defaults:
%
%   configCsv = "generate_train_scenarios_data_2000.csv"
%   outRoot   = "stage123_scenario_golden"
%
% Required project functions:
%
%   generate_geometry
%   generate_lsp
%   generate_channel_moments_no_cdl
%   compute_ch_rho_avg
%   compute_ch_eff_rho_avg_fast
%   generate_ris_response
%   generate_eff_moments
%   evaluate_gamma_metric
%
% Required parity exporters:
%
%   export_no_cdl_parity_case
%   export_rho_parity_case
%   export_stage3_parity_case

    if nargin < 1 || strlength(string(configCsv)) == 0
        configCsv = "generate_train_scenarios_data_2000.csv";
    end
    if nargin < 2 || strlength(string(outRoot)) == 0
        outRoot = "stage123_scenario_golden";
    end

    configCsv = string(configCsv);
    outRoot = string(outRoot);

    assert(isfile(configCsv), ...
        'Scenario CSV bulunamadi: %s',configCsv);

    requiredFunctions = { ...
        'generate_geometry', ...
        'generate_lsp', ...
        'generate_channel_moments_no_cdl', ...
        'compute_ch_rho_avg', ...
        'compute_ch_eff_rho_avg_fast', ...
        'generate_ris_response', ...
        'generate_eff_moments', ...
        'evaluate_gamma_metric', ...
        'export_no_cdl_parity_case', ...
        'export_rho_parity_case', ...
        'export_stage3_parity_case'};

    for k = 1:numel(requiredFunctions)
        assert(exist(requiredFunctions{k},'file') ~= 0, ...
            'Gerekli fonksiyon MATLAB path''inde yok: %s', ...
            requiredFunctions{k});
    end

    if isfolder(outRoot)
        rmdir(outRoot,'s');
    end
    mkdir(outRoot);

    opts = detectImportOptions(configCsv);
    opts = setvartype(opts,{'scenario_BR','scenario_RU'},'string');
    T = readtable(configCsv,opts);

    requiredCols = { ...
        'fc','scenario_BR','scenario_RU', ...
        'ris_x','ris_y','ris_z', ...
        'gnb_x','gnb_y','gnb_z', ...
        'ue_x','ue_y','ue_z', ...
        'nT1','nT2','nR1','nR2','nRIS_x','nRIS_y'};

    missing = setdiff(requiredCols,T.Properties.VariableNames);
    assert(isempty(missing), ...
        'Scenario CSV eksik kolonlar iceriyor: %s', ...
        strjoin(missing,', '));

    labels = [ ...
        "Indoor-Office-LOS"
        "Indoor-Office-NLOS"
        "UMi-LOS"
        "UMi-NLOS"
        "UMa-LOS"
        "UMa-NLOS"
        "RMa-LOS"
        "RMa-NLOS"];

    cLight = physconst('LightSpeed');

    manifest = table( ...
        'Size',[numel(labels) 13], ...
        'VariableTypes', { ...
            'string','string','string','double','double', ...
            'double','double','double','double','double', ...
            'double','double','double'}, ...
        'VariableNames', { ...
            'caseName','scenario','condition','tableRow','sourceIndex', ...
            'fc','nT','nR','nRIS','nT1','nT2','nRIS_x','nRIS_y'});

    fprintf('\n==============================================\n');
    fprintf(' Stage 1-2-3 scenario parity golden generator\n');
    fprintf('==============================================\n');
    fprintf('CSV    : %s\n',configCsv);
    fprintf('Output : %s\n\n',outRoot);

    for caseIndex = 1:numel(labels)

        label = labels(caseIndex);

        mask = T.scenario_BR == label & T.scenario_RU == label;
        tableRow = find(mask,1,'first');

        assert(~isempty(tableRow), ...
            ['CSV icinde scenario_BR == scenario_RU == %s ' ...
             'olan bir satir bulunamadi.'],label);

        row = T(tableRow,:);

        if ismember('index',T.Properties.VariableNames)
            sourceIndex = double(row.index);
        else
            sourceIndex = tableRow;
        end

        fc = double(row.fc);
        lambda0 = cLight/fc;

        geometry.ris = double([row.ris_x,row.ris_y,row.ris_z]);
        geometry.gnb = double([row.gnb_x,row.gnb_y,row.gnb_z]);
        geometry.ue  = double([row.ue_x,row.ue_y,row.ue_z]);
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

        % --- MATLAB channel array containers ---
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

        % --- Current project LSP path ---
        lspBR = generate_lsp( ...
            row.scenario_BR,fc,geometry.ris2gnb,geometry.gnb2ris);

        lspRU = generate_lsp( ...
            row.scenario_RU,fc,geometry.ue2ris,geometry.ris2ue);

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

        expectedLOS = endsWith(label,"-LOS");
        assert(logical(lspBR.isLOS) == expectedLOS, ...
            'BR isLOS scenario etiketiyle uyusmuyor: %s',label);
        assert(logical(lspRU.isLOS) == expectedLOS, ...
            'RU isLOS scenario etiketiyle uyusmuyor: %s',label);

        caseName = regexprep(label,'[^A-Za-z0-9]+','_');
        caseDir = fullfile(outRoot,caseName);
        mkdir(caseDir);

        fprintf('[%d/8] %-20s | row=%d | fc=%.2f GHz | nT=%d nR=%d nRIS=%d\n', ...
            caseIndex,label,tableRow,fc/1e9,nT,nR,nRIS);

        % -------- STAGE 1: both hops --------
        export_no_cdl_parity_case( ...
            fullfile(caseDir,'stage1_br.mat'), ...
            gnb2ris, ...
            geometry.ris2gnb, ...
            geometry.gnb2ris, ...
            cLight,lambda0,KBR,lspBR);

        export_no_cdl_parity_case( ...
            fullfile(caseDir,'stage1_ru.mat'), ...
            ris2ue, ...
            geometry.ue2ris, ...
            geometry.ris2ue, ...
            cLight,lambda0,KRU,lspRU);

        % -------- STAGE 2 --------
        export_rho_parity_case( ...
            fullfile(caseDir,'stage2.mat'), ...
            gnb2ris,ris2ue,geometry,cLight,lambda0, ...
            KBR,KRU,lspBR,lspRU);

        % -------- STAGE 3 --------
        export_stage3_parity_case( ...
            fullfile(caseDir,'stage3.mat'), ...
            gnb2ris,ris2ue,geometry,cLight,lambda0, ...
            KBR,KRU,lspBR,lspRU);

        if expectedLOS
            condition = "LOS";
        else
            condition = "NLOS";
        end

        scenario = erase(label,"-" + condition);

        manifest.caseName(caseIndex) = caseName;
        manifest.scenario(caseIndex) = scenario;
        manifest.condition(caseIndex) = condition;
        manifest.tableRow(caseIndex) = tableRow;
        manifest.sourceIndex(caseIndex) = sourceIndex;
        manifest.fc(caseIndex) = fc;
        manifest.nT(caseIndex) = nT;
        manifest.nR(caseIndex) = nR;
        manifest.nRIS(caseIndex) = nRIS;
        manifest.nT1(caseIndex) = nT1;
        manifest.nT2(caseIndex) = nT2;
        manifest.nRIS_x(caseIndex) = nRISx;
        manifest.nRIS_y(caseIndex) = nRISy;
    end

    writetable(manifest,fullfile(outRoot,'manifest.csv'));

    zipFile = outRoot + ".zip";
    if isfile(zipFile)
        delete(zipFile);
    end
    zip(zipFile,outRoot);

    fprintf('\n==============================================\n');
    fprintf(' 8/8 golden cases tamamlandi.\n');
    fprintf(' Manifest : %s\n',fullfile(outRoot,'manifest.csv'));
    fprintf(' ZIP      : %s\n',zipFile);
    fprintf('==============================================\n\n');

    disp(manifest);
end
