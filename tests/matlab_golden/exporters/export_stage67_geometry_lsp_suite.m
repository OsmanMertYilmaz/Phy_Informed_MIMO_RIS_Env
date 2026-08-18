function outFile = export_stage67_geometry_lsp_suite(configCsv,outFile)
%EXPORT_STAGE67_GEOMETRY_LSP_SUITE
% Export exact MATLAB golden outputs for:
%
%   Stage 6: generate_geometry
%   Stage 7: generate_lsp
%
% Default input:
%   generate_train_scenarios_data_2000.csv
%
% Default output:
%   stage67_geometry_lsp_golden.csv
%
% Every row in the input CSV is exported. Both BR and RU LSPs are evaluated.

    if nargin < 1 || strlength(string(configCsv)) == 0
        configCsv = "generate_train_scenarios_data_2000.csv";
    end
    if nargin < 2 || strlength(string(outFile)) == 0
        outFile = "stage67_geometry_lsp_golden.csv";
    end

    configCsv = string(configCsv);
    outFile = string(outFile);

    assert(isfile(configCsv),'CSV bulunamadi: %s',configCsv);
    assert(exist('generate_geometry','file') ~= 0, ...
        'generate_geometry MATLAB path''inde bulunamadi.');
    assert(exist('generate_lsp','file') ~= 0, ...
        'generate_lsp MATLAB path''inde bulunamadi.');

    opts = detectImportOptions(configCsv);
    opts = setvartype(opts,{'scenario_BR','scenario_RU'},'string');
    T = readtable(configCsv,opts);

    n = height(T);

    % Input columns + golden outputs.
    G = table();
    G.rowIndex = (1:n).';

    if ismember('index',T.Properties.VariableNames)
        G.sourceIndex = double(T.index);
    else
        G.sourceIndex = G.rowIndex;
    end

    G.fc = double(T.fc);
    G.scenario_BR = string(T.scenario_BR);
    G.scenario_RU = string(T.scenario_RU);

    coordNames = { ...
        'ris_x','ris_y','ris_z', ...
        'gnb_x','gnb_y','gnb_z', ...
        'ue_x','ue_y','ue_z'};

    for k = 1:numel(coordNames)
        G.(coordNames{k}) = double(T.(coordNames{k}));
    end

    vectorFields = {'gnb2ris','ris2gnb','ris2ue','ue2ris'};
    for k = 1:numel(vectorFields)
        name = vectorFields{k};
        G.([name '_x']) = zeros(n,1);
        G.([name '_y']) = zeros(n,1);
        G.([name '_z']) = zeros(n,1);
    end
    G.dist_gnb2ris = zeros(n,1);
    G.dist_ris2ue = zeros(n,1);

    floatFields = { ...
        'mu_K','sigma_K','mu_XPR','sigma_XPR', ...
        'mu_ASA','sigma_ASA','mu_ZSA','sigma_ZSA', ...
        'mu_ASD','sigma_ASD','mu_ZSD','sigma_ZSD', ...
        'c_ASA','c_ZSA','c_ASD','c_ZSD','mu_offset_ZOD'};

    prefixes = {'BR','RU'};
    for p = 1:numel(prefixes)
        prefix = prefixes{p};
        for k = 1:numel(floatFields)
            G.([prefix '_' floatFields{k}]) = zeros(n,1);
        end
        G.([prefix '_M']) = zeros(n,1);
        G.([prefix '_L']) = zeros(n,1);
        G.([prefix '_isLOS']) = false(n,1);
        G.([prefix '_K_linear']) = zeros(n,1);
    end

    for i = 1:n
        row = T(i,:);

        geometry.ris = double([row.ris_x,row.ris_y,row.ris_z]);
        geometry.gnb = double([row.gnb_x,row.gnb_y,row.gnb_z]);
        geometry.ue  = double([row.ue_x,row.ue_y,row.ue_z]);
        geometry = generate_geometry(geometry);

        for k = 1:numel(vectorFields)
            name = vectorFields{k};
            v = double(geometry.(name));
            G.([name '_x'])(i) = v(1);
            G.([name '_y'])(i) = v(2);
            G.([name '_z'])(i) = v(3);
        end
        G.dist_gnb2ris(i) = double(geometry.dist_gnb2ris);
        G.dist_ris2ue(i) = double(geometry.dist_ris2ue);

        fc = double(row.fc);

        lspBR = generate_lsp( ...
            string(row.scenario_BR),fc, ...
            geometry.ris2gnb,geometry.gnb2ris);

        lspRU = generate_lsp( ...
            string(row.scenario_RU),fc, ...
            geometry.ue2ris,geometry.ris2ue);

        lspList = {lspBR,lspRU};

        for p = 1:2
            prefix = prefixes{p};
            lsp = lspList{p};

            for k = 1:numel(floatFields)
                fieldName = floatFields{k};
                G.([prefix '_' fieldName])(i) = double(lsp.(fieldName));
            end

            G.([prefix '_M'])(i) = double(lsp.M);
            G.([prefix '_L'])(i) = double(lsp.L);
            G.([prefix '_isLOS'])(i) = logical(lsp.isLOS);

            if lsp.isLOS
                G.([prefix '_K_linear'])(i) = 10^(double(lsp.mu_K)/10);
            else
                G.([prefix '_K_linear'])(i) = 0;
            end
        end
    end

    writetable(G,outFile);

    fprintf('\n=========================================\n');
    fprintf(' Stage 6 + 7 golden export completed\n');
    fprintf('=========================================\n');
    fprintf('Input rows : %d\n',n);
    fprintf('Output     : %s\n',outFile);
    fprintf('BR labels  : %d unique\n',numel(unique(G.scenario_BR)));
    fprintf('RU labels  : %d unique\n',numel(unique(G.scenario_RU)));
    fprintf('=========================================\n\n');
end
