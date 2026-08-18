function manifest = export_stage4_parity_suite(outRoot)
%EXPORT_STAGE4_PARITY_SUITE
% Golden suite for:
%
%   z -> phi -> beta -> gamma
%
% Official phase levels:
%
%   [45 135] degrees
%
% Tests:
%
%   nRIS = 64, 128, 256, 512
%
% For each nRIS:
%   - all zero
%   - all one
%   - alternating 0101...
%   - alternating 1010...
%   - 28 deterministic random patterns
%
% Total = 32 patterns per nRIS.
%
% Output:
%   stage4_ris_golden/
%       nRIS_64.mat
%       nRIS_128.mat
%       nRIS_256.mat
%       nRIS_512.mat
%       manifest.csv
%
% and:
%   stage4_ris_golden.zip

    if nargin < 1 || strlength(string(outRoot)) == 0
        outRoot = "stage4_ris_golden";
    end

    outRoot = string(outRoot);

    assert(exist('generate_ris_response','file') ~= 0, ...
        'generate_ris_response MATLAB path''inde bulunamadi.');

    if isfolder(outRoot)
        rmdir(outRoot,'s');
    end
    mkdir(outRoot);

    nRISValues = [64 128 256 512];
    phaseLevels = deg2rad([45 135]);
    nPatterns = 32;

    manifest = table( ...
        'Size',[numel(nRISValues) 3], ...
        'VariableTypes',{'double','double','string'}, ...
        'VariableNames',{'nRIS','nPatterns','file'});

    fprintf('\n===================================\n');
    fprintf(' Stage 4 RIS response golden suite\n');
    fprintf('===================================\n');
    fprintf('Phase levels: [45 135] deg\n\n');

    for k = 1:numel(nRISValues)

        nRIS = nRISValues(k);

        Z = zeros(nPatterns,nRIS);

        % 1) all zero
        Z(1,:) = 0;

        % 2) all one
        Z(2,:) = 1;

        % 3) 0101...
        Z(3,:) = mod(0:nRIS-1,2);

        % 4) 1010...
        Z(4,:) = 1-Z(3,:);

        % Remaining deterministic random patterns.
        rng(910000+nRIS,'twister');
        Z(5:end,:) = randi([0 1],nPatterns-4,nRIS);

        phi = phaseLevels(Z+1);

        beta = zeros(size(phi));
        gamma = complex(zeros(size(phi)));

        % generate_ris_response vectorizes by flattening phi internally.
        [betaVec,gammaVec] = generate_ris_response(phi(:));

        beta(:) = betaVec;
        gamma(:) = gammaVec;

        fileName = sprintf('nRIS_%d.mat',nRIS);
        filePath = fullfile(outRoot,fileName);

        save(filePath, ...
            'nRIS','nPatterns','phaseLevels', ...
            'Z','phi','beta','gamma', ...
            '-v7');

        manifest.nRIS(k) = nRIS;
        manifest.nPatterns(k) = nPatterns;
        manifest.file(k) = string(fileName);

        fprintf('nRIS=%3d | patterns=%d | saved=%s\n', ...
            nRIS,nPatterns,fileName);
    end

    writetable(manifest,fullfile(outRoot,'manifest.csv'));

    zipFile = outRoot + ".zip";
    if isfile(zipFile)
        delete(zipFile);
    end
    zip(zipFile,outRoot);

    fprintf('\nManifest: %s\n',fullfile(outRoot,'manifest.csv'));
    fprintf('ZIP     : %s\n',zipFile);
    fprintf('===================================\n\n');

    disp(manifest);
end
