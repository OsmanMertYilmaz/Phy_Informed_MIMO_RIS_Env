function manifest = export_stage5_precoder_suite(outRoot)
%EXPORT_STAGE5_PRECODER_SUITE
% Golden suite for current Type-I rank-1 path:
%
%   generate_codebook(2,N1,N2,1,1)
%   selection_precoder(codebook,1,V)
%
% Tests:
%   (N1,N2) = (1,1), (2,1), (4,1), (2,2), (4,2)
%
% For each array:
%   1) export every rank-1 codeword in explicit MATLAB loop order
%   2) export 32 deterministic random H matrices
%   3) MATLAB SVD -> V(:,1)
%   4) MATLAB selection_precoder -> W,WIdx
%
% Python then tests:
%   - codebook numerical parity
%   - exact index map parity
%   - selector parity using MATLAB V(:,1)
%   - end-to-end Python SVD + selector WIdx parity

    if nargin < 1 || strlength(string(outRoot)) == 0
        outRoot = "stage5_precoder_golden";
    end

    outRoot = string(outRoot);

    assert(exist('generate_codebook','file') ~= 0, ...
        'generate_codebook MATLAB path''inde yok.');
    assert(exist('selection_precoder','file') ~= 0, ...
        'selection_precoder MATLAB path''inde yok.');

    if isfolder(outRoot)
        rmdir(outRoot,'s');
    end
    mkdir(outRoot);

    XP = 2;
    cb_mode = 1;
    nl = 1;
    nTrials = 32;

    cases = [ ...
        1 1
        2 1
        4 1
        2 2
        4 2];

    manifest = table( ...
        'Size',[size(cases,1) 5], ...
        'VariableTypes',{'double','double','double','double','string'}, ...
        'VariableNames',{'N1','N2','nT','nTrials','file'});

    fprintf('\n========================================\n');
    fprintf(' Stage 5 Type-I rank-1 precoder suite\n');
    fprintf('========================================\n');
    fprintf('XP=2, cb_mode=1, nl=1\n\n');

    for caseIndex = 1:size(cases,1)

        N1 = cases(caseIndex,1);
        N2 = cases(caseIndex,2);
        nT = XP*N1*N2;

        codebook = generate_codebook(XP,N1,N2,cb_mode,nl);

        nI2 = size(codebook,3);
        nI11 = size(codebook,4);
        nI12 = size(codebook,5);

        nCodewords = nI2*nI11*nI12;

        CBflat = complex(zeros(nT,nCodewords));
        CBidx = zeros(nCodewords,3);

        q = 0;
        for i11 = 1:nI11
            for i12 = 1:nI12
                for i2 = 1:nI2
                    q = q+1;
                    CBflat(:,q) = codebook(:,1,i2,i11,i12);
                    CBidx(q,:) = [i11 i12 i2];
                end
            end
        end

        Hstack = complex(zeros(nT,nT,nTrials));
        V1MAT = complex(zeros(nT,nTrials));
        WMAT = complex(zeros(nT,nTrials));
        WIdxMAT = zeros(nTrials,3);

        rng(850000 + 100*N1 + N2,'twister');

        for trialIndex = 1:nTrials

            H = randn(nT,nT)+1i*randn(nT,nT);

            % Make accidental singular-value degeneracy even less likely.
            H = H + diag((1:nT)*1e-3);

            [~,~,V] = svd(H,'econ');
            [W,WIdx] = selection_precoder(codebook,1,V);

            Hstack(:,:,trialIndex) = H;
            V1MAT(:,trialIndex) = V(:,1);
            WMAT(:,trialIndex) = W(:,1);
            WIdxMAT(trialIndex,:) = WIdx;
        end

        fileName = sprintf('N1_%d_N2_%d.mat',N1,N2);
        filePath = fullfile(outRoot,fileName);

        save(filePath, ...
            'XP','N1','N2','cb_mode','nl','nT','nTrials', ...
            'CBflat','CBidx', ...
            'Hstack','V1MAT','WMAT','WIdxMAT', ...
            '-v7');

        manifest.N1(caseIndex) = N1;
        manifest.N2(caseIndex) = N2;
        manifest.nT(caseIndex) = nT;
        manifest.nTrials(caseIndex) = nTrials;
        manifest.file(caseIndex) = string(fileName);

        fprintf( ...
            'N1=%d N2=%d | nT=%2d | codewords=%4d | trials=%d\n', ...
            N1,N2,nT,nCodewords,nTrials);
    end

    writetable(manifest,fullfile(outRoot,'manifest.csv'));

    zipFile = outRoot + ".zip";
    if isfile(zipFile)
        delete(zipFile);
    end
    zip(zipFile,outRoot);

    fprintf('\nManifest: %s\n',fullfile(outRoot,'manifest.csv'));
    fprintf('ZIP     : %s\n',zipFile);
    fprintf('========================================\n\n');

    disp(manifest);
end
