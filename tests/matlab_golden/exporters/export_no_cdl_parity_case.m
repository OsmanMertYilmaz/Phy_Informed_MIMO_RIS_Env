function export_no_cdl_parity_case( ...
    outFile,channel,aVector,dVector,c0,lambda0,K,lsp)
%EXPORT_NO_CDL_PARITY_CASE Save one MATLAB golden case for Python parity.
%
% Example:
%
% export_no_cdl_parity_case( ...
%     "gnb2ris_case.mat", ...
%     gnb2ris, ...
%     geometry.ris2gnb, ...
%     geometry.gnb2ris, ...
%     c0,lambda_0,K_BR,lsp_BR);
%
% The output is saved as -v7 so scipy.io.loadmat can read it directly.

    arguments
        outFile
        channel
        aVector
        dVector
        c0
        lambda0
        K
        lsp
    end

    [muH,sigma2H,dbarT,dbarR] = ...
        generate_channel_moments_no_cdl( ...
            channel,aVector,dVector,c0,lambda0,K,lsp);

    txSize = double(channel.TransmitAntennaArray.Size);
    rxSize = double(channel.ReceiveAntennaArray.Size);

    txSpacing = double(channel.TransmitAntennaArray.ElementSpacing);
    rxSpacing = double(channel.ReceiveAntennaArray.ElementSpacing);

    txPolAngles = double(channel.TransmitAntennaArray.PolarizationAngles);
    rxPolAngles = double(channel.ReceiveAntennaArray.PolarizationAngles);

    txArrayOrientation = double(channel.TransmitArrayOrientation);
    rxArrayOrientation = double(channel.ReceiveArrayOrientation);

    fc = double(channel.CarrierFrequency);
    K = double(K);
    c0 = double(c0);
    lambda0 = double(lambda0);

    muXPR = double(lsp.mu_XPR);
    sigmaXPR = double(lsp.sigma_XPR);

    aVector = double(aVector(:));
    dVector = double(dVector(:));

    % Probe the internal isotropic power pattern. If this internal symbol is
    % unavailable in a future MATLAB release, save NaN and rely on final
    % muH parity instead.
    AisoProbe = nan(1,4);
    try
        AisoProbe = double([ ...
            wireless.internal.channelmodels.A_isotropic(0,0), ...
            wireless.internal.channelmodels.A_isotropic(30,40), ...
            wireless.internal.channelmodels.A_isotropic(90,-120), ...
            wireless.internal.channelmodels.A_isotropic(150,179)]);
    catch ME
        warning("A_isotropic probe skipped: %s",ME.message);
    end

    % Hard guards for the current Python Stage-1 implementation.
    assert(isequal(txSize(3:5),[2 1 1]), ...
        'Python Stage-1 expects Tx Size=[M N 2 1 1].');
    assert(isequal(rxSize(3:5),[2 1 1]), ...
        'Python Stage-1 expects Rx Size=[M N 2 1 1].');

    assert(all(abs(txSpacing(1:2)-[0.5 0.5])<1e-15), ...
        'Python Stage-1 expects Tx dV=dH=0.5 lambda.');
    assert(all(abs(rxSpacing(1:2)-[0.5 0.5])<1e-15), ...
        'Python Stage-1 expects Rx dV=dH=0.5 lambda.');

    assert(isequal(txPolAngles,[45 -45]), ...
        'Python Stage-1 expects Tx polarization [+45 -45].');
    assert(isequal(rxPolAngles,[45 -45]), ...
        'Python Stage-1 expects Rx polarization [+45 -45].');

    assert(all(abs(txArrayOrientation(:))<1e-15), ...
        'Python Stage-1 expects Tx array orientation [0 0 0].');
    assert(all(abs(rxArrayOrientation(:))<1e-15), ...
        'Python Stage-1 expects Rx array orientation [0 0 0].');

    save(outFile, ...
        'txSize','rxSize', ...
        'txSpacing','rxSpacing', ...
        'txPolAngles','rxPolAngles', ...
        'txArrayOrientation','rxArrayOrientation', ...
        'fc','K','c0','lambda0', ...
        'muXPR','sigmaXPR', ...
        'aVector','dVector', ...
        'AisoProbe', ...
        'muH','sigma2H','dbarT','dbarR', ...
        '-v7');

    fprintf('\nGolden parity case saved:\n  %s\n',string(outFile));
    fprintf('Tx ports : %d\n',prod(txSize));
    fprintf('Rx ports : %d\n',prod(rxSize));
    fprintf('muH size : %s\n',mat2str(size(muH)));
    fprintf('sigma2H  : %.16g\n',sigma2H);
end
