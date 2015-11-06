function [ I, D ] = dgsknn( X, k, q, s, options )

  [ N, d ] = size( X );

  switch nargin
    case 2
      q = 1:N;
      s = 1:N;
      options = 2;
    case 4
      options = 2;
    case 5
      % Do nothing
    otherwise
      disp( 'dgsknn() error: inputs must be 2, 4 or 5 arguments.' );
  end

  m = length( q );
  n = length( s );

  % Compute X2
  X2 = zeros( 1, N );
  for i=1:N
    X2( 1, i ) = X( i, : ) * X( i, : )';
  end

  display( 'dgsknn calling dgsknn_matlab.' );

  % Call the mex wrapper.
  [ I, D ] =  dgsknn_matlab( m, n, d, k, q, s, X, X2 );
