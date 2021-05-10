function mask = get_mask(imname, th)
if nargin < 2, th = 0.01; end
im = imread(imname);
if isa(im,'uint8') 
    mask = double(im) >= th*255.0;
    if max(im(:)) <= 1
        warning('%s max = %d - double-check!\n', imname, max(im(:)));
    end
    return;
end

if isa(im,'uint16') 
    mask = double(im) >= th*65535.0;
    if max(im(:)) <= 1
        warning('%s max = %d - double-check!\n', imname, max(im(:)));
    end
    return;
end



if isa(im,'logical') 
    mask = im;
    return ;
end

error('not implemented');
end
