sequences = ["sunset1","sunset2","daytime","morning","midnigt","sunrise"];
PathOrigin = [23781.0381082358,-39334.9893085574];

for seq = sequences
    ninterval = 10;
    prefix = "/home/jhlee/data/brisbane_vpr/";
    path_root = strcat(prefix,"datasets/",seq);
    
    loc_orig = readmatrix(strcat(path_root,"/location.txt"));
    nImages = size(loc_orig,1);

    llt = load(strcat(path_root,"/",seq,".txt"));
    xy = zeros(length(llt),2);
    for i = 1:length(llt)
        xy(i,:) = 0.01*gpstoxy(llt(i,2), llt(i,1));
    end
    xy = xy - PathOrigin;
    gttime = llt(:,3);
    fid = fopen(strcat(prefix,"datasets/",seq,"location.txt"),'w');

    for im_num = 1:nImages
        
        im_t = loc_orig(im_num,3);
        
        xyq = interp1(gttime,xy,im_t);
        fprintf(fid,"%.6f %.6f %.6f\n",[xyq,im_t]);
        fprintf("%s : %2.2f percent!\n",seq,100*im_num / nImages);
    end
end


function r = gpstoxy(lon, lat)
R_equ = 6378.137e3;      % Earth's radius [m]; WGS-84
f     = 1/298.257223563; % Flattening; WGS-84
e2     = f*(2-f);        % Square of eccentricity
CosLat = cos(lat);       % (Co)sine of geodetic latitude
SinLat = sin(lat);
% Position vector 
N = R_equ/sqrt(1-e2*SinLat*SinLat);
r(1) = (N)*CosLat*cos(lon);
r(2) = (N)*CosLat*sin(lon);
end