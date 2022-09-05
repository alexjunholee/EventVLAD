% sequences = ["sunset1","sunset2","daytime","morning","midnigt","sunrise"];
sequences = ["sunset2","daytime","morning","midnigt","sunrise"];
PathOrigin = [23781.0381082358,-39334.9893085574];

for seq = sequences
    ninterval = 10;
    prefix = "/media/jhlee/1TBNVME/data/brisbane_vpr/";
    fprintf("Reading source...\n")
    filename = strcat(prefix,"datasets/",seq);
    bagload = rosbag(strcat(prefix,"bagfile/",seq,'.bag'));
    imageMsg = select(bagload,'Topic','/dvs/image_raw');
    image_struct = readMessages(imageMsg,1:ninterval:imageMsg.NumMessages);
    fprintf("Done\n")
    
    nImages = floor(length(image_struct));

    path_root = strcat(prefix,"datasets/",seq);
    llt = load(strcat(path_root,"/",seq,".txt"));
    xy = zeros(length(llt),2);
    for i = 1:length(llt)
        xy(i,:) = 0.01*gpstoxy(llt(i,2), llt(i,1));
    end
    xy = xy - PathOrigin;
    TimeOrigin = imageMsg.StartTime;
    gttime = llt(:,3);
    fid = fopen(strcat(path_root,"/location.txt"),'w');
    [~,~] = mkdir(path_root);
    [~,~] = mkdir(strcat(path_root,'/evt'));
    [~,~] = mkdir(strcat(path_root,'/evt/0'));
    [~,~] = mkdir(strcat(path_root,'/evt/1'));
    [~,~] = mkdir(strcat(path_root,'/evt/2'));
    [~,~] = mkdir(strcat(path_root,'/img'));

    for im_num = 1:nImages
        eventarray.x = [];
        eventarray.y = [];
        eventarray.t = [];
        eventarray.p = logical([]);
        im_t = stamp2sec(image_struct{im_num}.Header.Stamp);
        xyq = interp1(gttime,xy,im_t - TimeOrigin);
        if any(isnan(xyq))
            continue;
        end
        
        t0_e = im_t - 0.11;
        t1_e = im_t + 0.11;
        eventMsg = select(bagload,'Topic','/dvs/events','Time',[t0_e,t1_e]);
        event_struct = readMessages(eventMsg,'DataFormat','struct');
        for i = 1:length(event_struct)
            timestamps = zeros(length(event_struct{i,1}.Events),1);
            for j = 1:length(event_struct{i,1}.Events)
                timestamps(j) = stamp2sec(event_struct{i,1}.Events(j).Ts);
            end
            [timestamps,idx] = sort(timestamps);
            eventarray.x = cat(1,eventarray.x,event_struct{i,1}.Events(idx).X);
            eventarray.y = cat(1,eventarray.y,event_struct{i,1}.Events(idx).Y);
            eventarray.p = cat(1,eventarray.p,logical([event_struct{i,1}.Events(idx).Polarity])');
            eventarray.t = cat(1,eventarray.t,timestamps);
        end
        ok = abs(eventarray.t - im_t) < 0.075; %50ms;
        eventarray.x = eventarray.x(ok);
        eventarray.y = eventarray.y(ok);
        eventarray.p = eventarray.p(ok);
        eventarray.t = eventarray.t(ok);
        if length(eventarray.t) < 260*346*0.01
            continue;
        end
        
        n_e = floor(length(eventarray.p)/3);
        idx0 = uv2idx(eventarray.x(1:n_e)+1,      eventarray.y(1:n_e)+1);
        idx1 = uv2idx(eventarray.x(n_e:2*n_e)+1,  eventarray.y(n_e:2*n_e)+1);
        idx2 = uv2idx(eventarray.x(2*n_e:3*n_e)+1,eventarray.y(2*n_e:3*n_e)+1);
        
        idx0_p = idx0(eventarray.p(1:n_e));
        idx0_n = idx0(~eventarray.p(1:n_e));
        idx1_p = idx1(eventarray.p(n_e:2*n_e));
        idx1_n = idx1(~eventarray.p(n_e:2*n_e));
        idx2_p = idx2(eventarray.p(2*n_e:3*n_e));
        idx2_n = idx2(~eventarray.p(2*n_e:3*n_e));
        
        dvs_img0 = zeros(260,346,3);
        dvs_img0p = zeros(346,260);
        dvs_img0n = zeros(346,260);
        dvs_img1 = zeros(260,346,3);
        dvs_img1p = zeros(346,260);
        dvs_img1n = zeros(346,260);
        dvs_img2 = zeros(260,346,3);
        dvs_img2p = zeros(346,260);
        dvs_img2n = zeros(346,260);
        
        dvs_img0p(idx0_p) = 1;
        dvs_img0n(idx0_n) = 1;
        dvs_img0(:,:,1) = dvs_img0p';
        dvs_img0(:,:,3) = dvs_img0n';
        dvs_img1p(idx1_p) = 1;
        dvs_img1n(idx1_n) = 1;
        dvs_img1(:,:,1) = dvs_img1p';
        dvs_img1(:,:,3) = dvs_img1n';
        dvs_img2p(idx2_p) = 1;
        dvs_img2n(idx2_n) = 1;
        dvs_img2(:,:,1) = dvs_img2p';
        dvs_img2(:,:,3) = dvs_img2n';
        
        rawimg = rgb2gray(im2double(readImage(image_struct{im_num})));
        if size(rawimg,2) ~= 346
            continue
        end
        imwrite(rawimg(3:258,13:332),sprintf(strcat(path_root,"/img/%.6f.png"),im_t - TimeOrigin));
        imwrite(dvs_img0(3:258,13:332,:),sprintf(strcat(path_root,"/evt/0/%.6f.png"),im_t - TimeOrigin));
        imwrite(dvs_img1(3:258,13:332,:),sprintf(strcat(path_root,"/evt/1/%.6f.png"),im_t - TimeOrigin));
        imwrite(dvs_img2(3:258,13:332,:),sprintf(strcat(path_root,"/evt/2/%.6f.png"),im_t - TimeOrigin));
        
        fprintf(fid,"%.6f %.6f %.6f\n",[xyq,im_t - TimeOrigin]);
        fprintf("%s : %2.2f percent!\n",seq,100*im_num / nImages);
    end
end


function idx = uv2idx(u,v)
idx = int64(u) + int64(v-1)*346;
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


function sec = stamp2sec(stamp)
sec = double(stamp.Sec) + 1e-9*double(stamp.Nsec);
end
