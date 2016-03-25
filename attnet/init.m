% Set lib path only.
global path;
path.lib.matConvNet = '/nickel/lib/matconvnet-1.0-beta12_cudnn0/';
path.lib.vocDevKit = '/nickel/datain/PASCALVOC/VOCdevkit/';
% Set dst dir.
path.dstDir = '/nickel/dataout/attnet';
% Set image DB path only.
path.db.voc2007.name = 'VOC2007';
path.db.voc2007.funh = @DB_VOC2007;
path.db.voc2007.root = fullfile( path.lib.vocDevKit, 'VOC2007' );
% Set pre-trained CNN path only.
path.net.vgg_m.name = 'VGGM';
path.net.vgg_m.path = '/nickel/nets/mat/imagenet-vgg-m.mat';
% Do not touch the following codes.
run( fullfile( path.lib.matConvNet, 'matlab/vl_setupnn.m' ) );  % MatConvnet.
addpath( fullfile( path.lib.vocDevKit, 'VOCcode' ) );           % VOC dev kit.