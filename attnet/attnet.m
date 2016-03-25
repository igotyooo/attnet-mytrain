%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpus                                    = 1;
setting.db                                      = path.db.voc2007;
setting.io.tsDb.numScaling                      = 24;
setting.io.tsDb.dilate                          = 1 / 4;
setting.io.tsDb.normalizeImageMaxSide           = 500;
setting.io.tsDb.maximumImageSize                = 9e6;
setting.io.tsDb.posGotoMargin                   = 2.4;
setting.io.tsDb.numQuantizeBetweenStopAndGoto   = 3;
setting.io.tsDb.negIntOverObjLessThan           = 0.1;
setting.io.net.pretrainedNetName                = path.net.vgg_m.name;
setting.io.net.suppressPretrainedLayerLearnRate = 1 / 4;
setting.io.general.shuffleSequance              = false; 
setting.io.general.batchSize                    = 128;
setting.io.general.numGoSmaplePerObj            = 1;
setting.io.general.numAnyDirectionSmaplePerObj  = 14; 2; 
setting.io.general.numStopSmaplePerObj          = 1;
setting.io.general.numBackgroundSmaplePerObj    = 16; 4; 
setting.net.normalizeImage                      = 'NONE';
setting.net.weightDecay                         = 0.0005;
setting.net.momentum                            = 0.9;
setting.net.modelType                           = 'dropout';
setting.net.learningRate                        = [ 0.01 * ones( 1, 8 ), 0.001 * ones( 1, 2 ) ];
setting.attNetProp.flip                         = false; 
setting.attNetProp.numScaling                   = setting.io.tsDb.numScaling;
setting.attNetProp.dilate                       = setting.io.tsDb.dilate * 2;
setting.attNetProp.normalizeImageMaxSide        = setting.io.tsDb.normalizeImageMaxSide;
setting.attNetProp.maximumImageSize             = setting.io.tsDb.maximumImageSize;
setting.attNetProp.posGotoMargin                = setting.io.tsDb.posGotoMargin;
setting.attNetProp.numTopClassification         = 3;
setting.attNetProp.numTopDirection              = 1; 2;
setting.attNetProp.directionVectorSize          = 30;
setting.attNetProp.minNumDetectionPerClass      = 0;
setting.attNetDet0.type                         = 'DYNAMIC';
setting.attNetDet0.rescaleBox                   = 1;
setting.attNetDet0.numTopClassification         = setting.attNetProp.numTopClassification;
setting.attNetDet0.numTopDirection              = setting.attNetProp.numTopDirection;
setting.attNetDet0.directionVectorSize          = setting.attNetProp.directionVectorSize;
setting.attNetDet0.minNumDetectionPerClass      = 0;
setting.attNetDet0.weightDirection              = 0.5;
setting.attNetMrg0.mergingOverlap               = 0.8; 
setting.attNetMrg0.mergingType                  = 'NMS';
setting.attNetMrg0.mergingMethod                = 'MAX';
setting.attNetMrg0.minimumNumSupportBox         = 1; 
setting.attNetMrg0.classWiseMerging             = true;
setting.attNetDet1.type                         = 'STATIC';
setting.attNetDet1.rescaleBox                   = 2.5; 
setting.attNetDet1.onlyTargetAndBackground      = true;
setting.attNetDet1.directionVectorSize          = 15;
setting.attNetDet1.minNumDetectionPerClass      = 1;
setting.attNetDet1.weightDirection              = setting.attNetDet0.weightDirection;
setting.attNetMrg1.mergingOverlap               = 0.5;
setting.attNetMrg1.mergingType                  = 'OV';
setting.attNetMrg1.mergingMethod                = 'MAX';
setting.attNetMrg1.minimumNumSupportBox         = 0;
setting.attNetMrg1.classWiseMerging             = true;

%% DO THE JOB.
reset( gpuDevice( setting.gpus ) );
db = Db( setting.db, path.dstDir );
db.genDb;
io = InOut( db, setting.io.tsDb, setting.io.net, setting.io.general );
io.init;
io.makeTsDb;
net = Net( io, setting.net );
net.init;
net.train( setting.gpus, 'visionresearchreport@gmail.com' );
net.fetchBestNet;
[ net, netName ] = net.provdNet;
net.name = netName;
net.normalization.averageImage = io.rgbMean;
attNet = AttNet( ...
    db, net, ...
    setting.attNetProp, ...
    setting.attNetDet0, ...
    setting.attNetMrg0, ...
    setting.attNetDet1, ...
    setting.attNetMrg1 );
attNet.init( setting.gpus );

%% DEMO.
clc; close all;
rng( 'shuffle' );
iid = db.getTeiids;
iid = randsample( iid', 1 );
attNet.demoDet( iid, true );
