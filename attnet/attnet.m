%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpus                                    = 1;
setting.db                                      = path.db.voc2007;
setting.io.tsDb.numScaling                      = 24;
setting.io.tsDb.dilate                          = 1 / 4;
setting.io.tsDb.normalizeImageMaxSide           = 500;
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
setting.prop.numScaling                         = 24; 48; 
setting.prop.dilate                             = 1 / 2;
setting.prop.normalizeImageMaxSide              = setting.io.tsDb.normalizeImageMaxSide;
setting.prop.posIntOverRegnMoreThan             = 1 / 8;
setting.det0.main.rescaleBox                    = 1;
setting.det0.main.directionVectorSize           = 30; 15; 
setting.det0.main.numMaxTest                    = 50; 
setting.det0.post.mergingOverlap                = 1; 0.85; 
setting.det0.post.mergingType                   = 'OV';
setting.det0.post.mergingMethod                 = 'WAVG';
setting.det0.post.minimumNumSupportBox          = 0; 1; 

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
prop = Prop( db, net, setting.prop );
prop.init( setting.gpus );
det0 = AttNet( db, net, prop, setting.det0.main, setting.det0.post );
det0.init( setting.gpus );