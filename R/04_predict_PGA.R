# Raw inputs in, raw RotD100 PGA in %g out. Do not pre-scale or multiply by 100.
script <- sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
source(file.path(dirname(normalizePath(script)),'common.R'))
args <- commandArgs(TRUE)
if(length(args)!=3L) stop('Usage: Rscript R/04_predict_PGA.R model.rds inputs.csv NEW_predictions.csv')
if(file.exists(args[3])) stop('Output exists; choose a new filename.')
require_model_packages();model <- readRDS(args[1])
if(!identical(model$coefnames,features)) stop('Not the nine-predictor XGB2.9G model.')
d <- read.csv(args[2],check.names=FALSE)
if(!nrow(d)) stop('Input has no records.')
for(name in c('station_latitude','station_longitude')) {
  alias <- if(name=='station_latitude')'LATITUDE' else 'LONGITUDE'
  if(!name%in%names(d) && alias%in%names(d)) d[[name]] <- d[[alias]]
}
check_numeric(d,c('Mw','event_depth_km','vs30_m_s'))
if(any(d$vs30_m_s<=0 | d$event_depth_km<0)) stop('Vs30 must be positive and depth nonnegative.')
supplied <- if('azimuth_cos'%in%names(d))d$azimuth_cos else NULL
d <- geometry(d,derive_missing_distances=TRUE)
if(!is.null(supplied) && (!is.numeric(supplied) || any(!is.finite(supplied)) ||
  max(abs(supplied-d$azimuth_cos))>1e-9)) stop('Supplied azimuth_cos conflicts with the adopted convention.')
check_numeric(d,features)
if(any(d$epicentral_distance_km<=0 | d$hypocentral_distance_km<=0)) stop('Distances must be positive.')
if(any(d$Mw<4.5 | d$Mw>7.8)) warning('Some magnitudes are outside training range 4.5-7.8.')
d$XGB2_9G_PGA_PCTG <- as.numeric(predict(model,newdata=d))
if(any(!is.finite(d$XGB2_9G_PGA_PCTG))) stop('Nonfinite predictions.')
if(any(d$XGB2_9G_PGA_PCTG<=0)) warning('Nonpositive predictions retained without clipping.')
dir.create(dirname(args[3]),recursive=TRUE,showWarnings=FALSE);write.csv(d,args[3],row.names=FALSE)
cat('Saved',nrow(d),'RotD100 PGA predictions in %g.\n')
