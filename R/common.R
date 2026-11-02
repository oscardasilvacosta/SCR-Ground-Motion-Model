# Shared definitions: distances km, Vs30 m/s, PGA already percent g.
script_path <- function() {
  arg <- grep('^--file=',commandArgs(trailingOnly=FALSE),value=TRUE)
  if(!length(arg)) stop('Run this workflow with Rscript.')
  normalizePath(sub('^--file=','',arg[1]),winslash='/',mustWork=TRUE)
}
repo_root <- dirname(dirname(script_path()))
features <- c('Mw','event_depth_km','hypocentral_distance_km','azimuth_cos',
  'event_latitude','event_longitude','station_latitude','vs30_m_s','epicentral_distance_km')
target <- 'rotD100_pga_g'
parameters <- data.frame(nrounds=1200,max_depth=14,eta=0.01,gamma=0,
  colsample_bytree=0.8,min_child_weight=1,subsample=0.9)

require_model_packages <- function() {
  for(pkg in c('caret','xgboost')) if(!requireNamespace(pkg,quietly=TRUE)) stop('Install the R package: ',pkg)
  suppressPackageStartupMessages({library(caret);library(xgboost)})
}
check_numeric <- function(d,columns) {
  missing <- setdiff(columns,names(d))
  if(length(missing)) stop('Missing columns: ',paste(missing,collapse=', '))
  if(!all(vapply(d[columns],is.numeric,logical(1))) || !all(is.finite(as.matrix(d[columns]))))
    stop('Required inputs must be finite numeric values.')
}
geometry <- function(d,derive_missing_distances=FALSE) {
  check_numeric(d,c('event_latitude','event_longitude','station_latitude','station_longitude'))
  if(any(abs(d$event_latitude)>90 | abs(d$station_latitude)>90) ||
     any(abs(d$event_longitude)>180 | abs(d$station_longitude)>180)) stop('Invalid geographic coordinates.')
  a <- d$event_latitude*pi/180;b <- d$station_latitude*pi/180
  dl <- (d$station_longitude-d$event_longitude)*pi/180
  east <- sin(dl)*cos(b)
  north <- cos(a)*sin(b)-sin(a)*cos(b)*cos(dl)
  if(any(east^2+north^2<1e-28)) stop('Initial azimuth is undefined at coincident or antipodal points.')
  # Counterclockwise from east, not cosine of a clockwise-from-north bearing.
  d$azimuth_cos <- cos(atan2(north,east))
  if(derive_missing_distances && !'epicentral_distance_km'%in%names(d)) {
    q <- pmin(1,pmax(0,sin((b-a)/2)^2+cos(a)*cos(b)*sin(dl/2)^2))
    d$epicentral_distance_km <- 6371.0088*2*atan2(sqrt(q),sqrt(1-q))
  }
  if(derive_missing_distances && !'hypocentral_distance_km'%in%names(d)) {
    check_numeric(d,c('epicentral_distance_km','event_depth_km'))
    d$hypocentral_distance_km <- sqrt(d$epicentral_distance_km^2+d$event_depth_km^2)
  }
  d
}
read_training <- function(path) {
  d <- read.csv(path,check.names=FALSE,stringsAsFactors=FALSE)
  check_numeric(d,c('record_id',target,features,'station_longitude'))
  if(nrow(d)!=9579L || !identical(as.integer(d$record_id),seq_len(9579L)))
    stop('Exact reproduction requires the retained 9,579 records in canonical record_id order.')
  if(any(d$vs30_m_s<=0 | d$epicentral_distance_km<=0 | d$hypocentral_distance_km<=0 | d$event_depth_km<0))
    stop('Invalid physical input values.')
  if(max(abs(geometry(d)$azimuth_cos-d$azimuth_cos))>1e-9) stop('Training azimuth convention differs.')
  d
}
primary_partition <- function() {
  d <- read.csv(file.path(repo_root,'config/record_partition.csv'),stringsAsFactors=FALSE)
  stopifnot(nrow(d)==9579L,identical(d$record_id,seq_len(9579L)))
  train <- which(d$split=='train');train <- train[order(d$training_position[train])]
  test <- which(d$split=='SCR_validation')
  stopifnot(length(train)==8236L,length(test)==1343L,
    all(d$analysis_group[test]=='SCR'),sum(d$analysis_group=='SCR')==4479L)
  list(train=train,test=test,metadata=d)
}
score <- function(y,p) {
  stopifnot(length(y)==length(p),length(y)>1L,all(is.finite(y)),all(is.finite(p)))
  sst <- sum((y-mean(y))^2)
  data.frame(n=length(y),R2_SSE=if(sst>0)1-sum((p-y)^2)/sst else NA_real_,
    R2_Pearson_squared=if(sd(y)>0 && sd(p)>0)cor(y,p)^2 else NA_real_,
    MAE_pctg=mean(abs(p-y)),RMSE_pctg=sqrt(mean((p-y)^2)),bias_pctg=mean(p-y),n_nonpositive=sum(p<=0))
}
new_output <- function(path) {
  if(file.exists(path) || dir.exists(path)) stop('Choose a NEW output path: ',path)
  dir.create(path,recursive=TRUE)
  normalizePath(path,winslash='/',mustWork=TRUE)
}
fit_model <- function(d,seed=123L,evaluation_final_seed=NULL,threads=4L) {
  require_model_packages();set.seed(seed)
  control <- if(is.null(evaluation_final_seed))caret::trainControl(method='none',allowParallel=FALSE) else
    caret::trainControl(method='none',seeds=list(as.integer(evaluation_final_seed)),allowParallel=FALSE)
  caret::train(reformulate(features,response=target),data=d,method='xgbTree',preProcess=c('center','scale'),
    trControl=control,tuneGrid=parameters,nthread=threads,verbosity=0)
}
inference_model <- function(model) {
  # Rebuild from native trees to discard training callbacks and captured environments.
  complete <- xgboost::xgb.Booster.complete(model$finalModel)
  booster <- xgboost::xgb.load.raw(xgboost::xgb.save.raw(complete),as_booster=TRUE)
  for(n in c('niter','params','feature_names','nfeatures','xNames','problemType','tuneValue','obsLevels','param'))
    if(!is.null(model$finalModel[[n]])) booster[[n]] <- model$finalModel[[n]]
  pp <- model$preProcess[c('mean','std','method','dim')];class(pp) <- class(model$preProcess)
  terms <- model$terms;attr(terms,'.Environment') <- baseenv()
  result <- list(method='xgbTree',modelInfo=caret::getModelInfo('xgbTree',regex=FALSE)[[1]],
    modelType='Regression',bestTune=model$bestTune,finalModel=booster,preProcess=pp,
    terms=terms,coefnames=model$coefnames,xlevels=list(),
    control=list(method='none',predictionBounds=c(FALSE,FALSE)),
    release_metadata=list(model='XGB2.9G',training_records_removed=TRUE,target_units='percent g'))
  class(result) <- c('train','train.formula');result
}
save_fit <- function(model,output,name) {
  saveRDS(inference_model(model),file.path(output,paste0(name,'.rds')))
  write.csv(data.frame(Feature=features,center=unname(model$preProcess$mean[features]),
    scale=unname(model$preProcess$std[features])),file.path(output,paste0(name,'_preprocessing.csv')),row.names=FALSE)
  capture.output(sessionInfo(),file=file.path(output,'R_sessionInfo.txt'))
}
