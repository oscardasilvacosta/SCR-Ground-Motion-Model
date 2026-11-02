# Expand the static site grid into coherent event-dependent model inputs.
script <- sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
source(file.path(dirname(normalizePath(script)),'common.R'))
args <- commandArgs(TRUE)
if(length(args)!=4L) stop('Usage: Rscript R/05_predict_scenario_grid.R model.rds grid.csv SCENARIO NEW_predictions.csv')
if(file.exists(args[4])) stop('Output exists; choose a new filename.')
require_model_packages();model <- readRDS(args[1]);stopifnot(identical(model$coefnames,features))
sites <- read.csv(args[2],check.names=FALSE)
check_numeric(sites,c('grid_id','station_latitude','station_longitude','vs30_m_s'))
if(!nrow(sites) || anyDuplicated(sites$grid_id) || any(sites$vs30_m_s<=0)) stop('Invalid site grid.')
events <- read.csv(file.path(repo_root,'config/scenarios.csv'));event <- events[events$SCENARIO==toupper(args[3]),]
if(nrow(event)!=1L) stop('Choose A, C, E, F, G, or H.')
d <- sites[c('grid_id','station_latitude','station_longitude','vs30_m_s')]
d$scenario <- event$SCENARIO;d$Mw <- event$MW;d$event_latitude <- event$EVENT_LAT
d$event_longitude <- event$EVENT_LON;d$event_depth_km <- event$DEPTH_KM
d <- geometry(d,derive_missing_distances=TRUE)
if(event$MW>7.8) warning('Scenario H exceeds the observed training magnitude range.')
d$XGB2_9G_PGA_PCTG <- as.numeric(predict(model,newdata=d))
dir.create(dirname(args[4]),recursive=TRUE,showWarnings=FALSE);write.csv(d,args[4],row.names=FALSE)
cat('Predicted',nrow(d),'site centres for',event$SCENARIO,'in %g, without interpolation or clipping.\n')
