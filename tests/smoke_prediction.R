script <- sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
source(file.path(dirname(dirname(normalizePath(script))),'R/common.R'))
require_model_packages()
for(path in list.files(file.path(repo_root,'R'),pattern='\\.R$',full.names=TRUE)) invisible(parse(file=path))
models <- lapply(c('FULL_DB','EVALUATION'),function(tag)readRDS(file.path(repo_root,'models',paste0('XGB2.9G_',tag,'.rds'))))
for(model in models) {
  stopifnot(identical(model$coefnames,features),is.null(model$trainingData),is.null(model$pred),
    is.null(model$resample),is.null(model$finalModel$callbacks),is.null(model$preProcess$data),
    identical(attr(model$terms,'.Environment'),baseenv()))
}
sites <- read.csv(file.path(repo_root,'data/sites/Lisbon_Vs30_prediction_grid.csv'))
stopifnot(nrow(sites)==1375L,!anyDuplicated(sites$grid_id),min(sites$vs30_m_s)==245,max(sites$vs30_m_s)==754)
events <- read.csv(file.path(repo_root,'config/scenarios.csv'))
max_error <- 0
for(scenario in c('A','C','E','F','G','H')) {
  event <- events[events$SCENARIO==scenario,]
  d <- sites
  d$Mw <- event$MW;d$event_latitude <- event$EVENT_LAT;d$event_longitude <- event$EVENT_LON;d$event_depth_km <- event$DEPTH_KM
  d <- geometry(d,derive_missing_distances=TRUE)
  p <- as.numeric(predict(models[[1]],newdata=d))
  expected <- read.csv(file.path(repo_root,'results/scenarios',paste0(scenario,'_Lisbon_centres_XGB29_GMPE_levels.csv')))
  expected <- expected[match(d$grid_id,expected$SITE_ID),]
  difference <- max(abs(p-expected$XGB_PGA))
  max_error <- max(max_error,difference)
  stopifnot(difference<1e-8)
  if(scenario=='A')stopifnot(abs(mean(p)-25.918177)<1e-6)
}
example <- read.csv(file.path(repo_root,'examples/Scenario_A_complete_model_inputs.csv'))
expected <- read.csv(file.path(repo_root,'results/scenarios/A_Lisbon_centres_XGB29_GMPE_levels.csv'))
p <- as.numeric(predict(models[[1]],newdata=example))
stopifnot(max(abs(p-expected$XGB_PGA[match(example$grid_id,expected$SITE_ID)]))<1e-8)
split <- primary_partition()
stopifnot(length(split$train)==8236L,length(split$test)==1343L)
expect_error <- function(expr) stopifnot(inherits(tryCatch(force(expr),error=function(e)e),'error'))
expect_error(check_numeric(data.frame(Mw=NA_real_),'Mw'))
expect_error(check_numeric(data.frame(Mw=1),'vs30_m_s'))
expect_error(geometry(data.frame(event_latitude=38,event_longitude=-9,station_latitude=38,station_longitude=-9)))
expect_error(geometry(data.frame(event_latitude=100,event_longitude=-9,station_latitude=38,station_longitude=-9)))
cardinal <- geometry(data.frame(event_latitude=c(0,0,0),event_longitude=c(0,0,0),
  station_latitude=c(0,1,0),station_longitude=c(1,0,-1)))
stopifnot(max(abs(cardinal$azimuth_cos-c(1,0,-1)))<1e-12)
stopifnot(score(c(1,2),c(-1,2))$n_nonpositive==1L)
cat('PASS: both stripped models; 8,250 municipal scenario predictions; complete example; split and input checks.\n')
cat('Maximum municipal scenario difference (%g):',max_error,'\n')
