# Optional NEW experiment, not reconstruction of the unarchived historical search.
script <- sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
source(file.path(dirname(normalizePath(script)),'common.R'))
args <- commandArgs(TRUE)
if(length(args)!=3L) stop('Usage: Rscript R/09_grid_search.R PRIVATE_training.csv candidate_grid.csv NEW_output_directory')
d <- read_training(args[1]);split <- primary_partition();out <- new_output(args[3]);require_model_packages()
grid <- read.csv(args[2],check.names=FALSE);check_numeric(grid,names(parameters))
if(nrow(grid)<2L) stop('Supply multiple candidate combinations for a grid search.')
set.seed(123);control <- caret::trainControl(method='cv',number=5,allowParallel=FALSE)
model <- caret::train(reformulate(features,response=target),data=d[split$train,],method='xgbTree',
  preProcess=c('center','scale'),trControl=control,tuneGrid=grid,nthread=4,verbosity=0)
write.csv(model$results,file.path(out,'five_fold_search_results.csv'),row.names=FALSE)
write.csv(model$bestTune,file.path(out,'selected_parameters.csv'),row.names=FALSE)
cat('New training-only five-fold search. Accepted published models are not replaced.\n')
