# Saved exact row-index splits; identical SCR test records in both protocols.
script <- sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
source(file.path(dirname(normalizePath(script)),'common.R'))
args <- commandArgs(TRUE)
if(!length(args)%in%c(2L,3L)) stop('Usage: Rscript R/06_event_monte_carlo.R PRIVATE_training.csv NEW_output_directory [repetitions=50]')
d <- read_training(args[1]);out <- new_output(args[2])
repetitions <- if(length(args)==3L)as.integer(args[3]) else 50L
stopifnot(length(repetitions)==1L,!is.na(repetitions),repetitions>=1L,repetitions<=50L)
splits <- readRDS(file.path(repo_root,'config/event_monte_carlo_indices.rds'));results <- list()
for(r in seq_len(repetitions)) {
  split <- splits[[r]];test <- split$SCR_test
  for(protocol in c('strict','nonSCR_retained')) {
    train <- if(protocol=='strict')split$strict_train else split$nonSCR_retained_train
    stopifnot(!length(intersect(train,test)))
    model <- fit_model(d[train,],seed=split$seed);p <- as.numeric(predict(model,newdata=d[test,]))
    results[[length(results)+1L]] <- data.frame(repetition=r,protocol=protocol,score(d[[target]][test],p))
    write.csv(data.frame(record_id=d$record_id[test],observed_pctg=d[[target]][test],predicted_pctg=p),
      file.path(out,sprintf('%s_%03d_predictions.csv',protocol,r)),row.names=FALSE)
    rm(model);gc(verbose=FALSE)
  }
  write.csv(do.call(rbind,results),file.path(out,'paired_repetition_metrics.csv'),row.names=FALSE)
  cat('Completed paired repetition',r,'of',repetitions,'\n');flush.console()
}
cat('Split percentiles are NOT confidence intervals. Retained non-SCR data can share test earthquakes.\n')
